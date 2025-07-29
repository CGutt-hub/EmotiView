"""
EmotiView Orchestrator
---------------------
Coordinates data loading, preprocessing, analysis, and reporting for the EmotiView pipeline.
All processing logic is delegated to modules in PsyAnalysisToolbox.
Config-driven, robust, and maintainable.
"""
import os
import sys
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Type, Tuple, Union
from configparser import ConfigParser, NoOptionError, NoSectionError
import pandas as pd
import numpy as np
from mne.io import Raw, RawArray
import matplotlib
matplotlib.use('Agg')

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
toolbox_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'PsyAnalysisToolbox', 'Python'))
for path in [repo_root, toolbox_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# --- Imports from toolbox modules ---
from readers.xdf_reader import XDFReader
from readers.txt_reader import TXTReader
from preprocessors.eeg_preprocessor import EEGPreprocessor
from preprocessors.eda_preprocessor import EDAPreprocessor
from preprocessors.ecg_preprocessor import ECGPreprocessor
from preprocessors.questionnaire_preprocessor import QuestionnairePreprocessor
from preprocessors.fnirs_preprocessor import FNIRSPreprocessor
from processors.fnirs_dm_processor import FNIRSDesignMatrixProcessor
from processors.questionnaire_scale_processor import QuestionnaireScaleProcessor
from processors.ecg_hrv_processor import ECGHRVProcessor
from processors.fai_fnirs_channel_selection_processor import FAIFNIRSChannelSelectionProcessor
from processors.eeg_epoch_processor import EEGEpochProcessor
from processors.mne_event_handling_processor import MNEEventHandlingProcessor
from analyzers.psd_analyzer import PSDAnalyzer
from analyzers.fai_analyzer import FAIAnalyzer
from analyzers.connectivity_analyzer import ConnectivityAnalyzer
from analyzers.glm_analyzer import GLMAnalyzer
from reporters.plot_reporter import PlotReporter
from reporters.log_reporter import LogReporter
from reporters.xml_reporter import XMLReporter
from utils.git_handler import GitHandler
from utils.parallel_runner import ParallelTaskRunner, DAGParallelTaskRunner
from PsyAnalysisToolbox.Python.utils.pipeline_dag import build_participant_dag
from PsyAnalysisToolbox.Python.utils.artifacts import ParticipantArtifacts
from PsyAnalysisToolbox.Python.utils.data_conversion import _create_eeg_mne_raw_from_df, _create_fnirs_mne_raw_from_df
from PsyAnalysisToolbox.Python.utils.logging_utils import log_progress_bar

# =============================================================================
# === Configuration & Component Instantiation
# =============================================================================

# Helper to get nested artifact (works for both dict and ParticipantArtifacts)
def get_nested_artifact(artifacts, *keys):
    if hasattr(artifacts, 'get'):
        obj = artifacts.get(keys[0])
    else:
        obj = artifacts[keys[0]]
    for k in keys[1:]:
        obj = obj[k]
    return obj

def load_configuration(config_path: str) -> ConfigParser:
    """Loads the .cfg file and preserves case of keys."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    config = ConfigParser()
    config.optionxform = lambda optionstr: optionstr
    config.read(config_path)
    return config

def setup_main_logger(config: ConfigParser) -> logging.Logger:
    """Sets up the main logger for the orchestrator."""
    main_logger_name = config.get('DEFAULT', 'main_logger_name', fallback='EmotiViewOrchestrator')
    log_level = config.get('DEFAULT', 'log_level', fallback='INFO').upper()
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(main_logger_name)
    logger.info("Orchestrator logger initialized.")
    return logger

def instantiate_components(config: ConfigParser, logger: logging.Logger, base_output_dir: str) -> Dict[str, Any]:
    """Instantiates and returns all required processing components in a dictionary."""
    components = {
        'xdf_reader': XDFReader(logger=logger,
                                eeg_stream_name=config.get('XDF', 'eeg_stream_name', fallback=None),
                                fnirs_stream_name=config.get('XDF', 'fnirs_stream_name', fallback=None),
                                marker_stream_name=config.get('XDF', 'marker_stream_name', fallback=None)),
        'questionnaire_reader': TXTReader(logger=logger),
        'questionnaire_preprocessor': QuestionnairePreprocessor(logger=logger),
        'questionnaire_scale_processor': QuestionnaireScaleProcessor(logger=logger),
        'eeg_preprocessor': EEGPreprocessor(logger=logger),
        'eda_preprocessor': EDAPreprocessor(logger=logger),
        'ecg_preprocessor': ECGPreprocessor(logger=logger),
        'fnirs_preprocessor': FNIRSPreprocessor(logger=logger),
        'mne_event_handler': MNEEventHandlingProcessor(
            config={
                'conditions_to_map': [c.strip() for c in config.get('Segmentation', 'conditions_to_map', fallback='').split(',') if c.strip()],
                'event_id_map': {k: v for k, v in config.items('EventMapping')}
            },
            logger=logger),
        'eeg_epoch_processor': EEGEpochProcessor(logger=logger),
        'ecg_hrv_processor': ECGHRVProcessor(logger=logger),
        'fai_channel_selector': FAIFNIRSChannelSelectionProcessor(logger=logger),
        'psd_analyzer': PSDAnalyzer(logger=logger),
        'fai_analyzer': FAIAnalyzer(logger=logger),
        'connectivity_analyzer': ConnectivityAnalyzer(logger=logger),
        'fnirs_dm_processor': FNIRSDesignMatrixProcessor(logger=logger,
            hrf_model_config=config.get('FNIRS_DM', 'hrf_model', fallback='glover'),
            drift_model_config=config.get('FNIRS_DM', 'drift_model', fallback='polynomial'),
            drift_order_config=config.getint('FNIRS_DM', 'drift_order', fallback=1)),
        'glm_analyzer': GLMAnalyzer(logger=logger),
        'xml_reporter': XMLReporter(logger=logger),
        'plot_reporter': PlotReporter(logger=logger, output_dir_base=base_output_dir, reporting_figure_format_config=config.get('Plotting', 'reporting_figure_format', fallback='png'), reporting_dpi_config=config.getint('Plotting', 'reporting_dpi', fallback=100)),
        'git_handler': GitHandler(logger=logger, repository_path=config.get('Git', 'repository_path'), default_remote_name='origin')
    }
    logger.info(f"{len(components)} components instantiated.")
    return components

# =============================================================================
# === 2. Participant-Level Helper Functions
# =============================================================================

def _report_data_quality(streams: Dict[str, Any], p_logger: logging.Logger):
    """Inspects loaded DataFrames for potential quality issues like NaNs."""
    p_logger.info("--- Running Data Quality Checks on Loaded Streams ---")
    for stream_name, stream_data in streams.items():
        if isinstance(stream_data, pd.DataFrame):
            nan_count = stream_data.isnull().sum().sum()
            if nan_count > 0:
                total_cells = stream_data.size
                nan_percentage = (nan_count / total_cells) * 100
                p_logger.warning(
                    f"Data Quality Issue in '{stream_name}': Found {nan_count} NaN "
                    f"values ({nan_percentage:.2f}% of total). This may affect subsequent analyses."
                )

def _load_data(participant_id: str, config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Loads all data for a single participant, handles duplicate/empty streams, and adds data to the artifacts dictionary."""
    
    def _get_stream_len(s: Any) -> int:
        """Helper to safely get the number of samples from various stream data types."""
        if isinstance(s, (Raw, RawArray)):
            return s.n_times
        elif isinstance(s, (pd.DataFrame, np.ndarray)):
            return s.shape[0] if s.shape else 0
        # The hasattr check is a fallback for potential raw pyxdf stream dicts, though less likely with current reader.
        elif hasattr(s, 'times') and s.times is not None:
            return len(s.times)
        return 0

    p_logger.info("--- Step 1: Data Loading ---")
    base_raw_dir = config.get('Data', 'base_raw_data_dir')

    # --- Get configured stream names for mapping and diagnostics ---
    eeg_stream_name_cfg = config.get('XDF', 'eeg_stream_name', fallback=None)
    fnirs_stream_name_cfg = config.get('XDF', 'fnirs_stream_name', fallback=None)
    marker_stream_name_cfg = config.get('XDF', 'marker_stream_name', fallback=None)

    # --- Robust stream mapping: allow fallback to available streams if expected name is missing ---
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def find_best_stream_key(expected, modality_keywords, all_keys):
        # 1. Exact match
        if expected in all_keys:
            return expected, f"exact match"
        # 2. Case-insensitive match
        for key in all_keys:
            if key.lower() == expected.lower():
                return key, f"case-insensitive match to '{expected}'"
        # 3. Partial match
        for key in all_keys:
            if expected.lower() in key.lower() or key.lower() in expected.lower():
                return key, f"partial match to '{expected}'"
        # 4. Fuzzy match (Levenshtein distance <= 2)
        for key in all_keys:
            if levenshtein(key.lower(), expected.lower()) <= 2:
                return key, f"fuzzy match (Levenshtein <=2) to '{expected}'"
        # 5. Modality keyword match (e.g., 'eeg', 'fnirs', 'marker')
        for key in all_keys:
            for kw in modality_keywords:
                if kw in key.lower():
                    return key, f"modality keyword '{kw}' in '{key}'"
        return None, "no match"

    # --- 1. Load
    xdf_path = os.path.join(base_raw_dir, participant_id, f"{participant_id}.xdf")
    if os.path.exists(xdf_path):
        # Load all streams from the XDF file.
        all_loaded_streams = components['xdf_reader'].load_participant_streams(participant_id, xdf_path)

        # --- Diagnostic Logging: What did we find? ---
        if isinstance(all_loaded_streams, dict):
            p_logger.info(f"XDF reader found the following streams and metadata keys: {list(all_loaded_streams.keys())}")
            for k, v in all_loaded_streams.items():
                if isinstance(v, pd.DataFrame):
                    p_logger.info(f"Stream '{k}': DataFrame shape {v.shape}, columns: {list(v.columns)}")
                else:
                    p_logger.info(f"Stream '{k}': type {type(v)}")
        else:
            p_logger.error(f"XDFReader did not return a dictionary of streams as expected. Got {type(all_loaded_streams)}. Cannot process streams.")
            return

        all_stream_keys = list(all_loaded_streams.keys())
        # --- Robust mapping logic: this is the ONLY valid place for such usage ---
        eeg_key, eeg_reason = find_best_stream_key(
            eeg_stream_name_cfg, ["eeg"], all_stream_keys)
        p_logger.info(f"EEG stream mapping: expected '{eeg_stream_name_cfg}', found '{eeg_key}' ({eeg_reason}). Candidates: {all_stream_keys}")
        fnirs_key, fnirs_reason = find_best_stream_key(
            fnirs_stream_name_cfg, ["fnirs", "nir"], all_stream_keys)
        p_logger.info(f"fNIRS stream mapping: expected '{fnirs_stream_name_cfg}', found '{fnirs_key}' ({fnirs_reason}). Candidates: {all_stream_keys}")
        marker_key, marker_reason = find_best_stream_key(
            marker_stream_name_cfg, ["marker"], all_stream_keys)
        p_logger.info(f"Marker stream mapping: expected '{marker_stream_name_cfg}', found '{marker_key}' ({marker_reason}). Candidates: {all_stream_keys}")

        stream_name_to_standard_key_map = {}
        if eeg_key:
            stream_name_to_standard_key_map[eeg_key] = 'eeg_df'
        else:
            p_logger.error(f"No suitable EEG stream found. EEG workflow will be skipped.")
        if fnirs_key:
            stream_name_to_standard_key_map[fnirs_key] = 'fnirs_od_df'
        else:
            p_logger.error(f"No suitable fNIRS stream found. fNIRS workflow will be skipped.")
        if marker_key:
            stream_name_to_standard_key_map[marker_key] = 'xdf_markers_df'
        else:
            p_logger.error(f"No suitable marker stream found. Marker-based epoching will be skipped.")

        # --- Diagnostic Logging: What are we looking for? ---
        p_logger.info(f"Configuration check: Looking for EEG stream named '{eeg_stream_name_cfg}'.")
        p_logger.info(f"Configuration check: Looking for Marker stream named '{marker_stream_name_cfg}'.")
        p_logger.info(f"Configuration check: Looking for fNIRS stream named '{fnirs_stream_name_cfg}'.")

        # --- Separate metadata from data streams and filter empty/invalid streams ---
        non_empty_streams = {}
        if isinstance(all_loaded_streams, dict):
            # The reader returns a mix of data (DataFrames, Raw) and metadata (floats for start times).
            # We need to separate them first.
            data_streams_from_reader = {}
            
            # Construct the exact keys we expect for metadata based on the config
            eeg_start_time_key = f"{eeg_stream_name_cfg}_start_time_xdf" if eeg_stream_name_cfg else None
            fnirs_start_time_key = f"{fnirs_stream_name_cfg}_start_time_xdf" if fnirs_stream_name_cfg else None

            for key, value in all_loaded_streams.items():
                # Heuristic: if it's not a data container, it's metadata.
                if not isinstance(value, (pd.DataFrame, Raw, RawArray, np.ndarray)):
                    # --- Standardize metadata keys for downstream use ---
                    if eeg_start_time_key and key == eeg_start_time_key:
                        standard_key = 'eeg_stream_start_time_xdf'
                        artifacts[standard_key] = value
                        p_logger.info(f"Stored and standardized metadata: '{key}' -> '{standard_key}'")
                    elif fnirs_start_time_key and key == fnirs_start_time_key:
                        standard_key = 'fnirs_stream_start_time_xdf'
                        artifacts[standard_key] = value
                        p_logger.info(f"Stored and standardized metadata: '{key}' -> '{standard_key}'")
                    else:
                        artifacts[key] = value
                        p_logger.info(f"Stored unmapped metadata from XDF reader: '{key}' = '{value}'")
                else:
                    data_streams_from_reader[key] = value

            # Group streams by name to handle duplicates
            # This is necessary for XDF files that may contain multiple streams with the exact same name,
            # as shown in the xdf_inspector_log.txt for EV_007.
            from collections import defaultdict
            streams_by_name = defaultdict(list)
            for stream_name, stream_data in data_streams_from_reader.items():
                streams_by_name[stream_name].append(stream_data)

            for stream_name, stream_list in streams_by_name.items():
                if len(stream_list) > 1:
                    p_logger.warning(f"Found {len(stream_list)} duplicate streams named '{stream_name}'. Selecting the one with the most data points.")
                    # Select the stream with the most samples/events
                    best_stream = max(stream_list, key=_get_stream_len)
                    stream_data = best_stream
                else:
                    stream_data = stream_list[0]

                # Check if the selected stream is empty
                is_empty = stream_data is None or \
                           (isinstance(stream_data, pd.DataFrame) and stream_data.empty) or \
                           (isinstance(stream_data, Raw) and stream_data.n_times == 0) or \
                           (isinstance(stream_data, np.ndarray) and stream_data.ndim == 0) # Scalar numpy arrays are not valid streams.
                
                if not is_empty:
                    # --- Map to standardized key if it's a known stream type ---
                    standard_key = stream_name_to_standard_key_map.get(stream_name)
                    if standard_key:
                        non_empty_streams[standard_key] = stream_data
                        p_logger.info(f"Found and mapped valid stream '{stream_name}' to standard key '{standard_key}'.")
                    else:
                        # Keep other streams under their original name if needed, or just log and ignore.
                        non_empty_streams[stream_name] = stream_data
                        p_logger.info(f"Found valid, non-empty stream: '{stream_name}' (unmapped, kept original name).")
                    sample_count = _get_stream_len(stream_data)
                else:
                    p_logger.warning(f"Stream '{stream_name}' is empty or invalid after selection. It will be ignored.")

        artifacts['xdf_streams'] = non_empty_streams
        p_logger.info(f"Final standardized stream keys available after loading and mapping: {list(non_empty_streams.keys())}")

        # Run data quality checks on the streams we decided to keep.
        _report_data_quality(non_empty_streams, p_logger)

        # --- Convert loaded DataFrames to MNE Raw objects where needed ---
        if 'eeg_df' in non_empty_streams and isinstance(non_empty_streams['eeg_df'], pd.DataFrame):
            eeg_mne_artifacts = _create_eeg_mne_raw_from_df(non_empty_streams['eeg_df'], config, p_logger)
            if eeg_mne_artifacts and isinstance(eeg_mne_artifacts.get('eeg_mne_raw'), (Raw, RawArray)):
                non_empty_streams['eeg'] = eeg_mne_artifacts.pop('eeg_mne_raw')
                artifacts.update(eeg_mne_artifacts)
                p_logger.info('EEG DataFrame successfully converted to MNE Raw for preprocessing.')
            else:
                p_logger.error('Failed to create MNE Raw object from EEG DataFrame. EEG analysis will be skipped.')
                del non_empty_streams['eeg_df']
        
        if 'fnirs_od_df' in non_empty_streams and isinstance(non_empty_streams['fnirs_od_df'], pd.DataFrame):
            if config.getboolean('ProcessingSwitches', 'process_fnirs', fallback=True):
                fnirs_mne_artifacts = _create_fnirs_mne_raw_from_df(
                    non_empty_streams['fnirs_od_df'], config, p_logger)
                if fnirs_mne_artifacts:
                    non_empty_streams['fnirs_cw_amplitude'] = fnirs_mne_artifacts.pop('fnirs_mne_raw')
                    artifacts.update(fnirs_mne_artifacts)
                else:
                    p_logger.error("Failed to create MNE Raw object from fNIRS DataFrame. fNIRS analysis will be skipped.")
                    # This key might not exist if loading failed, so check first
                    if 'fnirs_od_df' in non_empty_streams:
                        del non_empty_streams['fnirs_od_df']
            else:
                p_logger.info("Skipping fNIRS MNE object creation because 'process_fnirs' is set to False in config.")
                # Also remove the raw dataframe so it's not processed later
                if 'fnirs_od_df' in non_empty_streams:
                    del non_empty_streams['fnirs_od_df']
        else:
            p_logger.info("fNIRS data not found or switch is off. Skipping fNIRS MNE object creation.")

        # --- Final Diagnostic Check for Critical Streams ---
        if eeg_stream_name_cfg and 'eeg' not in non_empty_streams: p_logger.warning(f"POST-LOADING CHECK FAILED: The configured EEG stream ('{eeg_stream_name_cfg}') was not successfully loaded and converted to an MNE object. Epoching will be skipped.")
        if marker_stream_name_cfg and 'xdf_markers_df' not in non_empty_streams: p_logger.warning(f"POST-LOADING CHECK FAILED: The configured marker stream ('{marker_stream_name_cfg}') was not found or was empty. Epoching will be skipped.")

        # Process streams if loading succeeded
        # This block extracts auxiliary channels (ECG, EDA) from the main EEG DataFrame
        # and adds them back into the streams dictionary so they can be preprocessed.
        streams = artifacts.get('xdf_streams')
        if streams and 'eeg_df' in streams:
            eeg_df = get_nested_artifact(streams, 'eeg_df')
            df_columns = list(eeg_df.columns)

            # Use config keys for channel names if present
            ecg_channel_name = config.get('ChannelManagement', 'ecg_channel_name', fallback=None)
            eda_channel_name = config.get('ChannelManagement', 'eda_channel_name', fallback=None)
            trigger_channel_name = config.get('ChannelManagement', 'eeg_trigger_channel_name', fallback=None)

            # Fallback to auto-detection if config keys are missing or not found in columns
            def find_column_by_keywords(keywords):
                for col in df_columns:
                    for kw in keywords:
                        if kw.lower() in col.lower():
                            return col
                return None

            if not ecg_channel_name or ecg_channel_name not in df_columns:
                ecg_channel_name = find_column_by_keywords(['ecg', 'ekg'])
            if not eda_channel_name or eda_channel_name not in df_columns:
                eda_channel_name = find_column_by_keywords(['eda'])
            if not trigger_channel_name or trigger_channel_name not in df_columns:
                trigger_channel_name = find_column_by_keywords(['trigger'])

            # EEG channels
            eeg_channel_names_cfg = config.get('ChannelManagement', 'eeg_channel_names', fallback='').strip()
            if eeg_channel_names_cfg:
                eeg_channel_names = [ch.strip() for ch in eeg_channel_names_cfg.split(',') if ch.strip()]
            else:
                import re
                exclude_keywords = ['ecg', 'ekg', 'eda', 'trigger', 'time']
                eeg_channel_names = [col for col in df_columns if not any(kw in col.lower() for kw in exclude_keywords) and re.match(r'^[A-Za-z0-9_\-]+$', col)]
            # Extraction as before
            if trigger_channel_name and trigger_channel_name in eeg_df.columns:
                artifacts['trigger_stream'] = eeg_df[trigger_channel_name].copy()
            if ecg_channel_name and ecg_channel_name in eeg_df.columns:
                artifacts['ecg_df'] = pd.DataFrame({
                    'ecg_signal': eeg_df[ecg_channel_name],
                    'time_xdf': eeg_df['time_xdf'] if 'time_xdf' in eeg_df.columns else np.nan,
                    'time_sec': eeg_df['time_sec'] if 'time_sec' in eeg_df.columns else np.nan
                })
            if eda_channel_name and eda_channel_name in eeg_df.columns:
                artifacts['eda_df'] = pd.DataFrame({
                    'eda_signal': eeg_df[eda_channel_name],
                    'time_xdf': eeg_df['time_xdf'] if 'time_xdf' in eeg_df.columns else np.nan,
                    'time_sec': eeg_df['time_sec'] if 'time_sec' in eeg_df.columns else np.nan
                })
            if eeg_channel_names:
                eeg_only_df = eeg_df[eeg_channel_names]
                streams['eeg_df'] = eeg_only_df
                p_logger.info(f"EEG DataFrame now contains only true EEG channels: {eeg_channel_names}")
            else:
                p_logger.error("No valid EEG channels found in EEG DataFrame after filtering. EEG analysis will be skipped.")
                streams['eeg_df'] = pd.DataFrame()  # Empty DataFrame to prevent downstream errors

        # The EEG and fNIRS channel names are now stored in artifacts during the MNE object conversion.
        # Check if streams dict exists before checking for a key to prevent a TypeError.
        if not streams or 'eeg' not in streams:
            p_logger.warning("EEG stream (as MNE Raw) not available after loading.")
        # Detach the trigger channel immediately after loading raw EEG DataFrame
        trigger_ch_name = config.get('ChannelManagement', 'eeg_trigger_channel_name', fallback=None)
        if streams is not None and isinstance(streams, dict) and 'eeg_df' in streams and trigger_ch_name and trigger_ch_name in streams['eeg_df'].columns:
            artifacts['trigger_stream'] = streams['eeg_df'][trigger_ch_name].copy()
            p_logger.info(f"Detached unmodified trigger stream '{trigger_ch_name}' after loading raw EEG data.")

        # Load Questionnaire data
        try:
            q_file_name = config.get('QuestionnaireReader', 'filename_template').format(participant_id=participant_id)
            q_path = os.path.join(base_raw_dir, participant_id, q_file_name)
            if os.path.exists(q_path):
                # Handle special characters for delimiter from config file
                delimiter_from_config = config.get('QuestionnaireReader', 'delimiter')
                if delimiter_from_config == '\\t':
                    actual_delimiter = '\t'
                    p_logger.info("Interpreted config delimiter '\\t' as a tab character.")
                else:
                    actual_delimiter = delimiter_from_config

                reader_type_from_config = config.get('QuestionnaireReader', 'reader_type', fallback='tabular')
                p_logger.info(f"Questionnaire reader type from config: '{reader_type_from_config}'")

                artifacts['questionnaire_raw_df'] = components['questionnaire_reader'].load_data(
                    file_path=q_path,
                    reader_type=reader_type_from_config,
                    file_type=config.get('QuestionnaireReader', 'file_type'), # file_type is still useful for tabular
                    delimiter=actual_delimiter,
                    encoding=config.get('QuestionnaireReader', 'encoding'))
                p_logger.info("Questionnaire raw data loaded.")
            else:
                p_logger.warning(f"Questionnaire file not found: {q_path}")
        except (NoSectionError, NoOptionError) as e:
            p_logger.warning(f"Questionnaire configuration missing in config file: {e}. Skipping questionnaire loading.")

def _create_events_from_xdf_markers(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Attempts to create an MNE events DataFrame from a separate XDF marker stream."""
    p_logger.info("Attempting to create events from separate XDF marker stream...")
    
    def _clean_marker_value(marker_val: Any) -> str:
        """Helper to robustly clean a marker value from pyxdf, which can be a list, bytes, or string."""
        s = ""
        if isinstance(marker_val, list) and marker_val:
            s = str(marker_val[0])
        elif isinstance(marker_val, bytes):
            s = marker_val.decode('utf-8', errors='ignore')
        else:
            s = str(marker_val)
        
        s = s.strip().upper()

        if 'POS' in s: return 'POS'
        if 'NEG' in s: return 'NEG'
        if 'NEU' in s: return 'NEU'
        
        import re
        numbers = re.findall(r'\d+', s)
        if numbers:
            return numbers[-1] # Return the last number found
        return s # Return the cleaned string if no numbers found

    has_markers = 'xdf_streams' in artifacts and 'xdf_markers_df' in artifacts['xdf_streams']
    eeg_start_time = artifacts.get('eeg_stream_start_time_xdf')
    eeg_raw = artifacts.get('eeg_processed_raw')

    if not has_markers or eeg_start_time is None or eeg_raw is None:
        p_logger.info("Skipping event creation from XDF markers: missing required data (marker stream, EEG start time, or raw EEG object).")
        return None
            
    try:
        raw_markers_df = get_nested_artifact(artifacts, 'xdf_streams', 'xdf_markers_df')
        if raw_markers_df is None:
            p_logger.error("No xdf_markers_df found in artifacts. Cannot create events from XDF markers.")
            return None
        p_logger.info(f"DIAGNOSTIC: Raw marker DataFrame has {raw_markers_df.shape[0]} rows.")
        unique_raw_markers = raw_markers_df['marker_value'].unique()
        p_logger.info(f"DIAGNOSTIC: Raw unique marker values: {unique_raw_markers}")

        prepared_markers_df = raw_markers_df.copy()
        prepared_markers_df['cleaned_marker'] = prepared_markers_df['marker_value'].apply(_clean_marker_value)
        
        raw_to_cleaned_map = pd.Series(prepared_markers_df.cleaned_marker.values, index=prepared_markers_df.marker_value.astype(str)).to_dict()
        p_logger.info(f"DIAGNOSTIC: Raw-to-Cleaned marker mapping: {raw_to_cleaned_map}")

        prepared_markers_df['onset_time_sec'] = prepared_markers_df['timestamp'] - eeg_start_time
        prepared_markers_df.rename(columns={'cleaned_marker': 'condition'}, inplace=True)
        
        prepared_markers_df['trial_identifier_eprime'] = 'N/A'

        events_df = components['mne_event_handler'].create_events_df(
            events_df=prepared_markers_df,
            sfreq=eeg_raw.info['sfreq']
        )
        
        if events_df is not None and not events_df.empty:
            p_logger.info(f"Successfully processed {len(events_df)} events from XDF marker stream.")
            return events_df
        else:
            p_logger.warning("Processing XDF markers resulted in an empty event list.")
            return None
    except Exception as e:
        p_logger.error(f"Error processing XDF markers: {e}", exc_info=True)
        return None

def _create_events_from_analog_trigger(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Creates MNE events by detecting changes in a digital trigger channel (not analog thresholding).
    This is the correct method for digital triggers encoded as event codes.
    """
    p_logger.info("Attempting to create events from digital trigger channel (TriggerStream)...")
    trigger_ch_name = config.get('ChannelManagement', 'eeg_trigger_channel_name', fallback=None)
    trigger_data = artifacts.get('trigger_stream', None)
    eeg_raw = artifacts.get('eeg_processed_raw')
    if trigger_data is None:
        p_logger.error(f"No unmodified trigger stream '{trigger_ch_name}' found in artifacts. Cannot create events from digital trigger.")
        return None
    if eeg_raw is None:
        p_logger.warning("Skipping digital trigger processing: Raw EEG not found.")
        return None
    try:
        sfreq = eeg_raw.info['sfreq']
        trigger_data = trigger_data.to_numpy() if hasattr(trigger_data, 'to_numpy') else np.array(trigger_data)
        # Find all indices where the trigger value changes (event onsets)
        onsets = np.where(np.diff(trigger_data, prepend=trigger_data[0]) != 0)[0]
        event_ids = (trigger_data[onsets] / 1_000_000)
        # Convert to plain Python ints
        event_ids = [int(v) for v in event_ids]
        p_logger.info(f"DIAGNOSTIC: Found {len(onsets)} event onsets in digital trigger channel.")
        p_logger.info(f"DIAGNOSTIC: Extracted event IDs from trigger stream: {sorted(set(event_ids))}")
        if len(onsets) == 0:
            p_logger.warning("No trigger events found in digital trigger channel.")
            return None
        events_df = pd.DataFrame({
            'onset_sample': onsets,
            'previous_event_id': np.zeros_like(onsets),
            'condition': event_ids
        })
        events_df_out = components['mne_event_handler'].create_events_df(events_df, sfreq)
        if events_df_out is None or events_df_out.empty:
            p_logger.error("[ERROR] No valid events created from digital trigger channel after mapping.")
        else:
            p_logger.info(f"[INFO] Created {len(events_df_out)} events from digital trigger channel.")
        return events_df_out
    except Exception as e:
        p_logger.error(f"Error processing digital trigger channel: {e}", exc_info=True)
        print(f"[ERROR] Exception in processing digital trigger channel: {e}")
        return None

def _create_events_from_stim_channel(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Attempts to create an MNE events DataFrame from an embedded stim channel."""
    p_logger.info("Attempting to create events from embedded 'stim' channel...")
    events_from_stim = artifacts.get('eeg_events_from_stim')
    eeg_raw = artifacts.get('eeg_processed_raw')

    if events_from_stim is None or events_from_stim.shape[0] == 0 or eeg_raw is None:
        p_logger.error("No stim channel events or EEG raw data found. Cannot create events from stim channel.")
        print("[ERROR] No stim channel events or EEG raw data found. Cannot create events from stim channel.")
        return None

    try:
        p_logger.info(f"Found {len(events_from_stim)} raw events in 'stim' channel.")
        events_df = pd.DataFrame({
            'sample': events_from_stim[:, 0],
            'previous_event_id': events_from_stim[:, 1],
            'event_id': events_from_stim[:, 2]
        })
        # Robust mapping from event_id (1-9) to condition names using config
        event_id_map = {int(k): v for k, v in config.items('EventMapping') if k.isdigit()}
        events_df['event_id'] = events_df['event_id'].astype(int)
        events_df['condition_name'] = events_df['event_id'].map(lambda x: event_id_map.get(x, None))
        p_logger.info(f"Event ID to condition mapping: {event_id_map}")
        unmapped_events = events_df[events_df['condition_name'].isnull()]
        if not unmapped_events.empty:
            p_logger.warning(f"[WARNING] Found {len(unmapped_events)} stim channel events with IDs not in config map: {events_df.loc[unmapped_events.index, 'event_id'].unique()}")
            print(f"[WARNING] Found {len(unmapped_events)} stim channel events with IDs not in config map: {events_df.loc[unmapped_events.index, 'event_id'].unique()}")
        conditions_to_map = [c.strip() for c in config.get('Segmentation', 'conditions_to_map', fallback='').split(',') if c.strip()]
        if conditions_to_map:
            events_df.dropna(subset=['condition_name'], inplace=True)
            events_df = events_df[events_df['condition_name'].isin(conditions_to_map)]
        if not events_df.empty:
            p_logger.info(f"Successfully processed {len(events_df)} events from stim channel.")
            print(f"[INFO] Successfully processed {len(events_df)} events from stim channel.")
            # The MNEEventHandlingProcessor expects integer event_ids, so we must re-create them from the condition map
            # This is the reverse of the mapping done by the processor for XDF markers.
            condition_to_id_map = {v: int(k) for k, v in components['mne_event_handler'].event_id_map.items() if k.isdigit()}
            col = events_df['condition_name']
            import numpy as np
            if isinstance(col, np.ndarray):
                col = col.tolist()
            events_df['event_id'] = pd.Series(col, index=events_df.index).map(lambda x: condition_to_id_map.get(x, np.nan))
            events_df = events_df.dropna(axis=0, subset=['event_id']).copy()  # type: ignore[arg-type]
            events_df['event_id'] = events_df['event_id'].astype(int)
            result_df = events_df.loc[:, ['sample', 'previous_event_id', 'event_id', 'condition_name']]
            if not isinstance(result_df, pd.DataFrame):
                result_df = pd.DataFrame(result_df)
            return result_df.copy()
        else:
            p_logger.error("Processing stim channel events resulted in an empty event list after mapping/filtering.")
            print("[ERROR] Processing stim channel events resulted in an empty event list after mapping/filtering.")
            return None
    except Exception as e:
        p_logger.error(f"Error processing stim channel events: {e}", exc_info=True)
        print(f"[ERROR] Exception in processing stim channel events: {e}")
        return None

def _create_mne_events(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """
    Creates MNE-compatible event arrays by trying the most reliable sources first.    Priority: 1. Separate XDF Marker Stream, 2. Analog Trigger, 3. Embedded Stim Channel.
    """
    p_logger.info("--- Creating MNE Events (Priority-Based) ---")
    
    if artifacts.get('eeg_processed_raw') is None:
        p_logger.warning("Skipping MNE event creation: 'eeg_processed_raw' artifact not found.")
        return

    # --- Priority 1: Attempt to create events from separate XDF marker stream ---
    events_from_xdf = _create_events_from_xdf_markers(config, components, p_logger, artifacts)
    if events_from_xdf is not None and not events_from_xdf.empty:
        p_logger.info("Decision: Successfully created events using the separate XDF marker stream.")
        artifacts['mne_events_df'] = events_from_xdf
        return  # Success, we are done.

    # --- Priority 2: Fallback to analog trigger channel ---
    p_logger.warning("Could not create events from XDF marker stream. Falling back to analog trigger channel.")
    events_from_analog = _create_events_from_analog_trigger(config, components, p_logger, artifacts)
    if events_from_analog is not None and not events_from_analog.empty:
        p_logger.info("Decision: Successfully created events using the analog trigger channel (fallback).")
        artifacts['mne_events_df'] = events_from_analog
        return # Success

    # --- Priority 3: Fallback to embedded stim channel ---
    p_logger.warning("Could not create events from analog trigger. Falling back to embedded stim channel.")
    events_from_stim = _create_events_from_stim_channel(config, components, p_logger, artifacts)
    if events_from_stim is not None and not events_from_stim.empty:
        p_logger.info("Decision: Successfully created events using the embedded stim channel (fallback).")
        artifacts['mne_events_df'] = events_from_stim
        return  # Success
    else:
        p_logger.error("Could not find any valid events from any source (XDF markers, analog trigger, or stim channel).")

def _calculate_and_store_durations(df: pd.DataFrame, sfreq: float, fallback_duration: float, p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """
    Calculates variable trial durations from an events DataFrame and stores results in artifacts.
    This is a key step to move from fixed-length to data-driven epoching.
    """
    if df.empty or 'sample' not in df.columns or 'condition_name' not in df.columns:
        p_logger.warning("Cannot calculate durations: input DataFrame is empty or missing required columns.")
        return

    p_logger.info("Calculating variable trial durations based on time between markers...")
    df_sorted = df.sort_values('sample').reset_index(drop=True)
    df_sorted['duration_samples'] = df_sorted['sample'].shift(-1) - df_sorted['sample']

    # For the very last event, duration is NaN. Use the config value as a fallback.
    fallback_samples = fallback_duration * sfreq
    df_sorted['duration_samples'].fillna(fallback_samples, inplace=True)
    p_logger.info(f"Used fallback duration of {fallback_duration:.2f}s for the last event in the recording.")

    # --- Determine epoching window based on shortest trial ---
    min_duration_samples = df_sorted['duration_samples'].min()
    if isinstance(min_duration_samples, pd.Series):
        min_duration_samples = min_duration_samples.iloc[0]
    if pd.isna(min_duration_samples):
        shortest_duration_sec = fallback_duration
    else:
        shortest_duration_sec = min_duration_samples / sfreq
    artifacts['shortest_epoch_duration_sec'] = shortest_duration_sec
    p_logger.info(f"Shortest trial duration found: {shortest_duration_sec:.2f}s. This will be used as the epoch window for all trials to ensure consistency.")

    # --- Calculate average duration per condition for fNIRS GLM ---
    avg_duration_samples = df_sorted.groupby('condition_name')['duration_samples'].mean()
    if isinstance(avg_duration_samples, pd.Series):
        avg_duration_sec_map = (avg_duration_samples.astype(float) / float(sfreq)).to_dict()
    else:
        avg_duration_sec_map = {}
    artifacts['avg_duration_per_condition_sec'] = avg_duration_sec_map
    p_logger.info(f"Average trial durations calculated for fNIRS GLM: {avg_duration_sec_map}")

def _run_epoching(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Handles EEG epoching based on previously created MNE events."""
    p_logger.info("Running EEG epoching...")
    eeg_raw = artifacts.get('eeg_processed_raw')
    events_df = artifacts.get('mne_events_df')
    # --- TYPE CHECK: Only allow MNE objects for epoching ---
    if not isinstance(eeg_raw, (Raw, RawArray)):
        p_logger.error("[ERROR] EEG epoching requires an MNE Raw/RawArray object. Got: {}. Skipping epoching.".format(type(eeg_raw)))
        return
    if events_df is None or not isinstance(events_df, pd.DataFrame) or events_df.empty:
        p_logger.error("[ERROR] Skipping epoching because the MNE events DataFrame is missing or empty. This usually means event creation failed.")
        print("[ERROR] Skipping epoching because the MNE events DataFrame is missing or empty. This usually means event creation failed.")
        return
    try:
        if events_df.empty:
            p_logger.error("[ERROR] Skipping epoching because the MNE events DataFrame is empty.")
            print("[ERROR] Skipping epoching because the MNE events DataFrame is empty.")
            return
        p_logger.info(f"Found {len(events_df)} events to create epochs from.")
        print(f"[INFO] Found {len(events_df)} events to create epochs from.")
        mne_events_array = events_df[['sample', 'previous_event_id', 'event_id']].to_numpy()
        present_conditions = events_df['condition_name'].unique()
        final_event_id_map = components['mne_event_handler'].get_final_event_map(present_conditions)
        if not final_event_id_map:
            p_logger.error("[ERROR] Could not create a valid event ID map. Conditions found in data do not match the event mapping in the config. Skipping epoching.")
            print("[ERROR] Could not create a valid event ID map. Conditions found in data do not match the event mapping in the config. Skipping epoching.")
            return
        shortest_duration_sec = artifacts.get('shortest_epoch_duration_sec')
        if shortest_duration_sec is None:
            p_logger.warning("Shortest epoch duration not calculated. Falling back to config 'trial_end_offset'.")
            tmax = config.getfloat('Segmentation', 'trial_end_offset')
        else:
            tmax = shortest_duration_sec
        eeg_epochs_mne = components['eeg_epoch_processor'].create_epochs(
            raw_processed=eeg_raw,
            events=mne_events_array,
            event_id=final_event_id_map,
            tmin=config.getfloat('Segmentation', 'trial_start_offset'),
            tmax=tmax)
        if eeg_epochs_mne is not None:
            artifacts['eeg_epochs_mne'] = eeg_epochs_mne
            artifacts['eeg_epochs_df'] = eeg_epochs_mne.to_data_frame()
            p_logger.info("EEG epoching completed.")
            print("[INFO] EEG epoching completed.")
        else:
            p_logger.error("[ERROR] EEG epoching failed: No epochs were created.")
            print("[ERROR] EEG epoching failed: No epochs were created.")
    except Exception as e:
        p_logger.error(f"[ERROR] An unexpected error occurred during epoching: {e}", exc_info=True)
        print(f"[ERROR] An unexpected error occurred during epoching: {e}")
        return

def _run_fnirs_glm(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Runs the first-level GLM analysis on fNIRS data."""
    p_logger.info("Running fNIRS GLM analysis...")
    
    # --- Check for all required data artifacts before proceeding ---
    required_data = {
        'processed_fnirs': 'fnirs_processed_hbo_hbr',
        'markers': 'xdf_markers_df',
        'fnirs_start_time': 'fnirs_stream_start_time_xdf'
    }
    # Special check for markers, which are nested
    has_markers = 'xdf_streams' in artifacts and required_data['markers'] in artifacts['xdf_streams']
    
    if not all(key in artifacts for key in [required_data['processed_fnirs'], required_data['fnirs_start_time']]) or not has_markers:
        missing = [name for name, key in required_data.items() if key not in artifacts and name != 'markers']
        if not has_markers:
            missing.append('markers')
        p_logger.info(f"Skipping fNIRS GLM: Missing required data: {', '.join(missing)}.")
        return

    try:
        pid = artifacts['participant_id']
        fnirs_raw: Raw = artifacts[required_data['processed_fnirs']]
        markers_df = artifacts['xdf_streams'][required_data['markers']]
        fnirs_start_time = artifacts[required_data['fnirs_start_time']]
        
        p_logger.info("Using marker timestamps directly for fNIRS GLM, assuming they mark stimulus onset.")

        # Ensure fNIRS GLM respects the same condition filtering as the EEG pipeline
        conditions_to_map = [c.strip() for c in config.get('Segmentation', 'conditions_to_map', fallback='').split(',') if c.strip()]

        # --- Determine condition durations for GLM ---
        # Prioritize dynamically calculated average durations if available.
        avg_durations = artifacts.get('avg_duration_per_condition_sec')
        if avg_durations:
            p_logger.info("Using dynamically calculated average trial durations for fNIRS GLM.")
            condition_duration_config = avg_durations
        else:
            p_logger.warning("Dynamic trial durations not found. Falling back to fixed durations from config for fNIRS GLM.")
            condition_duration_config = {k: float(v) for k, v in config.items('FNIRS_ConditionDurations')}

        dm = components['fnirs_dm_processor'].create_design_matrix(
            participant_id=pid,
            xdf_markers_df=markers_df,
            raw_fnirs_data=fnirs_raw,
            fnirs_stream_start_time_xdf=fnirs_start_time, # Use the generic EventMapping section
            # More robust mapping: read all keys as strings. The processor will handle matching.
            event_mapping_config={k: v for k, v in config.items('EventMapping')},
            condition_duration_config=condition_duration_config,
            conditions_to_include=conditions_to_map)
        
        if dm is not None:
            artifacts['fnirs_design_matrix_df'] = dm
            contrasts = {name: {c.split(':')[0]: float(c.split(':')[1]) for c in pairs.split(',')} for name, pairs in config.items('FNIRS_Contrasts')}
            artifacts['fnirs_glm_results_df'] = components['glm_analyzer'].run_first_level_glm(
                data_for_glm=fnirs_raw,
                design_matrix_prepared=dm,
                participant_id=pid,
                contrasts_config=contrasts)
            p_logger.info("fNIRS GLM analysis completed.")
        else:
            p_logger.warning("fNIRS Design Matrix could not be created. Skipping GLM analysis.")
    except Exception as e:
        p_logger.error(f"An unexpected error occurred during fNIRS GLM analysis: {e}", exc_info=True)

def _run_psd_and_fai(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any], suffix: str = ""):
    """Runs Power Spectral Density (PSD) and Frontal Alpha Asymmetry (FAI) analyses. Stores results with optional suffix for side-by-side pipelines."""
    if 'eeg_epochs_mne' not in artifacts:
        p_logger.info("Skipping PSD and FAI analysis: EEG epochs not available.")
        return
    try: # This try-block covers the entire PSD and FAI analysis
        # --- PSD Analysis ---
        p_logger.info("Running PSD analysis...")
        try:
            bands_str = config.get('FAI', 'bands_config')
            bands_config = {name: tuple(map(float, freqs.strip('()').split(','))) for name, freqs in (b.split(':') for b in bands_str.split(';'))}
            # --- FAI Channel Selection: Dynamic (fNIRS-guided) or Static ---
            left_ch_fai, right_ch_fai = None, None # Initialize to None
            if config.getboolean('FAI', 'use_fnirs_guided_channels', fallback=False):
                try:
                    selector_config = {
                        'contrast': config.get('FAI', 'fnirs_contrast_for_channel_selection'),
                        'p_thresh': config.getfloat('FAI', 'fnirs_channel_significance_threshold'),
                        'fnirs_eeg_map': dict(config.items('FNIRS_EEG_MAP')),
                        'left_hemi_fnirs': {ch.strip() for ch in config.get('FNIRS_HEMISPHERES', 'left_channels').split(',')},
                        'right_hemi_fnirs': {ch.strip() for ch in config.get('FNIRS_HEMISPHERES', 'right_channels').split(',')}
                    }
                    fai_channels = components['fai_channel_selector'].select_channels(
                        glm_results_df=artifacts.get('fnirs_glm_results_df'),
                        config=selector_config
                    )
                    if fai_channels:
                        left_ch_fai, right_ch_fai = fai_channels
                except (NoSectionError, NoOptionError) as e:
                    p_logger.error(f"Configuration for fNIRS-guided FAI selection is missing: {e}. Will use static channels.")
            # Fallback to static channels if dynamic selection was disabled or failed
            if not left_ch_fai or not right_ch_fai:
                p_logger.info("Using statically defined EEG channels for FAI analysis.")
                try:
                    primary_pair_str = config.get('FAI', 'fai_primary_pair', fallback='F3,F4')
                    secondary_pair_str = config.get('FAI', 'fai_secondary_pair', fallback='Fp1,Fp2')
                    primary_pair = [ch.strip() for ch in primary_pair_str.split(',')]
                    secondary_pair = [ch.strip() for ch in secondary_pair_str.split(',')]
                    available_eeg_channels = artifacts['eeg_epochs_mne'].ch_names
                    if all(ch in available_eeg_channels for ch in primary_pair):
                        left_ch_fai, right_ch_fai = primary_pair[0], primary_pair[1]
                        p_logger.info(f"Found primary FAI channel pair: {primary_pair}")
                    elif all(ch in available_eeg_channels for ch in secondary_pair):
                        left_ch_fai, right_ch_fai = secondary_pair[0], secondary_pair[1]
                        p_logger.info(f"Primary FAI pair not available. Found secondary FAI channel pair: {secondary_pair}")
                    else:
                        p_logger.warning(f"Neither primary ({primary_pair}) nor secondary ({secondary_pair}) FAI channel pairs are fully available in the cleaned EEG data. Available channels: {available_eeg_channels}. Skipping FAI.")
                        return # Exit the function if no valid pair is found
                except (NoOptionError, NoSectionError):
                    p_logger.error("Static FAI channel pairs ('fai_primary_pair', 'fai_secondary_pair') not found in [FAI] section of config.")
                    left_ch_fai, right_ch_fai = None, None # Ensure they are None
            # Final check before proceeding
            if not left_ch_fai or not right_ch_fai:
                p_logger.error("Could not determine FAI channels. Skipping PSD and FAI analysis.")
                return
            psd_results_df = components['psd_analyzer'].compute_psd(
                epochs=artifacts['eeg_epochs_mne'],
                bands=bands_config,
                participant_id=artifacts['participant_id'],
                channels_of_interest=[left_ch_fai, right_ch_fai]
            )
            psd_key = f'psd_results_df{suffix}' if suffix else 'psd_results_df'
            artifacts[psd_key] = psd_results_df
            p_logger.info(f"PSD analysis complete. Results stored as '{psd_key}'.")
        except (NoSectionError, NoOptionError) as e:
            p_logger.error(f"Could not run PSD analysis. Missing configuration in [FAI] section: {e}")
            return # Exit if basic config is missing
        except Exception as e:
            p_logger.error(f"An unexpected error occurred during PSD analysis: {e}", exc_info=True)
            return # Exit on other errors
        # --- FAI Analysis (depends on PSD) ---
        if psd_key in artifacts and artifacts[psd_key] is not None and not artifacts[psd_key].empty:
            psd_df_check = artifacts[psd_key]
            channels_in_psd = psd_df_check['channel'].unique()
            required_fai_channels = [left_ch_fai, right_ch_fai]
            if not all(ch in channels_in_psd for ch in required_fai_channels):
                p_logger.warning(f"FAI analysis requires PSD for channels {required_fai_channels}, but results were only found for {list(channels_in_psd)}. This can happen if a channel had poor signal quality. Skipping FAI calculation.")
                return
            p_logger.info("Running FAI analysis...")
            try:
                electrode_pairs = [(left_ch_fai, right_ch_fai)]
                bands_str = config.get('FAI', 'bands_config')
                bands_config = {name: tuple(map(float, freqs.strip('()').split(','))) for name, freqs in (b.split(':') for b in bands_str.split(';'))}
                all_fai_results = []
                for band_name in bands_config.keys():
                    fai_df_band = components['fai_analyzer'].compute_fai_from_psd_df(
                        psd_df=artifacts[psd_key], fai_band_name=band_name,
                        fai_electrode_pairs_config=electrode_pairs)
                    if fai_df_band is not None and not fai_df_band.empty:
                        fai_df_band['band'] = band_name
                        all_fai_results.append(fai_df_band)
                if all_fai_results:
                    fai_key = f'fai_results_df{suffix}' if suffix else 'fai_results_df'
                    artifacts[fai_key] = pd.concat(all_fai_results, ignore_index=True)
                    p_logger.info(f"FAI analysis complete. Results stored as '{fai_key}'.")
            except (NoSectionError, NoOptionError) as e:
                p_logger.error(f"Could not run FAI analysis. Missing configuration in [FAI] section: {e}")
            except Exception as e:
                p_logger.error(f"An unexpected error occurred during FAI analysis: {e}", exc_info=True)
    except Exception as e:
        p_logger.error(f"An unexpected error occurred during PSD or FAI analysis: {e}", exc_info=True)

def _select_plv_channels_from_fnirs(config: ConfigParser, artifacts: Dict[str, Any], p_logger: logging.Logger) -> List[str]:
    """
    Selects EEG channels for PLV analysis based on significant fNIRS GLM results.
    Returns a list of EEG channel names or an empty list if selection fails.
    """
    try:
        p_logger.info("Attempting to select EEG channels for PLV based on fNIRS GLM results.")
        glm_results_df = artifacts.get('fnirs_glm_results_df')
        if glm_results_df is None or glm_results_df.empty:
            p_logger.info("fNIRS GLM results not available, cannot perform dynamic channel selection.")
            return []

        contrast_to_use = config.get('PLV', 'fnirs_contrast_for_eeg_selection')
        p_val_threshold = config.getfloat('PLV', 'fnirs_channel_significance_threshold')
        fnirs_eeg_map = dict(config.items('FNIRS_EEG_MAP'))

        significant_results = glm_results_df[
            (glm_results_df['Contrast'] == contrast_to_use) &
            (glm_results_df['p-value'] < p_val_threshold)
        ]

        if significant_results.empty:
            p_logger.warning(f"No significant fNIRS channels found for contrast '{contrast_to_use}' at p < {p_val_threshold}. Cannot guide EEG channel selection.")
            return []

        significant_fnirs_channels = significant_results['Channel'].unique()
        p_logger.info(f"Found {len(significant_fnirs_channels)} significant fNIRS channels for contrast '{contrast_to_use}': {significant_fnirs_channels}")

        selected_eeg_channels = set()
        for fnirs_ch in significant_fnirs_channels:
            sd_pair = fnirs_ch.split(' ')[0].replace('_', '-')
            if sd_pair in fnirs_eeg_map:
                eeg_chs_for_sd = [ch.strip() for ch in fnirs_eeg_map[sd_pair].split(',')]
                selected_eeg_channels.update(eeg_chs_for_sd)

        if selected_eeg_channels:
            return sorted(list(selected_eeg_channels))
        else:
            p_logger.warning("Found significant fNIRS channels, but could not map them to any EEG channels using [FNIRS_EEG_MAP].")
            return []
    except (NoSectionError, NoOptionError) as e:
        p_logger.error(f"Configuration for fNIRS-guided EEG selection is missing: {e}. PLV will use statically defined channels.")
        return []

def _run_plv_analysis(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Runs Phase Locking Value (PLV) analysis, with optional fNIRS-guided channel selection."""
    if 'eeg_epochs_mne' not in artifacts or artifacts['eeg_epochs_mne'] is None:
        p_logger.info("Skipping PLV analysis: EEG epochs not available.")
        return
    all_plv_results = []
    pid = artifacts['participant_id']

    # --- Determine which EEG channels to use ---
    eeg_channels_for_plv = _select_plv_channels_from_fnirs(config, artifacts, p_logger) if config.getboolean('PLV', 'use_fnirs_guided_channels', fallback=False) else []
    if not eeg_channels_for_plv:
        p_logger.info("Using statically defined EEG channels for PLV analysis (or fNIRS-guided selection failed).")
        eeg_channels_for_plv = [ch.strip() for ch in config.get('PLV', 'eeg_channels').split(',')]

    if not eeg_channels_for_plv:
        p_logger.warning("No EEG channels selected for PLV analysis. Skipping.")
        return

    # --- Loop through available autonomic signals and compute PLV for each ---
    autonomic_signals_to_process = {
        'HRV': ('hrv_continuous_df', 'ecg_sfreq'),
        'EDA': ('eda_processed_df', 'eda_sfreq')
    }
    bands_config = {name: tuple(map(float, freqs.strip('()').split(','))) for name, freqs in (b.split(':') for b in config.get('PLV', 'eeg_bands_config').split(';'))}

    for sig_name, (df_key, sfreq_key) in autonomic_signals_to_process.items():
        autonomic_df = artifacts.get(df_key)
        autonomic_sfreq = artifacts.get(sfreq_key, 0)

        if autonomic_df is not None and not autonomic_df.empty and autonomic_sfreq > 0:
            p_logger.info(f"Running PLV analysis for EEG-{sig_name}...")
            plv_df = components['connectivity_analyzer'].compute_plv(
                signal1_epochs=artifacts['eeg_epochs_mne'],
                signal1_channels_to_average=eeg_channels_for_plv,
                signal1_bands_config=bands_config,
                autonomic_signal_df=autonomic_df,
                autonomic_signal_sfreq=autonomic_sfreq,
                signal1_name="EEG",
                autonomic_signal_name=sig_name,
                participant_id=pid,
                reject_s2=None # Explicitly disable rejection for the secondary signal
            )
            if plv_df is not None and not plv_df.empty:
                all_plv_results.append(plv_df)
        else:
            p_logger.info(f"Skipping PLV analysis for EEG-{sig_name}: Missing required data ('{df_key}' or '{sfreq_key}').")

    if all_plv_results:
        final_plv_df = pd.concat(all_plv_results, ignore_index=True)
        artifacts['plv_results_df'] = final_plv_df
        p_logger.info("PLV analysis complete.")

def _run_hrv_processing(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Runs HRV processing to generate continuous signal for PLV. Does not save discrete results."""
    pid = artifacts['participant_id']
    output_dir = artifacts['output_dir']

    # --- HRV Processing (for continuous signal) ---
    if 'rpeaks_df_out' in artifacts and artifacts.get('ecg_sfreq', 0) > 0:
        p_logger.info("Running HRV analysis...")
        rpeaks_df = artifacts['rpeaks_df_out']
        sfreq = artifacts['ecg_sfreq']
        rpeaks_samples = rpeaks_df['R_Peak_Sample'].to_numpy()
        
        hrv_artifacts = components['ecg_hrv_processor'].process_rpeaks_to_hrv(
            rpeaks_samples=rpeaks_samples, original_sfreq=sfreq,
            participant_id=pid, output_dir=os.path.join(output_dir, "HRV_Analysis"),
            total_duration_sec=artifacts.get('ecg_duration_sec')
        )
        
        if isinstance(hrv_artifacts, dict):
            artifacts.update(hrv_artifacts)
            p_logger.info("HRV processing for continuous signal completed.")
        elif hrv_artifacts is not None:
            p_logger.warning(f"HRV processor returned an unexpected type '{type(hrv_artifacts)}' instead of a dictionary. Skipping HRV artifact update.")
    else:
        p_logger.info("Skipping HRV analysis: Missing R-peaks or ECG sampling frequency.")

def _run_analyses(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Runs all physiological analyses based on preprocessed data by dispatching to helper functions."""
    p_logger.info("--- Step 4: Physiological Analyses ---")
    
    _create_mne_events(config, components, p_logger, artifacts)
    
    # --- NEW: Calculate variable durations after events are created ---
    if 'mne_events_df' in artifacts and 'eeg_processed_raw' in artifacts:
        _calculate_and_store_durations(
            df=artifacts['mne_events_df'],
            sfreq=artifacts['eeg_processed_raw'].info['sfreq'],
            fallback_duration=config.getfloat('Segmentation', 'trial_end_offset'),
            p_logger=p_logger,
            artifacts=artifacts
        )

    _run_epoching(config, components, p_logger, artifacts)
    if config.getboolean('ProcessingSwitches', 'process_fnirs', fallback=True):
        _run_fnirs_glm(config, components, p_logger, artifacts)
    _run_hrv_processing(config, components, p_logger, artifacts)
    _run_psd_and_fai(config, components, p_logger, artifacts)
    _run_plv_analysis(config, components, p_logger, artifacts)

def _save_and_report(config: ConfigParser, components: Dict[str, Any], p_logger: logging.Logger, artifacts: Dict[str, Any]):
    """Saves all generated DataFrames and creates plots."""
    p_logger.info("--- Step 5: Saving Results & Generating Reports ---")
    pid = artifacts['participant_id']
    output_dir = artifacts['output_dir']
    xml_reporter = components['xml_reporter']
    plot_reporter = components['plot_reporter'] # Added to generate plots

    # --- A. Merge Questionnaire scores into physiological results for easier analysis ---
    q_results_df = artifacts.get('questionnaire_results_df')
    fai_results_df = artifacts.get('fai_results_df')
    plv_results_df = artifacts.get('plv_results_df')

    if q_results_df is not None and not q_results_df.empty:
        try:
            # Extract the first row of scores as a dictionary
            q_scores = q_results_df.iloc[0].to_dict()
            # Remove the participant ID from the scores to merge, as it's already in the target DFs
            id_col_name = config.get('QuestionnairePreprocessing', 'output_participant_id_col_name')
            if id_col_name in q_scores:
                del q_scores[id_col_name]
            
            p_logger.info(f"Merging questionnaire scores into physiological results: {list(q_scores.keys())}")

            if fai_results_df is not None and not fai_results_df.empty:
                artifacts['fai_results_df'] = fai_results_df.assign(**q_scores)
                p_logger.info("Successfully merged questionnaire scores into FAI results.")

            if plv_results_df is not None and not plv_results_df.empty:
                artifacts['plv_results_df'] = plv_results_df.assign(**q_scores)
                p_logger.info("Successfully merged questionnaire scores into PLV results.")
        except (NoSectionError, NoOptionError) as e:
            p_logger.error(f"Could not merge questionnaire scores due to missing config: {e}")

    # --- B. Save all result DataFrames to XML ---    
    # Define which artifacts should be saved if they exist as valid DataFrames.
    # Note: 'eeg_epochs_df' is intentionally excluded as it contains complex objects
    # that are not suitable for direct XML serialization.
    artifact_keys_to_save = [
        'questionnaire_results_df',
        'mne_events_df',
        'fnirs_design_matrix_df',
        'fnirs_glm_results_df',
        'psd_results_df',
        'fai_results_df',
        'plv_results_df'
    ]

    for key in artifact_keys_to_save:
        df = artifacts.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            filename_base = key.replace('_df', '')
            filename = f"{pid}_{filename_base}.xml"
            xml_reporter.save_dataframe(data_df=df, output_dir=output_dir, filename=filename)
        else:
            p_logger.debug(f"Skipping save for '{key}': Artifact not found, is not a DataFrame, or is empty.")
    
    # --- C. Generate Plots and Corresponding Data Files ---
    # For each plot, we also save the exact DataFrame used to generate it as an XML file.

    # --- 1. fNIRS GLM Results Plot ---
    if 'fnirs_glm_results_df' in artifacts and artifacts['fnirs_glm_results_df'] is not None and not artifacts['fnirs_glm_results_df'].empty:
        plot_data = artifacts['fnirs_glm_results_df']
        plot_config_glm = {
            'plot_name': 'fnirs_glm_contrasts_{participant_id_or_group}',
            'plot_type': 'faceted_barplot',
            'plot_category_subdir': 'FNIRS_Plots',
            'data_mapping': {'x_col': 'Contrast', 'y_col': 'theta', 'facet_col': 'Channel'},
            'plot_params': {
                'title': 'fNIRS GLM Contrasts for {participant_id_or_group}',
                'x_axis_label': 'Contrast', 'y_axis_label': 'Beta Coefficient (theta)',
                'facet_col_wrap': 4, 'facet_col_title_template': '{col_name}',
                'xticks_rotation': 45, 'xticks_ha': 'right',
                'stat_results': None # Placeholder for future stat results to draw significance stars
            }
        }
        plot_reporter.generate_plot(participant_id_or_group=pid, plot_config=plot_config_glm, data_payload=plot_data)
        xml_reporter.save_dataframe(data_df=plot_data, output_dir=os.path.join(output_dir, "FNIRS_Plots"), filename=f"{pid}_fnirs_glm_contrasts_data.xml")

    # --- 2. PSD Results Plot ---
    if 'psd_results_df' in artifacts and artifacts['psd_results_df'] is not None and not artifacts['psd_results_df'].empty:
        plot_data = artifacts['psd_results_df']
        plot_config_psd = {
            'plot_name': 'psd_by_band_{participant_id_or_group}',
            'plot_type': 'faceted_barplot',
            'plot_category_subdir': 'PSD_Plots',
            'data_mapping': {'x_col': 'band', 'y_col': 'power', 'hue_col': 'condition', 'facet_col': 'channel'},
            'plot_params': {
                'title': 'Power Spectral Density for {participant_id_or_group}',
                'x_axis_label': 'Frequency Band', 'y_axis_label': 'Power (uV^2/Hz)',
                'facet_col_title_template': '{col_name}',
                'xticks_rotation': 0,
                'stat_results': None # Placeholder for future stat results
            }
        }
        plot_reporter.generate_plot(participant_id_or_group=pid, plot_config=plot_config_psd, data_payload=plot_data)
        xml_reporter.save_dataframe(data_df=plot_data, output_dir=os.path.join(output_dir, "PSD_Plots"), filename=f"{pid}_psd_by_band_data.xml")

    # --- 3. FAI Results Plot ---
    if 'fai_results_df' in artifacts and artifacts['fai_results_df'] is not None and not artifacts['fai_results_df'].empty:
        plot_data = artifacts['fai_results_df']
        plot_config_fai = {
            'plot_name': 'fai_by_condition_{participant_id_or_group}',
            'plot_type': 'faceted_barplot',
            'plot_category_subdir': 'FAI_Plots',
            'data_mapping': {'x_col': 'condition', 'y_col': 'fai_value', 'facet_col': 'band'},
            'plot_params': {
                'x_axis_label': 'Experimental Condition', 'y_axis_label': 'FAI Value',
                'facet_col_title_template': '{col_name}', 'xticks_rotation': 45, 'xticks_ha': 'right',
                'stat_results': None # Placeholder for future stat results
            }
        }
        plot_reporter.generate_plot(participant_id_or_group=pid, plot_config=plot_config_fai, data_payload=plot_data)
        xml_reporter.save_dataframe(data_df=plot_data, output_dir=os.path.join(output_dir, "FAI_Plots"), filename=f"{pid}_fai_by_condition_data.xml")

    # --- 4. PLV Results Plot ---
    if 'plv_results_df' in artifacts and artifacts['plv_results_df'] is not None and not artifacts['plv_results_df'].empty:
        plot_data = artifacts['plv_results_df']
        plot_config_plv = {
            'plot_name': 'plv_by_condition_{participant_id_or_group}',
            'plot_type': 'faceted_barplot',
            'plot_category_subdir': 'PLV_Plots',
            'data_mapping': {'x_col': 'condition', 'y_col': 'plv_value', 'hue_col': 'modality_pair', 'facet_col': 'band'},
            'plot_params': {
                'x_axis_label': 'Experimental Condition', 'y_axis_label': 'PLV Value',
                'facet_col_title_template': '{col_name}', 'xticks_rotation': 45, 'xticks_ha': 'right',
                'stat_results': None # Placeholder for future stat results
            }
        }
        plot_reporter.generate_plot(participant_id_or_group=pid, plot_config=plot_config_plv, data_payload=plot_data)
        xml_reporter.save_dataframe(data_df=plot_data, output_dir=os.path.join(output_dir, "PLV_Plots"), filename=f"{pid}_plv_by_condition_data.xml")


# =============================================================================
# === 3. Main Participant Orchestration Function
# =============================================================================

def run_participant_pipeline(participant_config: Dict[str, Any], 
                             global_config: ConfigParser, 
                             components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the entire analysis pipeline for a single participant using DAG-based parallelism and modular services.
    """
    import time
    start_time = time.time()
    participant_id = participant_config['participant_id']
    base_output_dir = global_config.get('Data', 'base_output_dir')
    participant_output_dir = os.path.join(base_output_dir, participant_id)
    os.makedirs(participant_output_dir, exist_ok=True)

    # Setup participant-specific logger using logging_utils
    log_reporter = LogReporter(participant_base_output_dir=participant_output_dir,
                               participant_id=participant_id,
                               log_level_str=global_config.get('DEFAULT', 'log_level'))
    p_logger = log_reporter.get_logger()
    p_logger.propagate = False
    p_logger.info(f"--- Starting analysis for participant: {participant_id} ---")

    # Use ParticipantArtifacts for artifact management
    participant_artifacts = ParticipantArtifacts(participant_id, participant_output_dir)
    participant_artifacts.set('participant_id', participant_id)
    participant_artifacts.set('output_dir', participant_output_dir)

    # Step 1: Load Data (sequential, as all downstream tasks depend on it)
    _load_data(participant_id, global_config, components, p_logger, participant_artifacts._artifacts)
    p_logger.info(f"Data loading complete. (Elapsed: {time.time() - start_time:.2f}s)")        
    p_logger.debug(f"Artifacts after loading: {participant_artifacts.keys()}")

    # -------------------------------------------------------------------------
    # IMPORTANT: Mapping preprocessing outputs to DAG-expected keys
    # -------------------------------------------------------------------------
    # If you add a new modality, change a preprocessing output key, or change
    # what the DAG expects, update the mapping list below accordingly.
    # This ensures all downstream steps receive the correct data.
    # -------------------------------------------------------------------------
    mapping = [
        ('eeg_preprocessed', 'eeg_processed_raw'),
        ('ecg_preprocessed', 'rpeaks_df_out'),
        ('eda_preprocessed', 'eda_processed_df'),
        ('fnirs_preprocessed', 'fnirs_processed_hbo_hbr'),
        ('questionnaire_processed', 'questionnaire_results_df'),
    ]
    for expected, actual in mapping:
        if participant_artifacts.has(actual):
            participant_artifacts.set(expected, participant_artifacts.get(actual))

    # --- NEW: Explicitly map raw input keys for the DAG ---
    # These are the keys the DAG expects for preprocessing
    dag_input_mapping = [
        ('eeg', 'eeg'),
        ('ecg_df', 'ecg_df'),
        ('eda_df', 'eda_df'),
        ('fnirs_od_df', 'fnirs_od_df'),
    ]
    for dag_key, source_key in dag_input_mapping:
        # Try both artifacts and streams
        if participant_artifacts.has(source_key):
            participant_artifacts.set(dag_key, participant_artifacts.get(source_key))
        elif 'xdf_streams' in participant_artifacts._artifacts and source_key in participant_artifacts._artifacts['xdf_streams']:
            participant_artifacts.set(dag_key, participant_artifacts._artifacts['xdf_streams'][source_key])

    # --- DIAGNOSTIC: Dump all artifacts before building the DAG ---
    p_logger.info("[DIAGNOSTIC] Artifacts before DAG construction:")
    for k in participant_artifacts._artifacts:
        v = participant_artifacts._artifacts[k]
        if isinstance(v, pd.DataFrame):
            p_logger.info(f"  {k}: DataFrame shape {v.shape}, columns: {list(v.columns)}")
        else:
            p_logger.info(f"  {k}: {type(v).__name__}")

    # Build the DAG using the modular pipeline_dag
    # Map actual preprocessed keys to expected keys for the DAG
    mapping = [
        ('eeg_preprocessed', 'eeg_processed_raw'),
        ('ecg_preprocessed', 'rpeaks_df_out'),
        ('eda_preprocessed', 'eda_processed_df'),
        ('fnirs_preprocessed', 'fnirs_processed_hbo_hbr'),
        ('questionnaire_processed', 'questionnaire_results_df'),
    ]
    for expected, actual in mapping:
        if participant_artifacts.has(actual):
            participant_artifacts.set(expected, participant_artifacts.get(actual))
    # Debug log: show all artifact keys and their types/lengths
    artifact_keys = list(participant_artifacts._artifacts.keys())
    artifact_types = {k: type(participant_artifacts._artifacts[k]).__name__ for k in artifact_keys}
    artifact_lens = {k: (len(participant_artifacts._artifacts[k]) if hasattr(participant_artifacts._artifacts[k], '__len__') else 'n/a') for k in artifact_keys}
    p_logger.debug(f"Available artifact keys before analysis: {artifact_keys}")
    p_logger.debug(f"Artifact types: {artifact_types}")
    p_logger.debug(f"Artifact lengths: {artifact_lens}")
    dag_tasks = build_participant_dag(participant_artifacts, global_config, components, p_logger)
    dag_runner = DAGParallelTaskRunner(dag_tasks, max_workers=global_config.getint('Parallel', 'max_workers', fallback=4), logger=p_logger)
    dag_results = dag_runner.run()

    total_time = time.time() - start_time
    p_logger.info(f"--- Participant {participant_id} analysis completed successfully in {total_time:.2f} seconds. ---")
    return {'participant_id': participant_id, 'status': 'completed', 'duration_sec': total_time, 'dag_results': dag_results}

def get_participants_to_run(config: ConfigParser, base_raw_dir: str, logger: logging.Logger) -> List[str]:
    """Determines the list of participant IDs to process based on the config file."""
    # Priority 1: Use the explicit list of participant IDs if provided
    p_ids_str = config.get('Data', 'participant_ids', fallback='').strip()
    if p_ids_str:
        participant_ids = [pid.strip() for pid in p_ids_str.split(',') if pid.strip()]
        logger.info(f"Processing specific participants defined in config: {participant_ids}")
        return participant_ids

    # Priority 2: Scan the raw data directory for all potential participants
    logger.info(f"No specific participant_ids found in config. Scanning '{base_raw_dir}' for all potential participants.")
    try:
        all_potential_dirs = sorted([d for d in os.listdir(base_raw_dir) if os.path.isdir(os.path.join(base_raw_dir, d)) and d.startswith("EV_")])
        logger.info(f"Found {len(all_potential_dirs)} potential participant directories.")
        return all_potential_dirs
    except FileNotFoundError:
        logger.error(f"Raw data directory not found: {base_raw_dir}. Cannot discover participants.")
        return []

def continuous_participant_watcher(global_config, components, logger):
    """
    Continuously checks for new participants, processes them, and performs git commit/push after each.
    """
    base_raw_dir = global_config.get('Data', 'base_raw_data_dir')
    base_output_dir = global_config.get('Data', 'base_output_dir')
    polling_interval = global_config.getint('ContinuousMode', 'polling_interval_seconds', fallback=300)
    processed_participant_ids = set()
    repo_path = global_config.get('Git', 'repository_path')
    git_handler = GitHandler(repository_path=repo_path, logger=logger)

    while True:
        participants_to_run = get_participants_to_run(global_config, base_raw_dir, logger)
        new_participants_to_run = [pid for pid in participants_to_run if pid not in processed_participant_ids]
        if not new_participants_to_run:
            logger.info("No new participants found to process in this cycle. Sleeping...")
            time.sleep(polling_interval)
            continue
        logger.info(f"Found {len(new_participants_to_run)} new participant(s) to process: {new_participants_to_run}")
        for pid in new_participants_to_run:
            result = run_participant_pipeline(
                participant_config={'participant_id': pid},
                global_config=global_config,
                components=components
            )
            if result.get('status') == 'completed':
                processed_participant_ids.add(pid)
                # Perform git commit and push for this participant's results
                participant_output_path = os.path.join(base_output_dir, pid)
                commit_message = f"Auto-commit: Processed data for participant {pid}"
                git_handler.commit_and_sync_changes(
                    paths_to_add=[participant_output_path],
                    commit_message=commit_message
                )
                logger.info(f"Participant {pid} processed and committed to git.")
            else:
                logger.warning(f"Participant {pid} processing failed. It will be retried in the next polling cycle. Error: {result.get('error')}")
        logger.info(f"Polling cycle complete. Sleeping for {polling_interval} seconds...")
        time.sleep(polling_interval)

# =============================================================================
# === 4. Main Execution Block
# =============================================================================

if __name__ == "__main__":
    CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "EV_config.cfg")
    try:
        config = load_configuration(CONFIG_FILE_PATH)
        logger = setup_main_logger(config)
    except Exception as e:
        logging.basicConfig()
        logging.getLogger().critical(f"Failed to load configuration or set up logging: {e}")
        sys.exit(1)

    BASE_OUTPUT_DIR = config.get('Data', 'base_output_dir')
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    analysis_components = instantiate_components(config, logger, BASE_OUTPUT_DIR)
    
    def dispatch_participant_pipeline(task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatcher for inter-participant parallelism. Unpacks config and calls the main pipeline."""
        return run_participant_pipeline(
            participant_config=task_config['participant_config'],
            global_config=task_config['global_config'],
            components=task_config['components']
        )

    # --- Determine Execution Mode: Single Run vs. Continuous Polling ---
    run_continuously = config.getboolean('ContinuousMode', 'run_continuously', fallback=False)

    if not run_continuously:
        # --- Single Run Mode ---
        logger.info("--- Starting Orchestrator in Parallel Batch-Processing Mode ---")
        base_raw_dir = config.get('Data', 'base_raw_data_dir')
        participants_to_run = get_participants_to_run(config, base_raw_dir, logger)

        if not participants_to_run:
            logger.warning("No participants to process. Exiting.")
        else:
            # --- Create task configs for inter-participant parallelism ---
            participant_task_configs = []
            for pid in participants_to_run:
                participant_task_configs.append({
                    'participant_config': {'participant_id': pid},
                    'global_config': config,
                    'components': analysis_components
                })
            
            logger.info(f"Starting parallel processing for {len(participant_task_configs)} participants...")
            
            inter_participant_runner = ParallelTaskRunner(
                task_function=dispatch_participant_pipeline,
                task_configs=participant_task_configs,
                main_logger_name=logger.name,
                max_workers=config.getint('Parallel', 'max_workers', fallback=4),
                thread_name_prefix="ParticipantWorker"
            )
            results = inter_participant_runner.run()

            # --- Process results and handle Git commits sequentially after all runs are complete ---
            logger.info("--- Parallel processing complete. Finalizing results. ---")
            for result in results:
                if result and result.get('status') == 'completed':
                    pid = result['participant_id']
                    logger.info(f"Participant {pid} completed successfully in {result['duration_sec']:.2f}s.")
                    if config.getboolean('Git', 'use_git_tracking'):
                        participant_output_path = os.path.join(config.get('Data', 'base_output_dir'), pid)
                        analysis_components['git_handler'].commit_and_sync_changes(
                            paths_to_add=[participant_output_path],
                            commit_message=f"Auto-commit: Processed data for participant {pid}"
                        )
                elif result:
                    pid = result.get('participant_id', 'Unknown')
                    error = result.get('error', 'No error message.')
                    logger.error(f"Participant {pid} failed. Error: {error}")
    else:
        # --- Continuous Polling Mode ---
        polling_interval = config.getint('ContinuousMode', 'polling_interval_seconds', fallback=300)
        base_raw_dir = config.get('Data', 'base_raw_data_dir')
        processed_participant_ids = set()
        logger.info(f"--- Starting Orchestrator in Continuous Mode ---")
        logger.info(f"Polling for new participant data in '{base_raw_dir}' every {polling_interval} seconds. Press Ctrl+C to stop.")

        try:
            while True:
                logger.info("--- New Polling Cycle Started ---")
                try:
                    all_potential_dirs = {d for d in os.listdir(base_raw_dir) if os.path.isdir(os.path.join(base_raw_dir, d)) and d.startswith("EV_")}
                except FileNotFoundError:
                    logger.error(f"Raw data directory not found: {base_raw_dir}. Retrying after interval.")
                    time.sleep(polling_interval)
                    continue

                new_participants_to_run = sorted(list(all_potential_dirs - processed_participant_ids))

                if not new_participants_to_run:
                    logger.info("No new participants found to process in this cycle.")
                else:
                    logger.info(f"Found {len(new_participants_to_run)} new participant(s) to process: {new_participants_to_run}")
                    # In continuous mode, we process new arrivals sequentially to avoid complex state management.
                    # For batch processing of new arrivals, a more sophisticated queueing system would be needed.
                    for pid in new_participants_to_run:
                        result = run_participant_pipeline(
                            participant_config={'participant_id': pid},
                            global_config=config,
                            components=analysis_components
                        )
                        if result.get('status') == 'completed':
                            processed_participant_ids.add(pid)  # Add to processed set only on success
                            if config.getboolean('Git', 'use_git_tracking'):
                                participant_output_path = os.path.join(config.get('Data', 'base_output_dir'), pid)
                                analysis_components['git_handler'].commit_and_sync_changes(paths_to_add=[participant_output_path], commit_message=f"Auto-commit: Processed data for participant {pid}")
                        else:
                            logger.warning(f"Participant {pid} processing failed. It will be retried in the next polling cycle. Error: {result.get('error')}")

                logger.info(f"Polling cycle complete. Sleeping for {polling_interval} seconds...")
                time.sleep(polling_interval)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down orchestrator gracefully.")

    logger.info("="*78)
    logger.info("=== EmotiView Analysis Pipeline Terminated ===")
    if config.getboolean('Git', 'use_git_tracking'):
        logger.info("Performing final Git sync on shutdown...")
        analysis_components['git_handler'].commit_and_sync_changes(
            paths_to_add=[BASE_OUTPUT_DIR],
            commit_message='Final auto-commit on orchestrator shutdown',
            remote_name='origin'
            )

    # Start the continuous participant watcher in a background thread
    watcher_thread = threading.Thread(target=continuous_participant_watcher, args=(config, analysis_components, logger), daemon=True)
    watcher_thread.start()
    watcher_thread.join()

# Helper to validate and resolve artifacts before module calls
def validate_and_resolve_artifact(artifacts, artifact_key, expected_type: Union[Type, Tuple[Type, ...]], required_columns=None, logger=None, aliases=None):
    obj = artifacts.get(artifact_key)
    if obj is None:
        if logger:
            logger.warning(f"Artifact '{artifact_key}' not found.")
        return None
    # Type check
    if not isinstance(obj, expected_type):
        if logger:
            if isinstance(expected_type, tuple):
                type_names = ', '.join(t.__name__ for t in expected_type)
            else:
                type_names = expected_type.__name__
            logger.warning(f"Artifact '{artifact_key}' is not of expected type {type_names} (got {type(obj).__name__}).")
        return None
    # DataFrame column check
    if required_columns and isinstance(obj, pd.DataFrame):
        missing = [col for col in required_columns if col not in obj.columns]
        if missing:
            # Try to resolve by case-insensitive or alias mapping
            resolved_cols = list(obj.columns)
            col_map = {col.lower(): col for col in obj.columns}
            for col in missing:
                if col.lower() in col_map:
                    resolved_cols.append(col_map[col.lower()])
                elif aliases:
                    for alias in aliases.get(col, []):
                        if alias.lower() in col_map:
                            resolved_cols.append(col_map[alias.lower()])
                            break
            still_missing = [col for col in required_columns if col not in resolved_cols]
            if still_missing:
                if logger:
                    logger.error(f"Artifact '{artifact_key}' is missing required columns: {still_missing}")
                return None
            # Optionally, could rename columns here if needed
    return obj