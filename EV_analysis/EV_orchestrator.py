import os
import sys
import logging
from typing import Dict, Any, List, Optional, Union # Added List, Optional, Union
import time # For the polling loop
import pandas as pd
from mne import Epochs
import pickle # For saving/loading participant artifacts
import numpy as np  # Added for type hints like np.ndarray
from mne.io import Raw
from configparser import ConfigParser

# Add the PsyAnalysisToolbox path to the system path so modules can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'PsyAnalysisToolbox', 'Python')))

from readers.xdf_reader import XDFReader # type: ignore
from readers.txt_reader import TXTReader # type: ignore
from preprocessors.eeg_preprocessor import EEGPreprocessor # type: ignore
from preprocessors.eda_preprocessor import EDAPreprocessor # type: ignore
from preprocessors.ecg_preprocessor import ECGPreprocessor # type: ignore
from preprocessors.questionnaire_preprocessor import QuestionnairePreprocessor # type: ignore
from preprocessors.fnirs_preprocessor import FNIRSPreprocessor # type: ignore
from preprocessors.fnirs_design_matrix_preprocessor import FNIRSDesignMatrixPreprocessor # type: ignore
from processors.questionnaire_scale_processor import QuestionnaireScaleProcessor # type: ignore
from processors.ecg_hrv_processor import ECGHRVProcessor # type: ignore
from processors.eda_scr_processor import EDASCRProcessor # type: ignore
from processors.eeg_epoch_processor import EEGEpochProcessor # type: ignore
from analyzers.psd_analyzer import PSDAnalyzer # type: ignore
from analyzers.fai_analyzer import FAIAnalyzer # type: ignore
from analyzers.connectivity_analyzer import ConnectivityAnalyzer # type: ignore
from analyzers.hrv_analyzer import HRVAnalyzer # type: ignore
from analyzers.score_analyzer import ScoreAnalyzer # type: ignore
from analyzers.glm_analyzer import GLMAnalyzer # type: ignore
from reporters.plot_reporter import PlotReporter # type: ignore
from reporters.log_reporter import LogReporter # type: ignore
from reporters.data_quality_reporter import DataQualityReporter # type: ignore
from utils.git_handler import GitHandler # type: ignore
from utils.parallel_runner import ParallelTaskRunner # type: ignore
from utils.mne_event_handler import MNEEventHandler # type: ignore
from utils.fdr_corrector import FDRCorrector # type: ignore # Corrected based on user feedback
from analyzers.anova_analyzer import ANOVAAnalyzer # type: ignore
from processors.groupLevel_processor import GroupLevelProcessing # type: ignore
from analyzers.groupLevel_analyzer import GroupLevelAnalyzing # type: ignore


# === Configuration Loading ===
config_file = os.path.join(os.path.dirname(__file__), "EV_config.cfg")
config = ConfigParser()
config.read(config_file)

# === Logging Setup ===
main_logger_name = config['DEFAULT']['main_logger_name']
# Configure basicConfig for the root logger to catch messages if specific loggers aren't fully set up
logging.basicConfig(level=config['DEFAULT'].get('log_level', 'INFO').upper(),
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure console output
logger = logging.getLogger(main_logger_name)
logger.info("Orchestrator script started.")

# === Git Handling (Optional) ===
git_handler: Optional[GitHandler] = None
if config['Git'].getboolean('use_git_tracking'):
    try:
        git_path = config['Git']['repository_path']
        git_handler = GitHandler(git_path, logger)
        # Initial commit for starting the run
        git_handler.commit_and_sync_changes(paths_to_add=["."], commit_message="Automated: Start of analysis run") # type: ignore
    except Exception as e_git_init:
        logger.error(f"Failed to initialize GitHandler or perform initial commit: {e_git_init}. Git tracking disabled for this run.")
        git_handler = None

# === Utility Functions ===
def parse_bands_config(config_str: str, default_config_str: str = 'Alpha:(8,13)') -> Dict[str, tuple[float, float]]:
    """Parses a band configuration string like 'Alpha:(8,13);Beta:(13,30)' into a dictionary."""
    bands = {}
    if not config_str:
        config_str = default_config_str
    try:
        for band_entry in config_str.split(';'):
            if ':' not in band_entry or '(' not in band_entry or ')' not in band_entry or ',' not in band_entry:
                logger.warning(f"Skipping malformed band entry: {band_entry} in config string: {config_str}")
                continue
            name, freqs_str = band_entry.split(':', 1)
            freqs = tuple(map(float, freqs_str.strip().strip('()').split(',')))
            bands[name.strip()] = freqs # type: ignore
    except Exception as e:
        logger.error(f"Error parsing band configuration string '{config_str}': {e}. Using default: {default_config_str}")
        return parse_bands_config(default_config_str, '') # Recurse with default, ensuring it's parsed once
    return bands

# === Data Locations ===
base_raw_data_dir = config['Data']['base_raw_data_dir']
base_output_dir = config['Data']['base_output_dir']
os.makedirs(base_output_dir, exist_ok=True) # Ensure base output dir exists

# === Core Component Instantiation ===
xdf_reader = XDFReader(logger=logger,
                       eeg_stream_name=config['XDF']['eeg_stream_name'],
                       fnirs_stream_name=config['XDF']['fnirs_stream_name'],
                       ecg_stream_name=config['XDF']['ecg_stream_name'],
                       eda_stream_name=config['XDF']['eda_stream_name'],
                       marker_stream_name=config['XDF']['marker_stream_name'])

questionnaire_reader = TXTReader(logger=logger)
questionnaire_preprocessor = QuestionnairePreprocessor(logger=logger)
questionnaire_scale_processor = QuestionnaireScaleProcessor(logger=logger)
score_analyzer = ScoreAnalyzer(logger=logger)
eeg_epoch_processor = EEGEpochProcessor(logger=logger) # Instantiate EEGEpochProcessor

eeg_preprocessor = EEGPreprocessor(logger=logger)
eda_preprocessor = EDAPreprocessor(logger=logger)
ecg_preprocessor = ECGPreprocessor(logger=logger)
anova_analyzer = ANOVAAnalyzer(logger=logger) # Instantiate ANOVAAnalyzer
fnirs_preprocessor = FNIRSPreprocessor(logger=logger)

mne_event_handler = MNEEventHandler(config={'conditions_to_map': config['Segmentation'].get('conditions_to_map').split(',')}, logger=logger)

ecg_hrv_processor = ECGHRVProcessor(logger=logger)
eda_scr_processor = EDASCRProcessor(logger=logger)

psd_analyzer = PSDAnalyzer(logger=logger)
fai_analyzer = FAIAnalyzer(logger=logger)
connectivity_analyzer = ConnectivityAnalyzer(logger=logger) # Use ConnectivityAnalyzer
hrv_analyzer = HRVAnalyzer(logger=logger) # For overall HRV from NNI file if needed

fnirs_design_matrix_preprocessor = FNIRSDesignMatrixPreprocessor(logger=logger,
    hrf_model_config=config['FNIRS_DM'].get('hrf_model', FNIRSDesignMatrixPreprocessor.DEFAULT_HRF_MODEL),
    drift_model_config=config['FNIRS_DM'].get('drift_model', FNIRSDesignMatrixPreprocessor.DEFAULT_DRIFT_MODEL),
    drift_order_config=config['FNIRS_DM'].getint('drift_order', FNIRSDesignMatrixPreprocessor.DEFAULT_DRIFT_ORDER)
)

glm_analyzer = GLMAnalyzer(logger=logger)

plot_reporter = PlotReporter(logger=logger, output_dir_base=base_output_dir,
                             reporting_figure_format_config=config['Plotting']['reporting_figure_format'],
                             reporting_dpi_config=config['Plotting'].getint('reporting_dpi'))
data_quality_reporter = DataQualityReporter(logger=logger) # Instantiate DataQualityReporter
fdr_corrector = FDRCorrector(logger=logger) # Instantiate FDRCorrector once

# Populate available_analyzers for GroupLevelAnalyzing
available_analyzers = {
    "psd": psd_analyzer, "fai": fai_analyzer, "connectivity": connectivity_analyzer,
    "hrv": hrv_analyzer, "scores": score_analyzer, "glm": glm_analyzer,
    "anova": anova_analyzer
}

# Group Level Components
group_level_processor = GroupLevelProcessing(logger=logger, general_configs=dict(config.items('DEFAULT')))
group_level_analyzer = GroupLevelAnalyzing(logger=logger, available_analyzers=available_analyzers)


# === Helper Functions for Participant Analysis ===

def _load_participant_data(participant_id: str, participant_logger: logging.Logger, config: ConfigParser) -> Dict[str, Any]:
    """Loads raw data (XDF, Questionnaire) for a participant."""
    participant_data: Dict[str, Any] = {}
    base_raw_data_dir = config['Data']['base_raw_data_dir']

    # Load XDF
    xdf_file_path = os.path.join(base_raw_data_dir, participant_id, f"{participant_id}.xdf")
    if not os.path.exists(xdf_file_path):
        participant_logger.error(f"XDF file not found: {xdf_file_path}. Cannot proceed with physiological data.")
    else:
        try:
            participant_data['xdf_streams'] = xdf_reader.load_participant_streams(participant_id, xdf_file_path)
            participant_logger.info("XDF data loaded.")
        except Exception as e:
            participant_logger.error(f"Error loading XDF data from {xdf_file_path}: {e}", exc_info=True)

    # Load Questionnaire
    q_file_name = config['QuestionnaireReader'].get('filename_template', '{participant_id}_questionnaire.txt').format(participant_id=participant_id)
    q_file_path = os.path.join(base_raw_data_dir, participant_id, q_file_name)
    if os.path.exists(q_file_path):
        try:
            participant_data['questionnaire_raw_df'] = questionnaire_reader.load_data(
                file_path=q_file_path,
                file_type=config['QuestionnaireReader'].get('file_type', 'txt'),
                delimiter=config['QuestionnaireReader'].get('delimiter', '\t'),
                participant_id_col=config['QuestionnairePreprocessing'].get('participant_id_column_original', 'Subject')
            )
            participant_logger.info("Questionnaire data loaded.")
        except Exception as e:
            participant_logger.error(f"Error loading questionnaire data from {q_file_path}: {e}", exc_info=True)
    else:
        participant_logger.warning(f"Questionnaire file not found: {q_file_path}")

    return participant_data

def _process_questionnaires(participant_id: str, participant_logger: logging.Logger, config: ConfigParser, raw_q_df: Optional[pd.DataFrame], participant_output_dir: str) -> Dict[str, Any]:
    """Processes and scores questionnaire data."""
    q_artifacts: Dict[str, Any] = {}
    if raw_q_df is not None:
        try:
            q_artifacts['questionnaire_processed_items'] = questionnaire_preprocessor.extract_items(
                input_df=raw_q_df,
                config={
                    'participant_id_column_original': config['QuestionnairePreprocessing'].get('participant_id_column_original', 'Subject'),
                    'item_column_map': dict(config.items('QuestionnaireItemMap')),
                    'output_participant_id_col_name': config['QuestionnairePreprocessing'].get('output_participant_id_col_name', 'participant_id')
                }
            )
            if q_artifacts['questionnaire_processed_items'] is not None:
                scale_defs_config = {}
                for section in config.sections():
                    if section.startswith('ScaleDef_'):
                        scale_name = section.replace('ScaleDef_', '')
                        scale_defs_config[scale_name] = {
                            'items': [item.strip() for item in config[section].get('items', '').split(',')],
                            'scoring_method': config[section].get('scoring_method', 'sum'),
                            'reverse_coded_items': {k: {'min_val': int(v.split(',')[0].split(':')[1]), 'max_val': int(v.split(',')[1].split(':')[1])}
                                                    for k, v in config.items(f"ReverseCoded_{scale_name}")} if config.has_section(f"ReverseCoded_{scale_name}") else {},
                            'min_valid_items_ratio': config[section].getfloat('min_valid_items_ratio', fallback=None)
                        }
                q_artifacts['questionnaire_scored_scales'] = questionnaire_scale_processor.score_scales(
                    data_df=q_artifacts['questionnaire_processed_items'],
                    scale_definitions=scale_defs_config,
                    participant_id_col=config['QuestionnairePreprocessing'].get('output_participant_id_col_name', 'participant_id')
                )
                if q_artifacts['questionnaire_scored_scales'] is not None:
                    score_analysis_results = score_analyzer.analyze_scores(
                        scores_df=q_artifacts['questionnaire_scored_scales'],
                        participant_id_col=config['QuestionnairePreprocessing'].get('output_participant_id_col_name', 'participant_id'),
                        output_dir=os.path.join(participant_output_dir, "QuestionnaireAnalysis"),
                        save_results=True,
                        results_prefix=f"{participant_id}_questionnaire"
                    )
                    q_artifacts['questionnaire_score_analysis'] = score_analysis_results
            participant_logger.info("Questionnaire processing and scoring completed.")
        except Exception as e:
            participant_logger.error(f"Error processing questionnaire data: {e}", exc_info=True)
    else:
        participant_logger.info("No questionnaire data to process.")
    return q_artifacts

def _preprocess_physiological_data(
    participant_id: str,
    participant_logger: logging.Logger,
    config: ConfigParser,
    raw_eeg: Optional[Raw],
    raw_eda_signal: Optional[np.ndarray], eda_sfreq: Optional[float],
    raw_ecg_signal: Optional[np.ndarray], ecg_sfreq: Optional[float],
    raw_fnirs_od: Optional[Raw],
    participant_output_dir: str
) -> Dict[str, Any]:
    """Handles EEG, EDA, ECG, fNIRS preprocessing."""
    physio_artifacts: Dict[str, Any] = {}

    # EEG Preprocessing
    if raw_eeg:
        try:
            processed_eeg_data = eeg_preprocessor.process(
                raw_eeg=raw_eeg,
                eeg_filter_band_config=(config['EEG'].getfloat('eeg_filter_l_freq', 1.0), config['EEG'].getfloat('eeg_filter_h_freq', 40.0)),
                ica_n_components_config=config['EEG'].getint('ica_n_components', 30),
                ica_random_state_config=config['EEG'].getint('ica_random_state', 42),
                ica_accept_labels_config=[label.strip() for label in config['EEG'].get('ica_accept_labels', 'brain,Other').split(',')],
                ica_reject_threshold_config=config['EEG'].getfloat('ica_reject_threshold', 0.8)
            )
            physio_artifacts['eeg_processed_raw'] = processed_eeg_data
            participant_logger.info("EEG preprocessing completed.")
        except Exception as e:
            participant_logger.error(f"Error during EEG preprocessing: {e}", exc_info=True)

    # EDA Preprocessing
    if raw_eda_signal is not None and eda_sfreq is not None:
        try:
            _, _, physio_artifacts['eda_phasic_signal'], physio_artifacts['eda_tonic_signal'] = eda_preprocessor.preprocess_eda(
                eda_signal_raw=raw_eda_signal,
                eda_sampling_rate=eda_sfreq,
                participant_id=participant_id,
                output_dir=participant_output_dir,
                eda_cleaning_method_config=config['EDA'].get('eda_cleaning_method', 'neurokit')
            )
            participant_logger.info("EDA preprocessing completed.")
        except Exception as e:
            participant_logger.error(f"Error during EDA preprocessing: {e}", exc_info=True)

    # ECG Preprocessing
    if raw_ecg_signal is not None and ecg_sfreq is not None:
        try:
            _, physio_artifacts['ecg_rpeaks_samples'] = ecg_preprocessor.preprocess_ecg(
                ecg_signal=raw_ecg_signal,
                ecg_sfreq=ecg_sfreq,
                participant_id=participant_id,
                output_dir=participant_output_dir,
                ecg_rpeak_method_config=config['ECG'].get('ecg_rpeak_method', 'neurokit')
            )
            participant_logger.info("ECG preprocessing completed.")
        except Exception as e:
            participant_logger.error(f"Error during ECG preprocessing: {e}", exc_info=True)

    # fNIRS Preprocessing (Conversion to HbO/HbR)
    if raw_fnirs_od:
        try:
            processed_fnirs = fnirs_preprocessor.process(
                fnirs_raw_od=raw_fnirs_od,
                beer_lambert_ppf_config=config['FNIRS'].getfloat('beer_lambert_ppf'),
                short_channel_regression_config=config['FNIRS'].getboolean('short_channel_regression', True),
                motion_correction_method_config=config['FNIRS']['motion_correction_method'],
                filter_band_config=(config['FNIRS'].getfloat('filter_l_freq', 0.01), config['FNIRS'].getfloat('filter_h_freq', 0.1))
            )
            physio_artifacts['fnirs_processed_hbo_hbr'] = processed_fnirs
            
            if processed_fnirs:
                fnirs_quality_metrics = data_quality_reporter.report_fnirs_quality(
                    raw_od=raw_fnirs_od,
                    raw_haemo=processed_fnirs,
                    participant_id=participant_id
                )
                physio_artifacts['fnirs_quality_report'] = fnirs_quality_metrics

            participant_logger.info("fNIRS preprocessing (HbO/HbR conversion) completed.")
        except Exception as e:
            participant_logger.error(f"Error during fNIRS preprocessing (HbO/HbR conversion): {e}", exc_info=True)

    return physio_artifacts

def _extract_physiological_features(
    participant_id: str,
    participant_logger: logging.Logger,
    config: ConfigParser,
    raw_eeg: Optional[Raw],
    rpeaks_samples_array: Optional[np.ndarray],
    phasic_eda: Optional[np.ndarray], eda_sfreq: Optional[float],
    participant_output_dir: str
) -> Dict[str, Any]:
    """Handles extraction of physiological features like HRV and SCR."""
    feature_artifacts: Dict[str, Any] = {}

    # HRV Processing (Continuous)
    if rpeaks_samples_array is not None and raw_eeg is not None:
        try:
            _, _, continuous_hrv_signal, continuous_hrv_time_vector, _ = ecg_hrv_processor.process_rpeaks_to_hrv(
                rpeaks_samples=rpeaks_samples_array,
                original_sfreq=raw_eeg.info['sfreq'],
                participant_id=participant_id,
                output_dir=participant_output_dir,
                target_sfreq_continuous_hrv=config['HRVProcessing'].getfloat('target_sfreq_continuous_hrv', ECGHRVProcessor.DEFAULT_TARGET_SFREQ_CONTINUOUS_HRV)
            )
            feature_artifacts['hrv_continuous_signal'] = continuous_hrv_signal
            feature_artifacts['hrv_continuous_time_vector'] = continuous_hrv_time_vector
            participant_logger.info("HRV continuous signal extracted.")
        except Exception as e:
            participant_logger.error(f"Error during HRV continuous signal extraction: {e}", exc_info=True)

    # SCR Feature Extraction
    if phasic_eda is not None and eda_sfreq is not None:
        try:
            _, scr_features_df = eda_scr_processor.process_phasic_to_scr_features(
                phasic_eda_signal=phasic_eda,
                eda_sampling_rate=int(eda_sfreq),
                participant_id=participant_id,
                output_dir=participant_output_dir,
                scr_peak_method=config['SCRProcessing'].get('scr_peak_method', EDASCRProcessor.DEFAULT_SCR_PEAK_METHOD),
                scr_amplitude_min=config['SCRProcessing'].getfloat('scr_amplitude_min', EDASCRProcessor.DEFAULT_SCR_AMPLITUDE_MIN)
            )
            feature_artifacts['eda_scr_features_df'] = scr_features_df
            participant_logger.info("SCR features extracted.")
        except Exception as e:
            participant_logger.error(f"Error during SCR feature extraction: {e}", exc_info=True)

    return feature_artifacts

def _segment_data(
    participant_logger: logging.Logger,
    config: ConfigParser,
    processed_eeg: Optional[Raw],
    xdf_lsl_markers: Optional[pd.DataFrame]
) -> Optional[Epochs]:
    """Handles epoching of processed EEG data using EEGEpochProcessor."""
    epochs: Optional[Epochs] = None
    if processed_eeg is not None and xdf_lsl_markers is not None:
        try:
            mne_events, event_id_map, _ = mne_event_handler.create_events(events_df=xdf_lsl_markers, sfreq=processed_eeg.info['sfreq'])
            if mne_events is not None and mne_events.size > 0:
                participant_logger.info(f"Creating epochs with tmin={config['Segmentation'].getfloat('trial_start_offset')}, tmax={config['Segmentation'].getfloat('trial_end_offset')}")
                epochs = eeg_epoch_processor.create_epochs(
                    raw_processed=processed_eeg,
                    events=mne_events,
                    event_id=event_id_map,
                    tmin=config['Segmentation'].getfloat('trial_start_offset'),
                    tmax=config['Segmentation'].getfloat('trial_end_offset'),
                    baseline=None,
                    preload=config['Segmentation'].getboolean('preload_epochs', EEGEpochProcessor.DEFAULT_EPOCH_PRELOAD),
                )
                if epochs is not None:
                    participant_logger.info(f"Epoching completed. Created {len(epochs)} epochs.")
            else:
                participant_logger.warning("No MNE events created from markers. Skipping epoching.")
        except Exception as e:
            participant_logger.error(f"Error during epoching: {e}", exc_info=True)
    else:
        participant_logger.info("Processed EEG or markers not available. Skipping epoching.")
    return epochs

def _run_physiological_analyses(
    participant_id: str, participant_logger: logging.Logger, config: ConfigParser,
    epochs: Optional[Epochs],
    phasic_eda: Optional[np.ndarray], eda_sfreq: Optional[float],
    continuous_hrv_signal: Optional[np.ndarray],
    raw_eeg: Optional[Raw],
    processed_fnirs_hbo_hbr: Optional[Raw],
    xdf_lsl_markers: Optional[pd.DataFrame], fnirs_stream_start_time_xdf: Optional[float]
) -> Dict[str, Any]:
    """Runs physiological analyses (PLV, FAI, GLM)."""
    analysis_artifacts: Dict[str, Any] = {}

    if epochs:
        #  PLV (Neural-Autonomic Synchrony)
        if phasic_eda is not None and eda_sfreq is not None and continuous_hrv_signal is not None and raw_eeg is not None:
            plv_results_list = []
            plv_bands_eeg = parse_bands_config(config['PLV'].get('eeg_bands_config', 'Alpha:(8,13)'))
            eeg_channels_for_plv_list = config['PLV'].get('eeg_channels').split(',')

            # --- EEG-HRV PLV ---
            if continuous_hrv_signal is not None and config.has_option('HRVProcessing', 'target_sfreq_continuous_hrv'):
                try:
                    hrv_sfreq_for_plv = config['HRVProcessing'].getfloat('target_sfreq_continuous_hrv')
                    hrv_plv_df = connectivity_analyzer.calculate_epoched_vs_continuous_plv(
                        signal1_epochs=epochs,
                        signal1_channels_to_average=eeg_channels_for_plv_list,
                        signal1_bands_config=plv_bands_eeg,
                        signal1_original_sfreq_for_event_timing=raw_eeg.info['sfreq'],
                        signal2_continuous_data=continuous_hrv_signal,
                        signal2_sfreq=hrv_sfreq_for_plv,
                        signal1_name="EEG", signal2_name="HRV", participant_id=participant_id
                    )
                    if hrv_plv_df is not None and not hrv_plv_df.empty: plv_results_list.append(hrv_plv_df)
                    participant_logger.info("EEG-HRV PLV analysis completed.")
                except Exception as e: participant_logger.error(f"Error during EEG-HRV PLV analysis: {e}", exc_info=True)

            # --- EEG-EDA PLV ---
            if phasic_eda is not None and eda_sfreq is not None:
                try:
                    eda_plv_df = connectivity_analyzer.calculate_epoched_vs_continuous_plv(
                        signal1_epochs=epochs,
                        signal1_channels_to_average=eeg_channels_for_plv_list,
                        signal1_bands_config=plv_bands_eeg,
                        signal1_original_sfreq_for_event_timing=raw_eeg.info['sfreq'],
                        signal2_continuous_data=phasic_eda,
                        signal2_sfreq=eda_sfreq,
                        signal1_name="EEG", signal2_name="EDA", participant_id=participant_id
                    )
                    if eda_plv_df is not None and not eda_plv_df.empty: plv_results_list.append(eda_plv_df)
                    participant_logger.info("EEG-EDA PLV analysis completed.")
                except Exception as e: participant_logger.error(f"Error during EEG-EDA PLV analysis: {e}", exc_info=True)

            if plv_results_list:
                analysis_artifacts['plv_results_df'] = pd.concat(plv_results_list, ignore_index=True)
            else:
                analysis_artifacts['plv_results_df'] = pd.DataFrame()

        #  Frontal Asymmetry Index
        try:
            psd_channels = config['FAI'].get('channel_left').split(',') + config['FAI'].get('channel_right').split(',')
            fai_bands = parse_bands_config(config['FAI'].get('bands_config', 'Alpha:(8,13)'))
            psd_results = psd_analyzer.calculate_psd_from_epochs(
                epochs_processed_all_conditions=epochs, bands_config=fai_bands, psd_channels_of_interest=psd_channels
            )
            analysis_artifacts['psd_results_for_fai'] = psd_results
            if psd_results:
                fai_results_dict: Dict[str,Any] = {} # This will store {'band_name': {'condition': {'pair_name': fai_value}}}
                for condition_name, band_data in psd_results.items():
                    for band_name_fai in fai_bands.keys():
                        if band_name_fai in band_data: # Check if the band exists for this condition
                            # The FAIAnalyzer expects psd_results_all_bands to be {'condition': {'band': {'channel': power}}}
                            # We need to pass the full psd_results to FAIAnalyzer, and it will pick the correct band internally.
                            fai_res_for_band_and_condition = fai_analyzer.compute_fai_from_psd(
                                psd_results_all_bands={condition_name: psd_results[condition_name]}, # Pass only current condition's data
                                fai_band_name=band_name_fai,
                                fai_electrode_pairs_config=[(config['FAI']['channel_left'], config['FAI']['channel_right'])]
                            )
                            # fai_res_for_band_and_condition is {'condition': {'pair_name': fai_value}}
                            # We want to structure analysis_artifacts['fai_results'] as {'band_name': {'condition': {'pair_name': fai_value}}}
                            if band_name_fai not in fai_results_dict:
                                fai_results_dict[band_name_fai] = {}
                            if condition_name in fai_res_for_band_and_condition: # Check if FAI was computed for this condition
                                fai_results_dict[band_name_fai][condition_name] = fai_res_for_band_and_condition[condition_name]
                analysis_artifacts['fai_results'] = fai_results_dict
            participant_logger.info("FAI analysis completed.")
        except Exception as e: participant_logger.error(f"Error during FAI analysis: {e}", exc_info=True)

    #  GLM Analysis (fNIRS)
    if processed_fnirs_hbo_hbr is not None and xdf_lsl_markers is not None and fnirs_stream_start_time_xdf is not None:
        try:
            design_matrix = fnirs_design_matrix_preprocessor.create_design_matrix(
                participant_id=participant_id, xdf_markers_df=xdf_lsl_markers, raw_fnirs_data=processed_fnirs_hbo_hbr,
                fnirs_stream_start_time_xdf=fnirs_stream_start_time_xdf,
                event_mapping_config={int(k): v for k, v in config.items('FNIRS_EventMapping')},
                condition_duration_config={k: float(v) for k, v in config.items('FNIRS_ConditionDurations')}
            )
            analysis_artifacts['fnirs_design_matrix'] = design_matrix
            if design_matrix is not None:
                contrasts = {name: {cond: float(weight) for cond, weight in (pair.split(':') for pair in pairs.split(','))}
                             for name, pairs in config.items('FNIRS_Contrasts')}
                analysis_artifacts['fnirs_glm_results'] = glm_analyzer.run_first_level_glm(
                    data_for_glm=processed_fnirs_hbo_hbr, design_matrix_prepared=design_matrix,
                    participant_id=participant_id, contrasts_config=contrasts
                )
            participant_logger.info("fNIRS GLM analysis completed.")
        except Exception as e: participant_logger.error(f"Error during fNIRS GLM analysis: {e}", exc_info=True)

    return analysis_artifacts

def _generate_participant_plots(
    participant_id: str,
    participant_artifacts: Dict[str, Any],
    plot_reporter_instance: PlotReporter,
) -> None:
    """
    Generates and saves plots for a single participant based on available artifacts.
    This helper uses the PlotReporter instance.
    """
    participant_logger = plot_reporter_instance.logger # Use logger from plot_reporter for consistency here

    # Plot PLV results
    plv_df_results = participant_artifacts.get('plv_results_df')
    if isinstance(plv_df_results, pd.DataFrame) and not plv_df_results.empty:
        required_plv_cols = [
            ConnectivityAnalyzer.OUTPUT_COL_CONDITION,
            ConnectivityAnalyzer.OUTPUT_COL_PLV_VALUE,
            ConnectivityAnalyzer.OUTPUT_COL_SIGNAL1_BAND,
            ConnectivityAnalyzer.OUTPUT_COL_MODALITY_PAIR
        ]
        if all(col in plv_df_results.columns for col in required_plv_cols):
            plot_reporter_instance.generate_plot(
                participant_id_or_group=participant_id,
                plot_config={
                    'plot_type': 'faceted_barplot',
                    'data_mapping': {'plot_data_df': 'self'}, # 'self' means data_payload is the df
                    'plot_params': {
                        'title': f"PLV Results for {participant_id}",
                        'x_col': ConnectivityAnalyzer.OUTPUT_COL_CONDITION,
                        'y_col': ConnectivityAnalyzer.OUTPUT_COL_PLV_VALUE,
                        'hue_col': ConnectivityAnalyzer.OUTPUT_COL_SIGNAL1_BAND,
                        'facet_col': ConnectivityAnalyzer.OUTPUT_COL_MODALITY_PAIR
                    },
                    'plot_category_subdir': 'PLV_Plots'
                },
                data_payload=plv_df_results
            )
        else:
            missing_cols = [col for col in required_plv_cols if col not in plv_df_results.columns]
            participant_logger.warning(f"Skipping PLV plot for {participant_id}: Results DataFrame missing expected columns: {missing_cols}")

    # Plot FAI results
    # FAI results structure from orchestrator: {'band_name': {'condition': {'pair_name': fai_value}}}
    fai_results_bands = participant_artifacts.get('fai_results')
    if isinstance(fai_results_bands, dict):
        fai_plot_df_list = []
        for band_name, conditions_data in fai_results_bands.items():
            if isinstance(conditions_data, dict):
                for condition, pairs_data in conditions_data.items():
                    if isinstance(pairs_data, dict):
                        for pair_name, fai_val in pairs_data.items():
                            if fai_val is not None and not np.isnan(fai_val):
                                fai_plot_df_list.append({
                                    'participant_id': participant_id,
                                    'condition': condition,
                                    'band': band_name,
                                    'electrode_pair': pair_name,
                                    'fai_value': fai_val
                                })
                    else:
                        participant_logger.debug(f"FAI data for band '{band_name}', condition '{condition}' for {participant_id} is not a dict. Skipping FAI plot for this part.")
            else:
                participant_logger.debug(f"FAI data for band '{band_name}' for {participant_id} is not a dict. Skipping FAI plot for this band.")

        if fai_plot_df_list:
            fai_plot_df = pd.DataFrame(fai_plot_df_list)
            if not fai_plot_df.empty:
                plot_reporter_instance.generate_plot(
                    participant_id_or_group=participant_id,
                    plot_config={
                        'plot_type': 'faceted_barplot', # Using faceted_barplot for consistency, can be 'barplot'
                        'data_mapping': {'plot_data_df': 'self'},
                        'plot_params': {
                            'title': f"FAI Results for {participant_id}",
                            'x_col': 'condition',
                            'y_col': 'fai_value',
                            'hue_col': 'electrode_pair',
                            'facet_col': 'band', # Facet by band
                            'y_axis_label': 'FAI Value'
                        },
                        'plot_category_subdir': 'FAI_Plots'
                    },
                    data_payload=fai_plot_df
                )
        else:
            participant_logger.info(f"No valid FAI data to plot for {participant_id} after processing.")

    # Plot GLM results
    fnirs_glm_results = participant_artifacts.get('fnirs_glm_results')
    if fnirs_glm_results and isinstance(fnirs_glm_results, dict) and 'contrast_results' in fnirs_glm_results:
        contrast_dfs = fnirs_glm_results.get('contrast_results')
        if isinstance(contrast_dfs, dict):
            for contrast_name, contrast_df in contrast_dfs.items():
                if isinstance(contrast_df, pd.DataFrame) and not contrast_df.empty and \
                   'ch_name' in contrast_df.columns and 't_value' in contrast_df.columns:
                    plot_reporter_instance.generate_plot(
                        participant_id_or_group=participant_id,
                        plot_config={
                            'plot_type': 'horizontal_barplot',
                            'data_mapping': {'plot_data_df': 'self'},
                            'plot_params': {
                                'title': f"GLM T-values for {contrast_name} ({participant_id})",
                                'y_col': 'ch_name',
                                'x_col': 't_value',
                                'show_zeroline': True
                            },
                            'plot_category_subdir': f'GLM_Plots/{contrast_name}'
                        },
                        data_payload=contrast_df
                    )
                elif isinstance(contrast_df, pd.DataFrame) and not contrast_df.empty:
                    participant_logger.warning(f"Skipping GLM plot for contrast '{contrast_name}' for {participant_id}: Results DataFrame missing 'ch_name' or 't_value'.")
                elif contrast_df is not None:
                    participant_logger.debug(f"GLM contrast '{contrast_name}' results for {participant_id} is not a non-empty DataFrame. Skipping plot.")


def run_participant_analysis(participant_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates the analysis for a single participant.
    """
    participant_id = participant_config['participant_id']
    participant_output_dir = os.path.join(base_output_dir, participant_id)
    os.makedirs(participant_output_dir, exist_ok=True)

    # Setup participant-specific logger
    log_reporter_instance = LogReporter(participant_base_output_dir=participant_output_dir,
                                        participant_id=participant_id,
                                        log_level_str=config['DEFAULT']['log_level'])
    participant_logger = log_reporter_instance.get_logger()
    participant_logger.info(f"--- Starting analysis for participant: {participant_id} ---")

    # Initialize results dictionary for this participant
    participant_artifacts: Dict[str, Any] = {'participant_id': participant_id}

    try:
        # === 1. Data Loading (XDF) ===
        raw_data = _load_participant_data(participant_id, participant_logger, config)
        participant_artifacts.update(raw_data)

        streams = participant_artifacts.get('xdf_streams', {})
        raw_eeg: Optional[Raw] = streams.get('eeg')
        raw_ecg_signal: Optional[np.ndarray] = streams.get('ecg_signal')
        ecg_sfreq: Optional[float] = streams.get('ecg_sfreq')
        raw_eda_signal: Optional[np.ndarray] = streams.get('eda_signal')
        eda_sfreq: Optional[float] = streams.get('eda_sfreq')
        raw_fnirs_od: Optional[Raw] = streams.get('fnirs_od')
        fnirs_stream_start_time_xdf: Optional[float] = streams.get('fnirs_stream_start_time_xdf')
        xdf_lsl_markers: Optional[pd.DataFrame] = streams.get('xdf_markers_df')
        raw_q_df: Optional[pd.DataFrame] = participant_artifacts.get('questionnaire_raw_df')

        # === 1.1 Process Questionnaire Data ===
        q_artifacts = _process_questionnaires(participant_id, participant_logger, config, raw_q_df, participant_output_dir)
        participant_artifacts.update(q_artifacts)

        # === 2. Preprocessing Physiological Data ===
        physio_preproc_artifacts = _preprocess_physiological_data(
            participant_id, participant_logger, config,
            raw_eeg, raw_eda_signal, eda_sfreq,
            raw_ecg_signal, ecg_sfreq, raw_fnirs_od,
            participant_output_dir
        )
        participant_artifacts.update(physio_preproc_artifacts)

        processed_eeg: Optional[Raw] = participant_artifacts.get('eeg_processed_raw')
        phasic_eda: Optional[np.ndarray] = participant_artifacts.get('eda_phasic_signal')
        rpeaks_samples_array: Optional[np.ndarray] = participant_artifacts.get('ecg_rpeaks_samples')
        processed_fnirs_hbo_hbr: Optional[Raw] = participant_artifacts.get('fnirs_processed_hbo_hbr')

        if processed_eeg:
            eeg_quality_metrics = data_quality_reporter.report_eeg_quality(
                raw_original=raw_eeg,
                raw_processed=processed_eeg,
                participant_id=participant_id
            )
            participant_artifacts['eeg_quality_report'] = eeg_quality_metrics

        # === 3. Processing (Feature Extraction) ===
        feature_artifacts = _extract_physiological_features(
            participant_id, participant_logger, config,
            raw_eeg, rpeaks_samples_array, phasic_eda, eda_sfreq,
            participant_output_dir
        )
        participant_artifacts.update(feature_artifacts)
        continuous_hrv_signal: Optional[np.ndarray] = participant_artifacts.get('hrv_continuous_signal')

        # === 4. Segmentation (Epoching) ===
        epochs = _segment_data(participant_logger, config, processed_eeg, xdf_lsl_markers)
        participant_artifacts['eeg_epochs'] = epochs

        # === 5. Analyses ===
        analysis_artifacts = _run_physiological_analyses(
            participant_id, participant_logger, config,
            epochs, phasic_eda, eda_sfreq, continuous_hrv_signal,
            raw_eeg, processed_fnirs_hbo_hbr, xdf_lsl_markers, fnirs_stream_start_time_xdf
        )
        participant_artifacts.update(analysis_artifacts)

        # --- Handle Plotting after analyses ---
        _generate_participant_plots(participant_id, participant_artifacts, plot_reporter)

        # --- Save Data Quality Reports ---
        if 'eeg_quality_report' in participant_artifacts:
            eeg_qc_path = os.path.join(participant_output_dir, "QualityControl", f"{participant_id}_eeg_quality_report.json")
            os.makedirs(os.path.dirname(eeg_qc_path), exist_ok=True)
            data_quality_reporter.save_report_to_json(participant_artifacts['eeg_quality_report'], eeg_qc_path)
        if 'fnirs_quality_report' in participant_artifacts:
            fnirs_qc_path = os.path.join(participant_output_dir, "QualityControl", f"{participant_id}_fnirs_quality_report.json")
            os.makedirs(os.path.dirname(fnirs_qc_path), exist_ok=True)
            data_quality_reporter.save_report_to_json(participant_artifacts['fnirs_quality_report'], fnirs_qc_path)

        # Save all collected participant artifacts to disk
        artifacts_save_path = os.path.join(participant_output_dir, f"{participant_id}_artifacts.pkl")
        try:
            with open(artifacts_save_path, 'wb') as f_save:
                pickle.dump(participant_artifacts, f_save)
            participant_logger.info(f"Participant artifacts saved to {artifacts_save_path}")
        except Exception as e_save_artifact:
            participant_logger.error(f"Failed to save participant artifacts to {artifacts_save_path}: {e_save_artifact}")

        participant_logger.info(f"--- Participant {participant_id} analysis completed successfully. ---")
        return {'participant_id': participant_id, 'status': 'completed', 'artifacts': participant_artifacts}
    except Exception as e_participant:
        participant_logger.error(f"CRITICAL ERROR during participant {participant_id} analysis: {e_participant}", exc_info=True)
        return {'participant_id': participant_id, 'status': 'failed', 'error': str(e_participant), 'artifacts': participant_artifacts}
    finally:
        participant_logger.info(f"--- Closing log handler for participant: {participant_id} ---")
        log_reporter_instance.close_handlers()


# --- Functions for continuous monitoring ---
PROCESSED_PARTICIPANTS_LOG_FILE = os.path.join(base_output_dir, "processed_participants.log")

def load_processed_participants() -> set:
    processed = set()
    if os.path.exists(PROCESSED_PARTICIPANTS_LOG_FILE):
        try:
            with open(PROCESSED_PARTICIPANTS_LOG_FILE, 'r') as f:
                for line in f:
                    processed.add(line.strip())
        except Exception as e:
            logger.error(f"Error loading processed participants log: {e}")
    return processed

def save_processed_participant(participant_id: str):
    try:
        with open(PROCESSED_PARTICIPANTS_LOG_FILE, 'a') as f:
            f.write(f"{participant_id}\n")
    except Exception as e:
        logger.error(f"Error saving processed participant {participant_id} to log: {e}")

def discover_new_participants(all_participant_dirs: List[str], processed_ids: set) -> List[str]:
    new_pids = []
    for p_dir_name in all_participant_dirs:
        if p_dir_name not in processed_ids and os.path.isdir(os.path.join(base_raw_data_dir, p_dir_name)):
            expected_xdf_path = os.path.join(base_raw_data_dir, p_dir_name, f"{p_dir_name}.xdf")
            if os.path.exists(expected_xdf_path):
                new_pids.append(p_dir_name)
            else:
                logger.debug(f"Directory {p_dir_name} found but missing expected XDF file. Not considered new.")
    return new_pids

# ### Main Execution ###
if __name__ == "__main__":
    logger.info("=== Starting EmotiView Analysis Pipeline ===")
    polling_interval_seconds = config['ContinuousMode'].getint('polling_interval_seconds', 60)
    processed_participant_ids = load_processed_participants()
    logger.info(f"Loaded {len(processed_participant_ids)} previously processed participant IDs.")

    try:
        while True:
            logger.info(f"Scanning for new participants in {base_raw_data_dir}...")
            try:
                all_dirs_in_raw_data = [d for d in os.listdir(base_raw_data_dir) if os.path.isdir(os.path.join(base_raw_data_dir, d))]
            except FileNotFoundError:
                logger.error(f"Raw data directory not found: {base_raw_data_dir}. Please check config. Retrying in {polling_interval_seconds}s.")
                time.sleep(polling_interval_seconds)
                continue

            new_participants_to_process = discover_new_participants(all_dirs_in_raw_data, processed_participant_ids)

            if not new_participants_to_process:
                logger.info(f"No new participants found. Sleeping for {polling_interval_seconds} seconds.")
            else:
                logger.info(f"Found {len(new_participants_to_process)} new participant(s) to process: {new_participants_to_process}")
                for pid in new_participants_to_process:
                    os.makedirs(os.path.join(base_output_dir, pid), exist_ok=True)

                task_configs_list = [{'participant_id': pid} for pid in new_participants_to_process]
                parallel_runner = ParallelTaskRunner(task_function=run_participant_analysis,
                                                     task_configs=task_configs_list,
                                                     main_logger_name=main_logger_name,
                                                     max_workers=config['Parallel'].getint('max_workers'))
                all_results = parallel_runner.run()

                newly_successful_runs = 0
                newly_failed_runs = 0
                newly_processed_artifacts_list = []

                for result in all_results:
                    p_id = result.get('participant_id', 'N/A')
                    status = result.get('status', 'unknown_status')
                    logger.info(f"Participant {p_id} processing result: {status}")
                    if status == 'completed':
                        newly_successful_runs += 1
                        save_processed_participant(p_id)
                        processed_participant_ids.add(p_id)
                        if 'artifacts' in result:
                            newly_processed_artifacts_list.append(result['artifacts'])
                    else:
                        newly_failed_runs += 1
                    if result.get('error'):
                        logger.error(f"  Error for {p_id}: {result.get('error')}")
                logger.info(f"Current iteration: Processed {len(all_results)} new participants. Successful: {newly_successful_runs}, Failed: {newly_failed_runs}.")

                if newly_successful_runs > 0:
                    logger.info("\n=== (Re)Starting Group Level Processing and Analysis due to new data ===")
                    
                    all_loaded_participant_artifacts_for_group_run = []
                    current_processed_ids_for_group_run = load_processed_participants()
                    logger.info(f"Attempting to load artifacts for {len(current_processed_ids_for_group_run)} participants for group analysis.")

                    for pid_for_group in current_processed_ids_for_group_run:
                        participant_output_dir_for_load = os.path.join(base_output_dir, pid_for_group)
                        artifact_pkl_path = os.path.join(participant_output_dir_for_load, f"{pid_for_group}_artifacts.pkl")
                        if os.path.exists(artifact_pkl_path):
                            try:
                                with open(artifact_pkl_path, 'rb') as f_load:
                                    loaded_p_artifacts = pickle.load(f_load)
                                    all_loaded_participant_artifacts_for_group_run.append(loaded_p_artifacts)
                                logger.debug(f"Successfully loaded artifacts for {pid_for_group} from {artifact_pkl_path}")
                            except Exception as e_load_artifact:
                                logger.error(f"Failed to load artifacts for {pid_for_group} from {artifact_pkl_path}: {e_load_artifact}")
                        else:
                            logger.warning(f"Artifact file not found for {pid_for_group} at {artifact_pkl_path}. Skipping for group analysis.")

                    if not all_loaded_participant_artifacts_for_group_run:
                        logger.warning("No participant artifacts could be loaded for group analysis. Skipping group level.")
                    else:
                        group_results_dir_name = config['GroupLevelSettings'].get('results_dir_name', 'EV_GroupResults')
                        group_output_dir = os.path.join(base_output_dir, group_results_dir_name)
                        os.makedirs(group_output_dir, exist_ok=True)
                        logger.info(f"Group level results will be saved to: {group_output_dir}")
                        group_level_data_artifacts: Dict[str, pd.DataFrame] = {}

                        for section_name in config.sections():
                            if section_name.startswith('GroupPreproc_'):
                                step_name = section_name.replace('GroupPreproc_', '')
                                preproc_config = dict(config.items(section_name))
                                logger.info(f"Running group preprocessing step: {step_name}")
                                processed_group_df = group_level_processor.aggregate_data(
                                    all_participant_artifacts=all_loaded_participant_artifacts_for_group_run,
                                    aggregation_config=preproc_config,
                                    task_name=step_name
                                )
                                if processed_group_df is not None:
                                    output_key = preproc_config.get('output_artifact_key')
                                    if output_key:
                                        group_level_data_artifacts[output_key] = processed_group_df
                                        logger.info(f"Group preprocessing step '{step_name}' completed. Result stored as '{output_key}'. Shape: {processed_group_df.shape}")
                                    else:
                                        logger.warning(f"Group preprocessing step '{step_name}' completed but no 'output_artifact_key' defined. Result not stored.")
                                else:
                                    logger.error(f"Group preprocessing step '{step_name}' failed or returned no data.")

                        for section_name in config.sections():
                            if section_name.startswith('GroupAnalysis_'):
                                analysis_name = section_name.replace('GroupAnalysis_', '')
                                analysis_config_raw = dict(config.items(section_name))
                                logger.info(f"Running group analysis step: {analysis_name}")

                                input_data_key = analysis_config_raw.get('input_data_key')
                                if not input_data_key or input_data_key not in group_level_data_artifacts:
                                    logger.error(f"Input data key '{input_data_key}' for group analysis '{analysis_name}' not found in processed group artifacts. Skipping.")
                                    continue
                                group_df_for_analysis = group_level_data_artifacts[input_data_key]
                                method_params = {}
                                for key, value in analysis_config_raw.items():
                                    if key.startswith('method_param_') and key != 'method_param_p_value_col_name' and key != 'method_param_apply_fdr':
                                        param_name = key.replace('method_param_', '')
                                        if param_name in ['within', 'between', 'grouping_vars'] and isinstance(value, str) and ',' in value:
                                            method_params[param_name] = [v.strip() for v in value.split(',')]
                                        else:
                                            method_params[param_name] = value
                                analysis_step_config = {
                                    'analyzer_key': analysis_config_raw.get('analyzer_key'),
                                    'method_to_call': analysis_config_raw.get('method_to_call'),
                                    'method_params': method_params
                                }
                                analysis_results_df = group_level_analyzer.analyze_data(
                                    data_df=group_df_for_analysis,
                                    analysis_config=analysis_step_config,
                                    task_name=analysis_name
                                )
                                if analysis_results_df is not None:
                                    apply_fdr_flag = analysis_config_raw.getboolean('method_param_apply_fdr', fallback=False)
                                    p_val_col_for_fdr = analysis_config_raw.get('method_param_p_value_col_name', 'p-unc')
                                    if apply_fdr_flag and isinstance(analysis_results_df, pd.DataFrame) and p_val_col_for_fdr in analysis_results_df.columns:
                                        logger.info(f"Orchestrator (Group Analysis '{analysis_name}'): Applying FDR correction to column '{p_val_col_for_fdr}'.")
                                        p_values_numeric = pd.to_numeric(analysis_results_df[p_val_col_for_fdr], errors='coerce')
                                        rejected, p_corrected = fdr_corrector.apply_fdr_correction(p_values_numeric.dropna().values)
                                        p_corrected_series = pd.Series(p_corrected, index=p_values_numeric.dropna().index)
                                        analysis_results_df[f'{p_val_col_for_fdr}_corrected_fdr'] = p_corrected_series
                                        rejected_series = pd.Series(rejected, index=p_values_numeric.dropna().index)
                                        analysis_results_df[f'rejected_fdr'] = rejected_series
                                        logger.info(f"Orchestrator (Group Analysis '{analysis_name}'): FDR correction applied.")

                                    output_filename = analysis_config_raw.get('output_filename_csv')
                                    if output_filename:
                                        output_path = os.path.join(group_output_dir, output_filename)
                                        analysis_results_df.to_csv(output_path, index=True)
                                        logger.info(f"Group analysis '{analysis_name}' results saved to: {output_path}")
                else:
                    logger.info("No new participants successfully processed in this iteration. Group analysis not re-run.")

            if git_handler and new_participants_to_process:
                try:
                    git_handler.commit_and_sync_changes(paths_to_add=["."], commit_message=f"Automated: Processed new participants and updated group results. New: {new_participants_to_process}")
                except Exception as e_git_iter:
                    logger.error(f"Failed to perform Git commit during iteration: {e_git_iter}")

            time.sleep(polling_interval_seconds)

    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user (KeyboardInterrupt).")
    except Exception as e_main_loop:
        logger.critical(f"Critical error in orchestrator main loop: {e_main_loop}", exc_info=True)
    finally:
        logger.info("=== EmotiView Analysis Pipeline Shutting Down ===")
        if git_handler:
            try:
                logger.info("Attempting final Git commit on shutdown...")
                git_handler.commit_and_sync_changes(paths_to_add=["."], commit_message="Automated: Final commit on orchestrator shutdown")
            except Exception as e_git_shutdown:
                logger.error(f"Failed to perform final Git commit on shutdown: {e_git_shutdown}")