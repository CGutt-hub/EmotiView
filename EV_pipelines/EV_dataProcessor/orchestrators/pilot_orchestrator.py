# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\pilot_orchestrator.py
import os
import mne
import pandas as pd
import numpy as np
import logging # For main logger
from . import config # Corrected: config is in the same directory
from ..preprocessing.eeg_preprocessor import EEGPreprocessor
from ..preprocessing.ecg_preprocessor import ECGPreprocessor
from ..preprocessing.eda_preprocessor import EDAPreprocessor
from ..preprocessing.fnirs_preprocessor import FNIRSPreprocessor
from ..analysis.analysis_service import AnalysisService
from ..reporting.plotting_service import PlottingService
from ..data_handling.data_loader import DataLoader # Corrected: DataLoader is in data_handling
from ..utils.event_processor import EventProcessor # Import the new EventProcessor
from ..utils.participant_logger import ParticipantLogger # Adjusted: Assuming ParticipantLogger is in the main utils
# Import helper functions from the new helpers.py file within the utils folder
from ..utils.helpers import select_eeg_channels_by_fnirs_rois, apply_fdr_correction 

# Constants for data types
DATA_TYPES = ['eeg', 'fnirs', 'ecg', 'eda', 'events', 'survey'] 
EMOTIONAL_CONDITIONS = ['Positive', 'Negative', 'Neutral'] 

FAI_ELECTRODE_PAIRS = config.FAI_ELECTRODE_PAIRS 

def create_output_directories(base_dir, participant_id):
    """Creates necessary output directories for a participant."""
    dirs = {
        'base_participant': os.path.join(base_dir, participant_id),
        'raw_data_copied': os.path.join(base_dir, participant_id, 'raw_data_copied'), 
        'preprocessed_data': os.path.join(base_dir, participant_id, 'preprocessed_data'),
        'analysis_results': os.path.join(base_dir, participant_id, 'analysis_results'),
        'plots_root': os.path.join(base_dir, participant_id, 'plots') 
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def process_participant_data(participant_id, participant_raw_xdf_data_dir, eprime_txt_file_path, output_dirs, global_output_base_dir, p_logger):
    """
    Main processing pipeline for a single participant.
    """
    p_logger.info(f"Starting processing for participant: {participant_id}")

    preproc_results_dir = output_dirs['preprocessed_data']
    analysis_results_dir = output_dirs['analysis_results'] # Ensure this is used or passed where needed

    data_loader = DataLoader(p_logger)
    analysis_service = AnalysisService(p_logger) 
    plotting_service = PlottingService(p_logger, output_dirs['plots_root'])

    processed_data_artifacts = {
        'participant_id': participant_id,
        'log_file': p_logger.handlers[0].baseFilename if p_logger.handlers else None,
        'raw_data_paths': {}, 
        'event_times_df': None, 
        'preprocessed_data_paths': {},
        'mne_objects': {'raw': {}, 'epochs': {}},
        'analysis_outputs': {'metrics': {}, 'stats': {}, 'dataframes': {}, 'metadata': {}}
    }

    p_logger.info("--- Loading Data ---")
    loaded_physiological_data = {}
    # Use the direct path to the directory containing XDF files
    if participant_raw_xdf_data_dir and os.path.exists(participant_raw_xdf_data_dir):
         loaded_physiological_data = data_loader.load_participant_streams(participant_id, participant_raw_xdf_data_dir)
    else:
        p_logger.error(f"Could not determine or access raw data directory for XDF loading. Path checked: {participant_raw_xdf_data_dir}")

    raw_eeg_mne = loaded_physiological_data.get('eeg')
    eeg_sampling_rate = raw_eeg_mne.info['sfreq'] if raw_eeg_mne else None
    processed_data_artifacts['mne_objects']['raw']['eeg'] = raw_eeg_mne
    
    fnirs_od_mne = loaded_physiological_data.get('fnirs_od')
    fnirs_sampling_rate = fnirs_od_mne.info['sfreq'] if fnirs_od_mne else None
    processed_data_artifacts['mne_objects']['raw']['fnirs_od'] = fnirs_od_mne

    ecg_data_raw = loaded_physiological_data.get('ecg_signal')
    ecg_sampling_rate = loaded_physiological_data.get('ecg_sfreq')
    ecg_times_abs = loaded_physiological_data.get('ecg_times') 
    
    eda_data_raw = loaded_physiological_data.get('eda_signal')
    eda_sampling_rate = loaded_physiological_data.get('eda_sfreq')
    eda_times_abs = loaded_physiological_data.get('eda_times') 
    
    processed_data_artifacts['baseline_start_time_sec'] = loaded_physiological_data.get('baseline_start_time_sec')
    processed_data_artifacts['baseline_end_time_sec'] = loaded_physiological_data.get('baseline_end_time_sec')
    
    # eprime_txt_file_path is now a direct argument
    survey_df_per_trial = data_loader.load_survey_data(eprime_txt_file_path, participant_id)
    processed_data_artifacts['analysis_outputs']['dataframes']['survey_data_per_trial'] = survey_df_per_trial
    
    events_df = None
    if eprime_txt_file_path:
        # Determine reference sampling frequency for EventProcessor (e.g., EEG or fNIRS)
        # This is used if EventProcessor needs to convert E-Prime times to samples.
        # However, it's better if EventProcessor returns times in seconds, and synchronization
        # happens based on absolute timestamps if possible.
        ref_sfreq_for_event_proc = eeg_sampling_rate if eeg_sampling_rate else fnirs_sampling_rate
        
        event_processor = EventProcessor(p_logger, default_sfreq=ref_sfreq_for_event_proc)
        events_df_eprime_relative = event_processor.process_event_log(eprime_txt_file_path) 
        
        if events_df_eprime_relative is not None and not events_df_eprime_relative.empty:
            # --- Synchronization Step ---
            # Here, you need to synchronize events_df_eprime_relative['onset_time_sec']
            # with the physiological data's timeline.
            # This example assumes a simple offset based on the first XDF marker if available.
            # A more robust method would use a dedicated sync pulse.
            
            xdf_marker_times = loaded_physiological_data.get('marker_times')
            xdf_marker_values = loaded_physiological_data.get('marker_values')
            
            # Placeholder for synchronization offset
            # TODO: Implement robust synchronization logic
            # For now, if no XDF markers, assume E-Prime times are already aligned (highly unlikely for real data)
            # or that the first XDF marker corresponds to the first E-Prime event.
            # This is a critical step that needs careful implementation based on your setup.
            
            time_offset_sec = 0.0 # Default to no offset
            
            # Example: Try to find a common "ExperimentStart" marker
            # This requires 'ExperimentStart' to be a marker in both E-Prime (parsed by EventProcessor)
            # and in XDF (parsed by DataLoader).
            eprime_start_marker_event = events_df_eprime_relative[events_df_eprime_relative['condition'] == "ExperimentStart"] # Adjust marker name
            
            if not eprime_start_marker_event.empty and xdf_marker_times is not None:
                eprime_sync_time = eprime_start_marker_event['onset_time_sec'].iloc[0]
                xdf_sync_time = None
                for i, val in enumerate(xdf_marker_values):
                    if val == "ExperimentStart": # Adjust marker name
                        xdf_sync_time = xdf_marker_times[i]
                        break
                if xdf_sync_time is not None:
                    # If EEG/fNIRS stream started after the XDF sync marker, adjust xdf_sync_time
                    # This assumes xdf_marker_times are absolute XDF clock.
                    # And physiological stream start times are also absolute XDF clock.
                    # Prioritize EEG stream start time if available, else fNIRS, else 0
                    main_physio_stream_start_xdf = 0
                    if raw_eeg_mne:
                        main_physio_stream_start_xdf = loaded_physiological_data.get('eeg_stream_start_time_xdf', 0)
                    elif fnirs_od_mne:
                        main_physio_stream_start_xdf = loaded_physiological_data.get('fnirs_stream_start_time_xdf', 0)
                    
                    if main_physio_stream_start_xdf == 0:
                        p_logger.warning("Could not determine main physiological stream start time from XDF for precise synchronization offset calculation.")
                    xdf_sync_time_relative_to_physio = xdf_sync_time - main_physio_stream_start_xdf
                    time_offset_sec = xdf_sync_time_relative_to_physio - eprime_sync_time
                    p_logger.info(f"Synchronization: E-Prime sync @ {eprime_sync_time:.3f}s, XDF sync relative to physio @ {xdf_sync_time_relative_to_physio:.3f}s. Offset: {time_offset_sec:.3f}s")
                else:
                    p_logger.warning("Synchronization: 'ExperimentStart' marker not found in XDF. Cannot calculate precise offset.")
            else:
                p_logger.warning("Synchronization: 'ExperimentStart' marker not found in E-Prime events or no XDF markers. Using zero offset (E-Prime times assumed relative to physio start).")

            events_df = events_df_eprime_relative.copy()
            events_df['onset_time_sec'] = events_df['onset_time_sec'] + time_offset_sec
            # Filter out events that might now be negative after offset if physio started late
            events_df = events_df[events_df['onset_time_sec'] >= 0]

            processed_data_artifacts['event_times_df'] = events_df
            event_csv_path = os.path.join(preproc_results_dir, f"{participant_id}_event_times_synchronized.csv")
            events_df.to_csv(event_csv_path, index=False)
            processed_data_artifacts['preprocessed_data_paths']['event_times_csv'] = event_csv_path
            p_logger.info(f"Synchronized event times saved to: {event_csv_path}")
            
            if processed_data_artifacts['baseline_start_time_sec'] is None and 'onset_time_sec' in events_df.columns:
                baseline_start_event = events_df[events_df['condition'] == config.BASELINE_MARKER_START_EPRIME]
                if not baseline_start_event.empty:
                    processed_data_artifacts['baseline_start_time_sec'] = baseline_start_event['onset_time_sec'].iloc[0]
                baseline_end_event = events_df[events_df['condition'] == config.BASELINE_MARKER_END_EPRIME]
                if not baseline_end_event.empty:
                    processed_data_artifacts['baseline_end_time_sec'] = baseline_end_event['onset_time_sec'].iloc[0]
                if processed_data_artifacts['baseline_start_time_sec'] is not None and \
                   processed_data_artifacts['baseline_end_time_sec'] is None:
                    first_movie_event = events_df[events_df['condition'].isin(EMOTIONAL_CONDITIONS)]
                    if not first_movie_event.empty:
                        processed_data_artifacts['baseline_end_time_sec'] = first_movie_event['onset_time_sec'].min()
        else:
            p_logger.warning("Event processing from E-Prime .txt did not return a valid DataFrame.")
    else:
        p_logger.warning("No E-Prime .txt event file found. Attempting to use XDF annotations for events.")

    if events_df is None and raw_eeg_mne and raw_eeg_mne.annotations and raw_eeg_mne.annotations.duration.size > 0:
        p_logger.info("Using MNE annotations from XDF for event timing (E-Prime events failed or not available).")
        temp_events_list = []
        # MNE annotations are relative to the start of the raw object.
        # If raw.first_samp is 0, then annotation onsets are directly usable as 'onset_time_sec'.
        for ann in raw_eeg_mne.annotations:
            temp_events_list.append({
                'onset_time_sec': ann['onset'], 
                'duration_sec': ann['duration'],
                'condition': ann['description'], 
            })
        events_df = pd.DataFrame(temp_events_list)
        if not events_df.empty:
             def map_marker_to_condition(marker_desc):
                 if "NEG" in marker_desc.upper(): return "Negative"
                 if "POS" in marker_desc.upper(): return "Positive"
                 if "NEU" in marker_desc.upper(): return "Neutral"
                 if config.BASELINE_MARKER_START in marker_desc: return config.BASELINE_MARKER_START_EPRIME 
                 if config.BASELINE_MARKER_END in marker_desc: return config.BASELINE_MARKER_END_EPRIME
                 return marker_desc 
            
             events_df['condition'] = events_df['condition'].apply(map_marker_to_condition)
             if 'trial_identifier_eprime' not in events_df.columns:
                 events_df['trial_identifier_eprime'] = [f"XDF_Marker_{i}" for i in range(len(events_df))]
             processed_data_artifacts['event_times_df'] = events_df
             p_logger.info(f"Created events_df from XDF annotations. {len(events_df)} events. Note: trial_identifier_eprime is a placeholder.")
        else:
            p_logger.warning("No MNE annotations found in XDF to create events_df.")

    current_events_df = processed_data_artifacts.get('event_times_df')
    if current_events_df is None or current_events_df.empty or \
       ('onset_time_sec' not in current_events_df.columns and 'onset_sample' not in current_events_df.columns):
        p_logger.error("Final events_df is missing or invalid. Cannot proceed with epoch-based analysis.")
        return processed_data_artifacts

    # --- Preprocessing Stage ---
    # ... (rest of preprocessing logic remains the same) ...
    p_logger.info("--- Preprocessing Stage ---")
    if raw_eeg_mne:
        eeg_preprocessor = EEGPreprocessor(p_logger)
        raw_eeg_processed = eeg_preprocessor.process(raw_eeg_mne.copy())
        processed_data_artifacts['mne_objects']['raw']['eeg_processed'] = raw_eeg_processed
        if raw_eeg_processed:
            eeg_processed_path = os.path.join(preproc_results_dir, f"{participant_id}_eeg_processed_raw.fif")
            raw_eeg_processed.save(eeg_processed_path, overwrite=True, verbose=False)
            processed_data_artifacts['preprocessed_data_paths']['eeg_processed_fif'] = eeg_processed_path
            p_logger.info(f"Processed EEG data saved to: {eeg_processed_path}")
    else:
        p_logger.warning("No EEG data loaded. Skipping EEG preprocessing.")

    if fnirs_od_mne:
        fnirs_preprocessor = FNIRSPreprocessor(p_logger)
        fnirs_haemo_processed = fnirs_preprocessor.process(fnirs_od_mne.copy())
        processed_data_artifacts['mne_objects']['raw']['fnirs_haemo_processed'] = fnirs_haemo_processed
        if fnirs_haemo_processed:
            fnirs_haemo_path = fnirs_preprocessor.save_preprocessed_data(fnirs_haemo_processed, participant_id, preproc_results_dir)
            if fnirs_haemo_path:
                processed_data_artifacts['preprocessed_data_paths']['fnirs_haemo_fif'] = fnirs_haemo_path
            p_logger.info("fNIRS data preprocessed to haemoglobin concentration.")
    else:
        p_logger.warning("No fNIRS data loaded. Skipping fNIRS preprocessing.")

    rpeaks_samples_overall = None 
    if ecg_data_raw is not None and ecg_sampling_rate is not None:
        ecg_preprocessor = ECGPreprocessor(p_logger)
        rpeak_times_path, nn_intervals_path, rpeaks_samples_overall, nn_intervals_ms_overall = ecg_preprocessor.preprocess_ecg(
            ecg_data_raw, ecg_sampling_rate, participant_id, preproc_results_dir
        )
        if rpeak_times_path: # Check if preprocessing was successful
            processed_data_artifacts['preprocessed_data_paths']['ecg_rpeak_times_csv'] = rpeak_times_path
            processed_data_artifacts['preprocessed_data_paths']['ecg_nn_intervals_csv'] = nn_intervals_path
            processed_data_artifacts['analysis_outputs']['metrics']['ecg_nn_intervals_ms_overall'] = nn_intervals_ms_overall
            processed_data_artifacts['ecg_rpeaks_samples_overall'] = rpeaks_samples_overall

            # Calculate resting-state RMSSD using AnalysisService
            baseline_start_sec_abs = processed_data_artifacts.get('baseline_start_time_sec')
            baseline_end_sec_abs = processed_data_artifacts.get('baseline_end_time_sec')
            
            if baseline_start_sec_abs is not None and baseline_end_sec_abs is not None:
                resting_rmssd = analysis_service.calculate_resting_state_rmssd(
                    ecg_data_raw, ecg_sampling_rate, ecg_times_abs,
                    baseline_start_sec_abs, baseline_end_sec_abs
                )
                processed_data_artifacts['analysis_outputs']['metrics']['baseline_rmssd'] = resting_rmssd
                if not np.isnan(resting_rmssd):
                     p_logger.info(f"Resting state RMSSD calculated: {resting_rmssd:.2f} ms")
            else:
                p_logger.warning("Baseline start/end times not available, skipping resting state RMSSD calculation.")

        if nn_intervals_ms_overall is not None:
             p_logger.info(f"ECG preprocessed (overall). Found {len(nn_intervals_ms_overall)} NN intervals.")
    else:
        p_logger.warning("No ECG data loaded or sampling rate missing. Skipping ECG preprocessing.")

    phasic_eda_full_signal = None 
    eda_original_sfreq_for_plv = None 
    if eda_data_raw is not None and eda_sampling_rate is not None:
        eda_preprocessor = EDAPreprocessor(p_logger)
        phasic_path, tonic_path, phasic_eda_full_signal, tonic_eda_full_signal = eda_preprocessor.preprocess_eda(
            eda_data_raw, eda_sampling_rate, participant_id, preproc_results_dir
        )
        if phasic_path: # Check if preprocessing was successful
            eda_original_sfreq_for_plv = eda_sampling_rate 
            processed_data_artifacts['preprocessed_data_paths']['phasic_eda_csv'] = phasic_path
            processed_data_artifacts['preprocessed_data_paths']['tonic_eda_csv'] = tonic_path
            # The phasic_eda_full_signal is now directly available
            p_logger.info("EDA preprocessed. Phasic and Tonic components available.")
    else:
        p_logger.warning("No EDA data loaded or sampling rate missing. Skipping EDA preprocessing.")


    # --- Analysis Stage ---
    ref_sfreq_for_events = eeg_sampling_rate if eeg_sampling_rate else (fnirs_sampling_rate if fnirs_sampling_rate else None)
    if 'onset_sample' not in current_events_df.columns and 'onset_time_sec' in current_events_df.columns and ref_sfreq_for_events:
        current_events_df['onset_sample'] = (current_events_df['onset_time_sec'] * ref_sfreq_for_events).astype(int)
    elif 'onset_sample' not in current_events_df.columns:
        p_logger.error("Cannot determine 'onset_sample' for events. Missing 'onset_time_sec' or reference sfreq.")
        return processed_data_artifacts

    event_id_map = {name: i+1 for i, name in enumerate(EMOTIONAL_CONDITIONS)} 
    if 'condition' not in current_events_df.columns:
        p_logger.error("Events DataFrame missing 'condition' column for epoching.")
        return processed_data_artifacts
        
    trial_id_eprime_map = {} 
    if 'trial_identifier_eprime' not in current_events_df.columns:
        p_logger.error("Events DataFrame missing 'trial_identifier_eprime' column. Accurate survey linking for WP2 will fail.")
        current_events_df['trial_identifier_eprime_numeric'] = 0 
    else:
        unique_trial_ids_eprime = current_events_df['trial_identifier_eprime'].unique()
        trial_id_eprime_map = {name: i + 1000 for i, name in enumerate(unique_trial_ids_eprime)} 
        current_events_df['trial_identifier_eprime_numeric'] = current_events_df['trial_identifier_eprime'].map(trial_id_eprime_map).fillna(0).astype(int)

    current_events_df['condition_id'] = current_events_df['condition'].map(event_id_map).fillna(0).astype(int) 
    
    mne_events_df = current_events_df[current_events_df['condition_id'] > 0][['onset_sample', 'condition_id', 'trial_identifier_eprime_numeric']].copy()
    if mne_events_df.empty:
        p_logger.error("No valid events found after mapping conditions for epoching.")
        return processed_data_artifacts

    mne_events_df.insert(1, 'prev_event_id', 0) 
    mne_events_array = mne_events_df.values.astype(int)
    
    processed_data_artifacts['analysis_outputs']['metadata']['trial_id_eprime_map'] = trial_id_eprime_map
    processed_data_artifacts['analysis_outputs']['metadata']['event_id_map_mne'] = event_id_map 

    raw_eeg_proc = processed_data_artifacts['mne_objects']['raw'].get('eeg_processed')
    if raw_eeg_proc:
        try:
            epochs_eeg = mne.Epochs(raw_eeg_proc, mne_events_array, event_id=event_id_map, 
                                    tmin=config.ANALYSIS_EPOCH_TIMES[0], tmax=config.ANALYSIS_EPOCH_TIMES[1],
                                    baseline=config.ANALYSIS_BASELINE_TIMES, preload=True, verbose=False,
                                    picks='eeg', on_missing='warning') 
            processed_data_artifacts['mne_objects']['epochs']['eeg'] = epochs_eeg
            p_logger.info(f"EEG data epoched: {len(epochs_eeg)} epochs created across {len(epochs_eeg.event_id)} conditions.")
        except Exception as e:
            p_logger.error(f"Failed to epoch EEG data: {e}", exc_info=True)

    raw_fnirs_proc = processed_data_artifacts['mne_objects']['raw'].get('fnirs_haemo_processed')
    if raw_fnirs_proc:
        try:
            epochs_fnirs = mne.Epochs(raw_fnirs_proc, mne_events_array, event_id=event_id_map,
                                      tmin=config.ANALYSIS_EPOCH_TIMES[0], tmax=config.ANALYSIS_EPOCH_TIMES[1],
                                      baseline=config.ANALYSIS_BASELINE_TIMES, preload=True, verbose=False,
                                      on_missing='warning')
            processed_data_artifacts['mne_objects']['epochs']['fnirs'] = epochs_fnirs
            p_logger.info(f"fNIRS data epoched: {len(epochs_fnirs)} epochs created across {len(epochs_fnirs.event_id)} conditions.")
        except Exception as e:
            p_logger.error(f"Failed to epoch fNIRS data: {e}", exc_info=True)

    # --- Work Package 1: Emotional Modulation of Synchrony ---
    # ... (WP1 logic remains largely the same, but uses the refactored fNIRS GLM call)
    p_logger.info("--- WP1: Emotional Modulation of Synchrony ---")
    fnirs_epochs = processed_data_artifacts['mne_objects']['epochs'].get('fnirs')
    eeg_epochs = processed_data_artifacts['mne_objects']['epochs'].get('eeg')
    
    active_fnirs_roi_names_for_eeg_guidance = []
    if fnirs_epochs and raw_eeg_proc: 
        # Call the refactored fNIRS GLM method via AnalysisService
        event_id_map_for_glm = processed_data_artifacts['analysis_outputs']['metadata'].get('event_id_map_mne', {})
        fnirs_glm_output = analysis_service.run_fnirs_glm_and_contrasts(
            fnirs_epochs, event_id_map_for_glm, participant_id, analysis_results_dir
        )
        processed_data_artifacts['analysis_outputs']['stats']['fnirs_glm_output'] = fnirs_glm_output
        if fnirs_glm_output and 'contrast_results' in fnirs_glm_output:
            for contrast_name_plot, contrast_df_plot in fnirs_glm_output['contrast_results'].items():
                if contrast_df_plot is not None and not contrast_df_plot.empty:
                    plotting_service.plot_fnirs_contrast_results(participant_id, contrast_name_plot, contrast_df_plot, f"participant_fnirs_contrast_{contrast_name_plot}")
        active_fnirs_roi_names_for_eeg_guidance = fnirs_glm_output.get('active_rois_for_eeg_guidance', [])
        
        eeg_channels_for_plv_wp1 = select_eeg_channels_by_fnirs_rois(
            raw_eeg_proc.info, active_fnirs_roi_names_for_eeg_guidance, p_logger
        )
        if not eeg_channels_for_plv_wp1: 
            eeg_channels_for_plv_wp1 = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in raw_eeg_proc.ch_names]
        p_logger.info(f"Using EEG channels for WP1 PLV: {eeg_channels_for_plv_wp1}")
    elif raw_eeg_proc: 
        eeg_channels_for_plv_wp1 = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in raw_eeg_proc.ch_names]
        p_logger.info(f"No fNIRS epochs for GLM. Using default EEG channels for WP1 PLV: {eeg_channels_for_plv_wp1}")
    else:
        eeg_channels_for_plv_wp1 = []
        p_logger.warning("No EEG data available for PLV channel selection.")

    all_trial_plv_results_list = [] 
    if eeg_epochs and eeg_channels_for_plv_wp1 and \
       (rpeaks_samples_overall is not None or phasic_eda_full_signal is not None):
        p_logger.info("Calculating trial-wise PLV for WP1...")
        
        continuous_hrv_signal_for_plv = None
        hrv_sfreq_for_plv = None
        if rpeaks_samples_overall is not None and ecg_sampling_rate is not None:
            continuous_hrv_signal_for_plv, _ = analysis_service.get_continuous_hrv_signal(
                rpeaks_samples_overall, ecg_sampling_rate, config.PLV_RESAMPLE_SFREQ_AUTONOMIC
            )
            hrv_sfreq_for_plv = config.PLV_RESAMPLE_SFREQ_AUTONOMIC
            processed_data_artifacts['analysis_outputs']['metadata']['hrv_sfreq_for_plv'] = hrv_sfreq_for_plv

        trial_plv_df_participant = analysis_service.calculate_trial_plv(
            eeg_epochs, eeg_channels_for_plv_wp1,
            continuous_hrv_signal_for_plv, hrv_sfreq_for_plv,
            phasic_eda_full_signal, eda_original_sfreq_for_plv, 
            participant_id, 
            raw_eeg_proc.info['sfreq'], 
            processed_data_artifacts['analysis_outputs']['metadata'].get('trial_id_eprime_map', {})
        )
        if trial_plv_df_participant is not None and not trial_plv_df_participant.empty:
            all_trial_plv_results_list.append(trial_plv_df_participant)
        
        if all_trial_plv_results_list:
            trial_plv_df = pd.concat(all_trial_plv_results_list, ignore_index=True)
            processed_data_artifacts['analysis_outputs']['dataframes']['trial_plv_wp1'] = trial_plv_df
            p_logger.info(f"Calculated {len(trial_plv_df)} trial-wise PLV values for WP1 (across all bands/modalities).")
            avg_plv_wp1_df = trial_plv_df.groupby(['participant_id', 'condition', 'modality_pair', 'eeg_band'])['plv'].mean().reset_index()
            processed_data_artifacts['analysis_outputs']['dataframes']['avg_plv_wp1'] = avg_plv_wp1_df
            
            plotting_service.plot_plv_results(participant_id, avg_plv_wp1_df, "wp1_avg_plv")
            p_logger.info("ANOVA for WP1 PLV will be performed at the group level.")

    # --- Work Package 2, 3, 4 logic remains largely the same as before ---
    # ... (WP2, WP3, WP4 logic) ...
    p_logger.info("--- WP2: Synchrony and Subjective Arousal ---")
    trial_plv_df_wp1_for_wp2 = processed_data_artifacts['analysis_outputs']['dataframes'].get('trial_plv_wp1')

    if survey_df_per_trial is not None and not survey_df_per_trial.empty and \
       trial_plv_df_wp1_for_wp2 is not None and not trial_plv_df_wp1_for_wp2.empty:
        if 'sam_arousal' in survey_df_per_trial.columns and 'trial_identifier_eprime' in survey_df_per_trial.columns:
            processed_data_artifacts['analysis_outputs']['metadata']['wp2_has_sam_arousal_and_ids'] = True 
            p_logger.info(f"WP2: Participant {participant_id} has SAM arousal and trial identifiers for survey data.")
        else:
            p_logger.warning(f"WP2: Participant {participant_id} survey data missing 'sam_arousal' or 'trial_identifier_eprime'.")
    else:
        p_logger.info("Skipping WP2 prep due to missing survey or PLV data for this participant.")

    p_logger.info("--- WP3: Baseline Vagal Tone and Task-Related Synchrony ---")
    # Baseline RMSSD is now directly calculated and stored if successful
    baseline_rmssd = processed_data_artifacts['analysis_outputs']['metrics'].get('baseline_rmssd')
    if baseline_rmssd is not None and not np.isnan(baseline_rmssd):
        if baseline_rmssd is not None and not np.isnan(baseline_rmssd):
            p_logger.info(f"Calculated baseline RMSSD: {baseline_rmssd:.2f} ms")
            avg_plv_df_wp1_for_wp3 = processed_data_artifacts['analysis_outputs']['dataframes'].get('avg_plv_wp1')
            if avg_plv_df_wp1_for_wp3 is not None and not avg_plv_df_wp1_for_wp3.empty:
                plv_negative_series = avg_plv_df_wp1_for_wp3[
                    (avg_plv_df_wp1_for_wp3['condition'] == 'Negative') &
                    (avg_plv_df_wp1_for_wp3['modality_pair'] == 'EEG-HRV') & 
                    (avg_plv_df_wp1_for_wp3['eeg_band'] == config.PLV_PRIMARY_EEG_BAND_FOR_WP3)         
                ]['plv']
                if not plv_negative_series.empty:
                    avg_plv_negative = plv_negative_series.mean() 
                    processed_data_artifacts['analysis_outputs']['metrics']['wp3_avg_plv_negative_specific'] = avg_plv_negative
                    p_logger.info(f"WP3: P{participant_id} - Baseline RMSSD: {baseline_rmssd:.2f}, Avg PLV (Negative, {config.PLV_PRIMARY_EEG_BAND_FOR_WP3}, HRV): {avg_plv_negative:.3f}")
    else:
        p_logger.warning(f"WP3: Baseline RMSSD not available or NaN for P{participant_id}.")

    p_logger.info("--- WP4: Frontal Asymmetry and Branch-Specific Synchrony ---")
    if eeg_epochs: 
        psd_results, fai_results_per_condition = analysis_service.calculate_psd_and_fai(
            raw_eeg_proc, 
            mne_events_array, 
            event_id_map 
        )
        processed_data_artifacts['analysis_outputs']['metrics']['fai_results_per_condition_wp4'] = fai_results_per_condition
        
        avg_fai_f4f3_list = []
        target_fai_pair_wp4 = f"{config.FAI_ELECTRODE_PAIRS_FOR_WP4[1]}_vs_{config.FAI_ELECTRODE_PAIRS_FOR_WP4[0]}" 
        
        if fai_results_per_condition:
            for cond_name, fai_pairs in fai_results_per_condition.items():
                if cond_name in EMOTIONAL_CONDITIONS: 
                    if target_fai_pair_wp4 in fai_pairs:
                        avg_fai_f4f3_list.append(fai_pairs[target_fai_pair_wp4])
            
            if avg_fai_f4f3_list:
                processed_data_artifacts['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional'] = np.nanmean(avg_fai_f4f3_list)
                p_logger.info(f"WP4: P{participant_id} - Avg FAI ({target_fai_pair_wp4}) over emotional conditions: {processed_data_artifacts['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional']:.3f}")
    else:
        p_logger.warning(f"WP4: No EEG epochs for P{participant_id}, cannot calculate FAI.")

    p_logger.info("--- Analysis Stage Complete for Participant ---")
    return processed_data_artifacts


def main_orchestrator(data_root_dir, output_base_dir, participant_ids=None):
    """
    Main orchestrator for processing multiple participants.
    """ 
    main_log_file = os.path.join(output_base_dir, "orchestrator_log.txt")
    os.makedirs(output_base_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, mode='w'), 
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger("MainOrchestrator")
    main_logger.info("--- EmotiView Data Processor Starting ---")
    
    if participant_ids is None: 
        try:
            participant_ids = [d for d in os.listdir(data_root_dir) 
                               if os.path.isdir(os.path.join(data_root_dir, d)) and 
                               (d.startswith("P") or d.startswith("EV_P"))] 
        except FileNotFoundError:
            main_logger.error(f"Data root directory not found: {data_root_dir}")
            return []
            
        if not participant_ids:
            main_logger.error(f"No participant subdirectories matching pattern found in {data_root_dir}.")
            return []
        main_logger.info(f"Found {len(participant_ids)} potential participants: {participant_ids}")
    
    all_participants_summary_artifacts = [] 

    for p_id_raw in participant_ids:
        p_id = p_id_raw 

        main_logger.info(f"--- Processing participant: {p_id} ---")
        participant_output_dirs = create_output_directories(output_base_dir, p_id)
        
        p_log_manager = ParticipantLogger(participant_output_dirs['base_participant'], p_id, config.LOG_LEVEL)
        participant_logger_instance = p_log_manager.get_logger()
        
        try:
            participant_raw_data_path = os.path.join(data_root_dir, p_id_raw) 
            
            eprime_txt_file = None
            if os.path.exists(participant_raw_data_path):
                for f_name in os.listdir(participant_raw_data_path): 
                    if p_id in f_name and f_name.lower().endswith('.txt'):
                        eprime_txt_file = os.path.join(participant_raw_data_path, f_name)
                        break
            
            if not os.path.exists(participant_raw_data_path) and not eprime_txt_file:
                participant_logger_instance.warning(f"No data directory or E-Prime .txt file found for participant {p_id} in {participant_raw_data_path}. Skipping.")
                all_participants_summary_artifacts.append({
                    'participant_id': p_id, 'status': 'no_data_found',
                    'log_file': p_log_manager.log_file_path
                })
                continue

            processed_artifacts = process_participant_data(
                p_id, 
                participant_raw_data_path, # This is participant_raw_xdf_data_dir
                eprime_txt_file,           # This is eprime_txt_file_path
                participant_output_dirs, 
                output_base_dir, participant_logger_instance
            )
            all_participants_summary_artifacts.append(processed_artifacts)
            participant_logger_instance.info(f"--- Successfully processed participant: {p_id} ---")
        except Exception as e:
            participant_logger_instance.error(f"--- CRITICAL ERROR processing participant {p_id}: {e} ---", exc_info=True)
            all_participants_summary_artifacts.append({
                'participant_id': p_id, 'status': 'error', 'error_message': str(e),
                'log_file': p_log_manager.log_file_path
            })
        finally:
            p_log_manager.close_handlers() 

    main_logger.info("--- EmotiView Data Processor Finished Individual Participant Processing ---")
    
    # --- Group-Level Analyses (remains the same as your latest version) ---
    # ... (WP1, WP2, WP3, WP4 group analysis logic with FDR and plotting calls) ...
    group_analysis_service = AnalysisService(main_logger) 
    group_plotting_service = PlottingService(main_logger, os.path.join(output_base_dir, "_GROUP_PLOTS"))
    group_results_dir = os.path.join(output_base_dir, "_GROUP_RESULTS")
    os.makedirs(group_results_dir, exist_ok=True)

    # WP1: ANOVA on PLV
    all_avg_plv_wp1_dfs = [
        res['analysis_outputs']['dataframes']['avg_plv_wp1'] 
        for res in all_participants_summary_artifacts 
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('avg_plv_wp1') is not None
    ]
    if all_avg_plv_wp1_dfs:
        group_plv_data_wp1 = pd.concat(all_avg_plv_wp1_dfs, ignore_index=True)
        if not group_plv_data_wp1.empty:
            main_logger.info(f"WP1: Aggregated PLV data from {group_plv_data_wp1['participant_id'].nunique()} participants for group ANOVA.")
            df_for_anova_wp1 = group_plv_data_wp1[ 
                (group_plv_data_wp1['eeg_band'] == config.PLV_PRIMARY_EEG_BAND_FOR_WP1) & 
                (group_plv_data_wp1['modality_pair'] == 'EEG-HRV')
            ].copy()
            
            if not df_for_anova_wp1.empty and df_for_anova_wp1['participant_id'].nunique() > 1:
                anova_results_wp1 = group_analysis_service.run_repeated_measures_anova(
                   data_df=df_for_anova_wp1, dv='plv', 
                   within='condition', subject='participant_id'
                )
                main_logger.info(f"Group ANOVA results for PLV (WP1, {config.PLV_PRIMARY_EEG_BAND_FOR_WP1}, EEG-HRV):\n{anova_results_wp1}")
                if anova_results_wp1 is not None and not anova_results_wp1.empty:
                    p_values_to_correct_wp1 = anova_results_wp1['p-unc'].dropna().tolist()
                    if p_values_to_correct_wp1:
                        reject_fdr_wp1, pval_corr_fdr_wp1 = apply_fdr_correction(p_values_to_correct_wp1, alpha=0.05)
                        fdr_series_pval = pd.Series(pval_corr_fdr_wp1, index=anova_results_wp1.dropna(subset=['p-unc']).index)
                        fdr_series_reject = pd.Series(reject_fdr_wp1, index=anova_results_wp1.dropna(subset=['p-unc']).index)
                        anova_results_wp1['p-corr-fdr'] = fdr_series_pval
                        anova_results_wp1['reject_fdr'] = fdr_series_reject
                        main_logger.info(f"WP1 ANOVA with FDR correction:\n{anova_results_wp1}")
                    
                    condition_effect_row = anova_results_wp1[anova_results_wp1['Source'] == 'condition'] 
                    if not condition_effect_row.empty and 'p-corr-fdr' in condition_effect_row:
                        main_logger.info(
                            f"WP1 ANOVA 'condition' effect: p-unc={condition_effect_row['p-unc'].iloc[0]:.4f}, "
                            f"p-fdr={condition_effect_row['p-corr-fdr'].iloc[0]:.4f}, "
                            f"reject_fdr={condition_effect_row['reject_fdr'].iloc[0]}"
                        )
                    
                    anova_results_wp1.to_csv(os.path.join(group_results_dir, f"group_anova_wp1_plv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP1}_hrv.csv"))
                    group_plotting_service.plot_anova_results("GROUP", anova_results_wp1, df_for_anova_wp1, 'plv', 'condition', f"WP1: PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP1}, HRV) by Condition", f"wp1_plv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP1}_hrv") 
            else:
                main_logger.warning(f"WP1: Not enough data or participants for ANOVA on PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP1}, EEG-HRV).")

    all_trial_plv_wp1_dfs_for_wp2 = [ 
        res['analysis_outputs']['dataframes']['trial_plv_wp1']
        for res in all_participants_summary_artifacts
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('trial_plv_wp1') is not None
    ]
    all_survey_dfs_for_wp2 = [ 
        res['analysis_outputs']['dataframes']['survey_data_per_trial'] 
        for res in all_participants_summary_artifacts
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('survey_data_per_trial') is not None
    ]

    if all_trial_plv_wp1_dfs_for_wp2 and all_survey_dfs_for_wp2:
        group_trial_plv_df_wp2 = pd.concat(all_trial_plv_wp1_dfs_for_wp2, ignore_index=True)
        group_survey_df_wp2 = pd.concat(all_survey_dfs_for_wp2, ignore_index=True)

        if not group_trial_plv_df_wp2.empty and not group_survey_df_wp2.empty and \
           'sam_arousal' in group_survey_df_wp2.columns and \
           'trial_identifier_eprime' in group_trial_plv_df_wp2.columns and \
           'trial_identifier_eprime' in group_survey_df_wp2.columns:
            try:
                merged_wp2_group_df = pd.merge(group_trial_plv_df_wp2, group_survey_df_wp2, 
                                               on=['participant_id', 'condition', 'trial_identifier_eprime'], how='inner')
                if not merged_wp2_group_df.empty:
                    df_for_corr_wp2 = merged_wp2_group_df[
                        (merged_wp2_group_df['eeg_band'] == config.PLV_PRIMARY_EEG_BAND_FOR_WP2) & 
                        (merged_wp2_group_df['modality_pair'] == 'EEG-HRV') 
                    ].copy()
                    
                    if not df_for_corr_wp2.empty and df_for_corr_wp2['participant_id'].nunique() > 1 and len(df_for_corr_wp2) > 2:
                        corr_wp2 = group_analysis_service.run_correlation_analysis(
                            df_for_corr_wp2['sam_arousal'], df_for_corr_wp2['plv'], 
                            name1='SAM_Arousal', name2=f'Trial_PLV_{config.PLV_PRIMARY_EEG_BAND_FOR_WP2}_HRV'
                        )
                        main_logger.info(f"Group Correlation (WP2 - Trial SAM Arousal vs Trial PLV {config.PLV_PRIMARY_EEG_BAND_FOR_WP2}):\n{corr_wp2}")
                        if corr_wp2 is not None and not corr_wp2.empty: 
                            corr_wp2.to_csv(os.path.join(group_results_dir, f"group_corr_wp2_trial_arousal_vs_plv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP2}.csv"))
                            group_plotting_service.plot_correlation("GROUP", df_for_corr_wp2['sam_arousal'], df_for_corr_wp2['plv'], 
                                                                    "SAM Arousal (Trial)", f"PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP2}-HRV, Trial)", 
                                                                    f"WP2: Trial Arousal vs PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP2})", f"wp2_trial_arousal_plv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP2}", corr_wp2)
                    else:
                        main_logger.warning(f"WP2: Not enough data for correlation after filtering for band {config.PLV_PRIMARY_EEG_BAND_FOR_WP2} and modality.")
                else:
                    main_logger.warning("WP2: Merged DataFrame for PLV and survey data is empty. Check merge keys and data content.")
            except Exception as e_merge_wp2:
                 main_logger.error(f"WP2: Error merging PLV and survey data for group analysis: {e_merge_wp2}", exc_info=True)
        else:
            main_logger.warning("WP2: Missing necessary columns ('sam_arousal', 'trial_identifier_eprime') in PLV or survey data for group merge.")

    wp3_data_list = [{'participant_id': r['participant_id'],
                      'baseline_rmssd': r['analysis_outputs']['metrics'].get('baseline_rmssd'),
                      'plv_negative_specific': r['analysis_outputs']['metrics'].get('wp3_avg_plv_negative_specific')}
                     for r in all_participants_summary_artifacts if isinstance(r, dict) and 
                     r.get('analysis_outputs', {}).get('metrics', {}).get('baseline_rmssd') is not None and
                     not np.isnan(r['analysis_outputs']['metrics'].get('baseline_rmssd'))] 
    
    if wp3_data_list:
        wp3_group_df = pd.DataFrame(wp3_data_list).dropna() 
        if not wp3_group_df.empty and len(wp3_group_df) >=3:
            corr_wp3 = group_analysis_service.run_correlation_analysis(wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], name1='Baseline_RMSSD', name2=f'Avg_PLV_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}_HRV_Negative')
            main_logger.info(f"Group Correlation (WP3 - Baseline RMSSD vs Negative PLV {config.PLV_PRIMARY_EEG_BAND_FOR_WP3}):\n{corr_wp3}")
            if corr_wp3 is not None and not corr_wp3.empty:
                 corr_wp3.to_csv(os.path.join(group_results_dir, f"group_corr_wp3_rmssd_vs_plv_neg_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}.csv")) 
                 group_plotting_service.plot_correlation("GROUP", wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_specific'], "Baseline RMSSD (ms)", f"Avg PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP3}-HRV-Negative)", f"WP3: RMSSD vs Negative PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP3})", f"wp3_rmssd_plv_neg_{config.PLV_PRIMARY_EEG_BAND_FOR_WP3}", corr_wp3)
        else:
            main_logger.warning("WP3: Not enough valid data for RMSSD vs PLV correlation after NaN handling.")

    wp4_fai_list = []
    for res in all_participants_summary_artifacts:
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('metrics', {}).get('wp4_avg_fai_f4f3_emotional') is not None:
            avg_fai_val = res['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional']
            if not np.isnan(avg_fai_val): 
                 wp4_fai_list.append({'participant_id': res['participant_id'], 
                                      'avg_fai_f4f3_emotional': avg_fai_val})
    
    if wp4_fai_list and all_avg_plv_wp1_dfs: 
        wp4_fai_group_df = pd.DataFrame(wp4_fai_list)
        group_plv_data_wp1_for_wp4 = pd.concat(all_avg_plv_wp1_dfs, ignore_index=True) 
        
        plv_hrv_alpha_avg_emotional_wp4 = group_plv_data_wp1_for_wp4[
            (group_plv_data_wp1_for_wp4['eeg_band'] == config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV) & 
            (group_plv_data_wp1_for_wp4['modality_pair'] == 'EEG-HRV') &
            (group_plv_data_wp1_for_wp4['condition'].isin(EMOTIONAL_CONDITIONS)) 
        ].groupby('participant_id')['plv'].mean().reset_index().rename(columns={'plv': 'avg_plv_hrv_alpha_emotional'})

        merged_wp4_hrv = pd.merge(wp4_fai_group_df, plv_hrv_alpha_avg_emotional_wp4, on='participant_id')
        corr_wp4_hrv_result = None
        if not merged_wp4_hrv.empty and len(merged_wp4_hrv) >= 3:
            corr_wp4_hrv_result = group_analysis_service.run_correlation_analysis(merged_wp4_hrv['avg_fai_f4f3_emotional'], merged_wp4_hrv['avg_plv_hrv_alpha_emotional'], name1='Avg_FAI_F4F3_Emotional', name2=f'Avg_PLV_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV}_HRV_Emotional')
            main_logger.info(f"Group Correlation (WP4 - Avg FAI F4-F3 vs EEG-HRV PLV {config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV}):\n{corr_wp4_hrv_result}")
        
        plv_eda_alpha_avg_emotional_wp4 = group_plv_data_wp1_for_wp4[ 
            (group_plv_data_wp1_for_wp4['eeg_band'] == config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA) & 
            (group_plv_data_wp1_for_wp4['modality_pair'] == 'EEG-EDA') &
            (group_plv_data_wp1_for_wp4['condition'].isin(EMOTIONAL_CONDITIONS))
        ].groupby('participant_id')['plv'].mean().reset_index().rename(columns={'plv': 'avg_plv_eda_alpha_emotional'})
        
        merged_wp4_eda = pd.merge(wp4_fai_group_df, plv_eda_alpha_avg_emotional_wp4, on='participant_id')
        corr_wp4_eda_result = None
        if not merged_wp4_eda.empty and len(merged_wp4_eda) >= 3:
            corr_wp4_eda_result = group_analysis_service.run_correlation_analysis(merged_wp4_eda['avg_fai_f4f3_emotional'], merged_wp4_eda['avg_plv_eda_alpha_emotional'], name1='Avg_FAI_F4F3_Emotional', name2=f'Avg_PLV_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA}_EDA_Emotional')
            main_logger.info(f"Group Correlation (WP4 - Avg FAI F4-F3 vs EEG-EDA PLV {config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA}):\n{corr_wp4_eda_result}")

        wp4_corr_p_values_fdr = []
        wp4_corr_dfs_for_fdr = []
        if corr_wp4_hrv_result is not None and not corr_wp4_hrv_result.empty:
            wp4_corr_p_values_fdr.append(corr_wp4_hrv_result['p-val'].iloc[0])
            wp4_corr_dfs_for_fdr.append(corr_wp4_hrv_result)
        if corr_wp4_eda_result is not None and not corr_wp4_eda_result.empty:
            wp4_corr_p_values_fdr.append(corr_wp4_eda_result['p-val'].iloc[0])
            wp4_corr_dfs_for_fdr.append(corr_wp4_eda_result)

        if wp4_corr_p_values_fdr:
            reject_fdr_wp4, pval_corr_fdr_wp4 = apply_fdr_correction(wp4_corr_p_values_fdr, alpha=0.05)
            for i, corr_df in enumerate(wp4_corr_dfs_for_fdr):
                corr_df['p-corr-fdr'] = pval_corr_fdr_wp4[i]
                corr_df['reject_fdr'] = reject_fdr_wp4[i]
                main_logger.info(f"WP4 Correlation ({corr_df.index[0]}) with FDR:\n{corr_df}")
                
                if corr_df is corr_wp4_hrv_result: 
                    corr_df.to_csv(os.path.join(group_results_dir, f"group_corr_wp4_fai_vs_plv_hrv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV}.csv"))
                    group_plotting_service.plot_correlation("GROUP", merged_wp4_hrv['avg_fai_f4f3_emotional'], merged_wp4_hrv['avg_plv_hrv_alpha_emotional'], "Avg FAI (F4-F3) Emotional", f"Avg PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV}-HRV) Emotional", f"WP4: FAI vs EEG-HRV PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV})", f"wp4_fai_plv_hrv_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_HRV}", corr_df)
                elif corr_df is corr_wp4_eda_result: 
                    corr_df.to_csv(os.path.join(group_results_dir, f"group_corr_wp4_fai_vs_plv_eda_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA}.csv"))
                    group_plotting_service.plot_correlation("GROUP", merged_wp4_eda['avg_fai_f4f3_emotional'], merged_wp4_eda['avg_plv_eda_alpha_emotional'], "Avg FAI (F4-F3) Emotional", f"Avg PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA}-EDA) Emotional", f"WP4: FAI vs EEG-EDA PLV ({config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA})", f"wp4_fai_plv_eda_{config.PLV_PRIMARY_EEG_BAND_FOR_WP4_EDA}", corr_df)
    else:
        main_logger.warning("WP4: Not enough FAI data or PLV data for group correlations.")

    summary_list = []
    for r_idx, r_val in enumerate(all_participants_summary_artifacts):
        if isinstance(r_val, dict):
            summary_list.append({
                'participant_id': r_val.get('participant_id', f'unknown_participant_{r_idx}'),
                'status': r_val.get('status', 'error' if 'error_message' in r_val else 'processed'),
                'log_file': r_val.get('log_file')
            })
        else: 
            summary_list.append({'participant_id': f'unknown_participant_{r_idx}', 'status': 'unknown_format', 'log_file': None})
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(output_base_dir, "processing_summary.csv"), index=False)
    main_logger.info(f"Processing summary saved to {os.path.join(output_base_dir, 'processing_summary.csv')}")

    return all_participants_summary_artifacts

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) 

    default_data_root = os.path.join(project_root, "sample_data") 
    default_output_root = os.path.join(project_root, "EV_Processed_Data_Orchestrator_Pilot")

    print(f"Default Data Root: {default_data_root}")
    print(f"Default Output Root: {default_output_root}")

    main_orchestrator(default_data_root, default_output_root, participant_ids=None)