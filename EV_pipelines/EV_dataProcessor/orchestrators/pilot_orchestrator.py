# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\pilot_orchestrator.py
import os
import mne
import pandas as pd
import numpy as np
import logging # For main logger
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from ..analysis.group_analyzer import GroupAnalyzer
from ..utils.parallel_runner import ParallelTaskRunner # New import
from ..utils.helpers import select_eeg_channels_by_fnirs_rois, create_output_directories, create_mne_events_from_dataframe # New helper

# Constants for data types
DATA_TYPES = ['eeg', 'fnirs', 'ecg', 'eda', 'events', 'survey'] 
EMOTIONAL_CONDITIONS = ['Positive', 'Negative', 'Neutral'] 

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
    
    # Retrieve events_df from DataLoader.
    # DataLoader is responsible for providing an events_df that is appropriately timed.
    # If LabRecorder synchronizes everything, XDF markers (used as fallback by DataLoader)
    # will have absolute, correct timings.
    # If DataLoader parsed an E-Prime .txt, its times are relative to E-Prime start;
    # we assume this E-Prime start is aligned with the XDF recording start.
    events_df = loaded_physiological_data.get('events_df')

    if events_df is not None and not events_df.empty:
        p_logger.info("Using events_df provided by DataLoader.")
        # Ensure 'onset_time_sec' exists, as it's crucial.
        if 'onset_time_sec' not in events_df.columns:
            p_logger.error("Critical: 'onset_time_sec' column missing in events_df from DataLoader.")
            processed_data_artifacts['status'] = 'error_event_timing_missing'
            return processed_data_artifacts
            
        # Save the (assumed synchronized) event times
        processed_data_artifacts['event_times_df'] = events_df
        event_csv_path = os.path.join(preproc_results_dir, f"{participant_id}_event_times_synchronized.csv")
        events_df.to_csv(event_csv_path, index=False)
        processed_data_artifacts['preprocessed_data_paths']['event_times_csv'] = event_csv_path
        p_logger.info(f"Event times (from DataLoader) saved to: {event_csv_path}")

        # If baseline times were not found from XDF by DataLoader, try to derive from events_df
        # (which might be from E-Prime via DataLoader or XDF markers via DataLoader)
        if processed_data_artifacts['baseline_start_time_sec'] is None and 'onset_time_sec' in events_df.columns:
            baseline_start_event = events_df[events_df['condition'] == config.BASELINE_MARKER_START_EPRIME] # Assuming 'condition' column exists
            if not baseline_start_event.empty:
                processed_data_artifacts['baseline_start_time_sec'] = baseline_start_event['onset_time_sec'].iloc[0]
            baseline_end_event = events_df[events_df['condition'] == config.BASELINE_MARKER_END_EPRIME]
            if not baseline_end_event.empty:
                processed_data_artifacts['baseline_end_time_sec'] = baseline_end_event['onset_time_sec'].iloc[0]
            
            # Fallback for baseline end if only start is marked and end is still None
            if processed_data_artifacts['baseline_start_time_sec'] is not None and \
               processed_data_artifacts['baseline_end_time_sec'] is None:
                # Find the earliest emotional stimulus onset after the baseline start
                emotional_stim_after_baseline = events_df[
                    (events_df['condition'].isin(EMOTIONAL_CONDITIONS)) & 
                    (events_df['onset_time_sec'] > processed_data_artifacts['baseline_start_time_sec'])
                ]
                if not emotional_stim_after_baseline.empty:
                    processed_data_artifacts['baseline_end_time_sec'] = emotional_stim_after_baseline['onset_time_sec'].min()
    else:
        p_logger.error("No events_df provided by DataLoader. Cannot proceed with epoch-based analysis.")
        processed_data_artifacts['status'] = 'error_no_events'
        return processed_data_artifacts


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
        # This conversion will now happen inside create_mne_events_from_dataframe if needed
        pass
    elif 'onset_sample' not in current_events_df.columns:
        p_logger.error("Cannot determine 'onset_sample' for events. Missing 'onset_time_sec' or reference sfreq.")
        return processed_data_artifacts

    mne_events_array, event_id_map, trial_id_eprime_map = create_mne_events_from_dataframe(
        current_events_df, EMOTIONAL_CONDITIONS, ref_sfreq_for_events, p_logger
    )

    if mne_events_array is None or mne_events_array.size == 0:
        p_logger.error("No valid events found after mapping conditions for epoching.")
        return processed_data_artifacts
    
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
            fnirs_epochs, participant_id, analysis_results_dir # event_id_map_for_glm removed as it's in fnirs_epochs
        )
        processed_data_artifacts['analysis_outputs']['stats']['fnirs_glm_output'] = fnirs_glm_output
        if fnirs_glm_output and 'contrast_results' in fnirs_glm_output:
            for contrast_name_plot, contrast_df_plot in fnirs_glm_output['contrast_results'].items():
                if contrast_df_plot is not None and not contrast_df_plot.empty:
                    plotting_service.plot_fnirs_contrast_results(participant_id, contrast_name_plot, contrast_df_plot, f"participant_fnirs_contrast_{contrast_name_plot}")
        active_fnirs_roi_names_for_eeg_guidance = fnirs_glm_output.get('active_rois_for_eeg_guidance', [])
        
        eeg_channels_for_plv_wp1 = select_eeg_channels_by_fnirs_rois(
            raw_eeg_proc.info, active_fnirs_roi_names_for_eeg_guidance, 
            config.FNIRS_ROI_TO_EEG_CHANNELS_MAP, # Added missing argument
            p_logger
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

def _process_single_participant_task(task_config):
    """
    Worker function to process a single participant.
    This function will be executed by each thread.
    It includes logger setup and teardown for the participant.
    Args:
        task_config (dict): A dictionary containing all necessary parameters for this task.
                            Expected keys: 'p_id_raw', 'p_id', 'data_root_dir',
                                           'output_base_dir', 'main_logger_name'.
    """
    p_id_raw = task_config['p_id_raw']
    p_id = task_config['p_id']
    data_root_dir = task_config['data_root_dir']
    output_base_dir = task_config['output_base_dir']
    main_logger_name = task_config['main_logger_name']

    main_logger = logging.getLogger(main_logger_name) # Get main logger instance for thread
    main_logger.info(f"Thread starting for participant: {p_id}")

    participant_output_dirs = create_output_directories(output_base_dir, p_id)
    p_log_manager = ParticipantLogger(participant_output_dirs['base_participant'], p_id, config.LOG_LEVEL)
    participant_logger_instance = p_log_manager.get_logger()
    
    processed_artifacts = None
    try:
        participant_raw_data_path = os.path.join(data_root_dir, p_id_raw) 
        
        eprime_txt_file = None
        if os.path.exists(participant_raw_data_path):
            for f_name in os.listdir(participant_raw_data_path): 
                if p_id in f_name and f_name.lower().endswith('.txt'): # Simplified check
                    eprime_txt_file = os.path.join(participant_raw_data_path, f_name)
                    break
        
        if not os.path.exists(participant_raw_data_path) and not eprime_txt_file:
            participant_logger_instance.warning(f"No data directory or E-Prime .txt file found for participant {p_id} in {participant_raw_data_path}. Skipping.")
            return {'participant_id': p_id, 'status': 'no_data_found', 'log_file': p_log_manager.log_file_path}

        processed_artifacts = process_participant_data(
            p_id, 
            participant_raw_data_path,
            eprime_txt_file,
            participant_output_dirs, 
            output_base_dir, participant_logger_instance
        )
        participant_logger_instance.info(f"--- Successfully processed participant: {p_id} (in thread) ---")
    except Exception as e:
        participant_logger_instance.error(f"--- CRITICAL ERROR processing participant {p_id} (in thread): {e} ---", exc_info=True)
        processed_artifacts = {'participant_id': p_id, 'status': 'error', 'error_message': str(e), 'log_file': p_log_manager.log_file_path}
    finally:
        p_log_manager.close_handlers()
    return processed_artifacts

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
    
    # Determine the number of worker threads
    # For a 64-core CPU, you might start with 64 or slightly fewer.
    # os.cpu_count() can give the total number of logical cores.
    # Let's make it configurable or default to a reasonable number.
    max_workers = getattr(config, 'MAX_PARALLEL_PARTICIPANTS', os.cpu_count() or 4)
    main_logger.info(f"Using up to {max_workers} threads for parallel participant processing.")
    
    # Prepare task configurations for the parallel runner
    participant_task_configs = []
    for p_id_raw in participant_ids:
        # Assuming p_id_raw is the folder name and also the identifier used internally
        p_id = p_id_raw 
        participant_task_configs.append({
            'p_id_raw': p_id_raw,
            'p_id': p_id,
            'data_root_dir': data_root_dir,
            'output_base_dir': output_base_dir,
            'main_logger_name': main_logger.name
        })

    runner = ParallelTaskRunner(
        task_function=_process_single_participant_task,
        task_configs=participant_task_configs,
        max_workers=max_workers,
        main_logger_name=main_logger.name
    )
    all_participants_summary_artifacts = runner.run() # runner.run() now returns the list of results/errors

    main_logger.info("--- EmotiView Data Processor Finished Individual Participant Processing ---")

    # --- Group-Level Analyses (Delegated) ---
    if all_participants_summary_artifacts:
        # Filter out participants that did not process successfully before group analysis
        successful_artifacts = [
            p_artifact for p_artifact in all_participants_summary_artifacts 
            if isinstance(p_artifact, dict) and p_artifact.get('status') == 'success'
        ]
        if successful_artifacts:
            group_orchestrator = GroupAnalyzer(main_logger, output_base_dir)
            group_orchestrator.run_group_analysis(successful_artifacts)
        else:
            main_logger.warning("No successfully processed participant artifacts available for group analysis.")
    else:
        main_logger.warning("No participant artifacts collected. Skipping group analysis.")

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