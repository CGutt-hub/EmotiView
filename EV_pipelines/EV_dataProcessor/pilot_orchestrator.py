import os
import mne
import pandas as pd
import numpy as np
import logging # For main logger
from . import config
from .preprocessing.eeg_preprocessor import EEGPreprocessor
from .preprocessing.ecg_preprocessor import ECGPreprocessor
from .preprocessing.eda_preprocessor import EDAPreprocessor
from .preprocessing.fnirs_preprocessor import FNIRSPreprocessor
from .analysis.analysis_service import AnalysisService
from .reporting.plotting_service import PlottingService
from .utils.data_loader import DataLoader
from .utils.event_processor import EventProcessor
from .utils.participant_logger import ParticipantLogger
from .utils import select_eeg_channels_by_fnirs_rois, apply_fdr_correction

# Constants for data types
DATA_TYPES = ['eeg', 'fnirs', 'ecg', 'eda', 'events', 'survey']
EMOTIONAL_CONDITIONS = ['Positive', 'Negative', 'Neutral'] # As per proposal, adjust if needed

# Define homologous frontal pairs for FAI (example, adjust based on actual cap layout and proposal)
# These should be actual channel names present in your EEG data.
FAI_ELECTRODE_PAIRS = [('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8')]

def create_output_directories(base_dir, participant_id):
    """Creates necessary output directories for a participant."""
    dirs = {
        'base_participant': os.path.join(base_dir, participant_id),
        'raw_data_copied': os.path.join(base_dir, participant_id, 'raw_data_copied'),
        'preprocessed_data': os.path.join(base_dir, participant_id, 'preprocessed_data'),
        'analysis_results': os.path.join(base_dir, participant_id, 'analysis_results'),
        'plots_root': os.path.join(base_dir, participant_id, 'plots') # Root for all plots for this participant
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def process_participant_data(participant_id, participant_files, output_dirs, global_output_base_dir, p_logger):
    """
    Main processing pipeline for a single participant.
    Includes loading, preprocessing, analysis, and reporting.
    """
    p_logger.info(f"Starting processing for participant: {participant_id}")

    # --- Output Subdirectories ---
    preproc_results_dir = output_dirs['preprocessed_data']
    analysis_results_dir = output_dirs['analysis_results']
    # Plots will be saved by PlottingService in subdirs of output_dirs['plots_root']

    # --- Initialize Services ---
    data_loader = DataLoader(p_logger)
    event_processor = EventProcessor(p_logger)
    analysis_service = AnalysisService(p_logger)
    plotting_service = PlottingService(p_logger, output_dirs['plots_root'])


    # Dictionary to store paths and objects for this participant
    processed_data_artifacts = {
        'participant_id': participant_id,
        'log_file': p_logger.handlers[0].baseFilename if p_logger.handlers else None, # Get log file path
        'raw_data_paths': {},
        'event_times_df': None,
        'preprocessed_data_paths': {},
        'mne_objects': {'raw': {}, 'epochs': {}}, # Store raw and epoched MNE objects
        'analysis_outputs': {'metrics': {}, 'stats': {}, 'dataframes': {}}
    }

    # --- Data Loading ---
    p_logger.info("--- Loading Data ---")
    # Use the DataLoader instance to load all data for the participant
    # The DataLoader's load_all_data method should populate baseline_start/end_time_sec if found
    loaded_data_package = data_loader.load_all_data(participant_id, participant_files)

    raw_eeg_mne = loaded_data_package.get('eeg_raw')
    eeg_sampling_rate = loaded_data_package.get('eeg_sfreq')
    processed_data_artifacts['mne_objects']['raw']['eeg'] = raw_eeg_mne
    
    fnirs_od_mne = loaded_data_package.get('fnirs_raw_od')
    fnirs_sampling_rate = loaded_data_package.get('fnirs_sfreq')
    processed_data_artifacts['mne_objects']['raw']['fnirs_od'] = fnirs_od_mne

    ecg_data_raw = loaded_data_package.get('ecg_data')
    ecg_sampling_rate = loaded_data_package.get('ecg_sfreq')
    
    eda_data_raw = loaded_data_package.get('eda_data')
    eda_sampling_rate = loaded_data_package.get('eda_sfreq')
    
    # Load survey data (for WP2)
    survey_df = data_loader.load_survey_data(participant_files.get('survey'), participant_id) # Keep this separate if not in load_all_data
    processed_data_artifacts['analysis_outputs']['dataframes']['survey_data_raw'] = survey_df


    # --- Event Processing ---
    event_file_path = participant_files.get('events')
    if event_file_path:
        events_df_raw = event_processor.process_event_log(event_file_path)
        if events_df_raw is not None and not events_df_raw.empty:
            # Store baseline times if DataLoader populated them
            processed_data_artifacts['baseline_start_time_sec'] = loaded_data_package.get('baseline_start_time_sec')
            processed_data_artifacts['baseline_end_time_sec'] = loaded_data_package.get('baseline_end_time_sec')
            
            ref_sfreq_for_events = eeg_sampling_rate if eeg_sampling_rate else (fnirs_sampling_rate if fnirs_sampling_rate else None)
            if ref_sfreq_for_events:
                 if 'onset_time_sec' in events_df_raw.columns: 
                    events_df_raw['onset_sample'] = (events_df_raw['onset_time_sec'] * ref_sfreq_for_events).astype(int)
                 else:
                    p_logger.warning("Event file loaded but 'onset_time_sec' column missing. Cannot calculate 'onset_sample'.")
                    events_df_raw = None 
            else:
                p_logger.warning("Cannot determine reference sampling rate for event 'onset_sample' calculation.")
                events_df_raw = None

            processed_data_artifacts['event_times_df'] = events_df_raw
            if events_df_raw is not None:
                event_csv_path = os.path.join(preproc_results_dir, f"{participant_id}_event_times_processed.csv")
                events_df_raw.to_csv(event_csv_path, index=False)
                processed_data_artifacts['preprocessed_data_paths']['event_times_csv'] = event_csv_path
                p_logger.info(f"Processed event times saved to: {event_csv_path}")
        else:
            p_logger.warning("Event processing did not return a valid DataFrame.")
    else:
        p_logger.warning("No event file found or loaded.")

    # --- Preprocessing Stage ---
    p_logger.info("--- Preprocessing Stage ---")
    # EEG
    if raw_eeg_mne:
        eeg_preprocessor = EEGPreprocessor(p_logger)
        raw_eeg_processed = eeg_preprocessor.process(raw_eeg_mne.copy()) # Process a copy
        processed_data_artifacts['mne_objects']['raw']['eeg_processed'] = raw_eeg_processed
        if raw_eeg_processed:
            eeg_processed_path = os.path.join(preproc_results_dir, f"{participant_id}_eeg_processed_raw.fif")
            raw_eeg_processed.save(eeg_processed_path, overwrite=True, verbose=False)
            processed_data_artifacts['preprocessed_data_paths']['eeg_processed_fif'] = eeg_processed_path
            p_logger.info(f"Processed EEG data saved to: {eeg_processed_path}")
    else:
        p_logger.warning("No EEG data loaded. Skipping EEG preprocessing.")

    # fNIRS
    if fnirs_od_mne:
        fnirs_preprocessor = FNIRSPreprocessor(p_logger)
        fnirs_haemo_processed = fnirs_preprocessor.process(fnirs_od_mne.copy()) # Process a copy
        processed_data_artifacts['mne_objects']['raw']['fnirs_haemo_processed'] = fnirs_haemo_processed
        if fnirs_haemo_processed:
            fnirs_haemo_path = fnirs_preprocessor.save_preprocessed_data(fnirs_haemo_processed, participant_id, preproc_results_dir)
            if fnirs_haemo_path:
                processed_data_artifacts['preprocessed_data_paths']['fnirs_haemo_fif'] = fnirs_haemo_path
            p_logger.info("fNIRS data preprocessed to haemoglobin concentration.")
    else:
        p_logger.warning("No fNIRS data loaded. Skipping fNIRS preprocessing.")

    # ECG
    if ecg_data_raw is not None and ecg_sampling_rate is not None:
        ecg_preprocessor = ECGPreprocessor(p_logger)
        # For overall NN-intervals (e.g., if no baseline period is strictly defined for some analyses)
        nn_intervals_ms_overall, _, rpeaks_samples_overall = ecg_preprocessor.preprocess_ecg_neurokit(
            ecg_data_raw, ecg_sampling_rate, participant_id, preproc_results_dir
        )
        processed_data_artifacts['analysis_outputs']['metrics']['ecg_nn_intervals_ms_overall'] = nn_intervals_ms_overall
        processed_data_artifacts['ecg_rpeaks_samples_overall'] = rpeaks_samples_overall # Store for potential trial-wise HRV

        # For WP3: Baseline RMSSD - requires segmenting ECG first
        baseline_start_sec = processed_data_artifacts.get('baseline_start_time_sec')
        baseline_end_sec = processed_data_artifacts.get('baseline_end_time_sec')

        if baseline_start_sec is not None and baseline_end_sec is not None and baseline_start_sec < baseline_end_sec:
            p_logger.info(f"Attempting to extract baseline ECG segment: {baseline_start_sec:.2f}s to {baseline_end_sec:.2f}s")
            start_sample_baseline = int(baseline_start_sec * ecg_sampling_rate)
            end_sample_baseline = int(baseline_end_sec * ecg_sampling_rate)
            if start_sample_baseline < end_sample_baseline and end_sample_baseline <= len(ecg_data_raw):
                ecg_baseline_segment = ecg_data_raw[start_sample_baseline:end_sample_baseline]
                nn_intervals_ms_baseline, _, _ = ecg_preprocessor.preprocess_ecg_neurokit(
                    ecg_baseline_segment, ecg_sampling_rate, f"{participant_id}_baseline", preproc_results_dir 
                )
                processed_data_artifacts['analysis_outputs']['metrics']['ecg_nn_intervals_ms_baseline'] = nn_intervals_ms_baseline
            else:
                p_logger.warning("Baseline ECG segment indices are invalid or out of bounds.")
        
        if nn_intervals_ms_overall is not None:
             p_logger.info(f"ECG preprocessed (overall). Found {len(nn_intervals_ms_overall)} NN intervals.")
    else:
        p_logger.warning("No ECG data loaded or sampling rate missing. Skipping ECG preprocessing.")

    # EDA
    if eda_data_raw is not None and eda_sampling_rate is not None:
        eda_preprocessor = EDAPreprocessor(p_logger)
        phasic_eda_path, tonic_eda_path = eda_preprocessor.preprocess_eda(
            eda_data_raw, eda_sampling_rate, participant_id, preproc_results_dir
        )
        processed_data_artifacts['preprocessed_data_paths']['phasic_eda_csv'] = phasic_eda_path
        processed_data_artifacts['preprocessed_data_paths']['tonic_eda_csv'] = tonic_eda_path
        if phasic_eda_path:
            p_logger.info("EDA preprocessed. Phasic and Tonic components saved.")
    else:
        p_logger.warning("No EDA data loaded or sampling rate missing. Skipping EDA preprocessing.")

    # --- Analysis Stage ---
    p_logger.info("--- Starting Analysis Stage ---")
    events_df = processed_data_artifacts.get('event_times_df')
    if events_df is None or events_df.empty or 'onset_sample' not in events_df.columns:
        p_logger.error("Event data is missing or invalid for epoching. Cannot proceed with epoch-based analysis.")
        return processed_data_artifacts

    # Map conditions to event_ids for MNE
    event_id_map = {name: i+1 for i, name in enumerate(EMOTIONAL_CONDITIONS)}
    if 'condition' not in events_df.columns:
        p_logger.error("Events DataFrame missing 'condition' column for epoching.")
        return processed_data_artifacts
        
    events_df['condition_id'] = events_df['condition'].map(event_id_map).fillna(0).astype(int) 
    
    mne_events_df = events_df[events_df['condition_id'] > 0][['onset_sample', 'condition_id']].copy()
    if mne_events_df.empty:
        p_logger.error("No valid events found after mapping conditions for epoching.")
        return processed_data_artifacts

    mne_events_df.insert(1, 'prev_event_id', 0) 
    mne_events_array = mne_events_df.values.astype(int)

    # Epoch EEG
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

    # Epoch fNIRS
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
    p_logger.info("--- WP1: Emotional Modulation of Synchrony ---")
    fnirs_epochs = processed_data_artifacts['mne_objects']['epochs'].get('fnirs')
    eeg_epochs = processed_data_artifacts['mne_objects']['epochs'].get('eeg')
    rpeaks_samples_overall = processed_data_artifacts.get('ecg_rpeaks_samples_overall')
    phasic_eda_csv_path = processed_data_artifacts['preprocessed_data_paths'].get('phasic_eda_csv')
    
    # 1. fNIRS GLM to identify ROIs
    active_fnirs_roi_names_for_eeg_guidance = []
    if fnirs_epochs and raw_eeg_proc: 
        fnirs_glm_output = analysis_service.run_fnirs_glm_and_contrasts(
            fnirs_epochs, events_df, event_id_map # Pass event_id_map used for epoching
        )
        processed_data_artifacts['analysis_outputs']['stats']['fnirs_glm_output'] = fnirs_glm_output
        # Plot per-participant fNIRS contrast results (placeholder plot for now)
        if fnirs_glm_output and 'contrast_results' in fnirs_glm_output:
            for contrast_name_plot, contrast_df_plot in fnirs_glm_output['contrast_results'].items():
                if contrast_df_plot is not None and not contrast_df_plot.empty:
                    # Pass the DataFrame to the plotting function
                    plotting_service.plot_fnirs_contrast_results(participant_id, contrast_name_plot, contrast_df_plot, f"participant_fnirs_contrast_{contrast_name_plot}")

        active_fnirs_roi_names_for_eeg_guidance = fnirs_glm_output.get('active_rois_for_eeg_guidance', [])
        
        # 2. fNIRS-Guided EEG Channel Selection
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

    # 3. Calculate PLV per trial, then average per condition
    all_trial_plv_results = []
    if eeg_epochs and eeg_channels_for_plv_wp1 and \
       (rpeaks_samples_overall is not None or phasic_eda_csv_path):
        p_logger.info("Calculating trial-wise PLV for WP1...")
        for condition_name, _ in event_id_map.items(): # Iterate through defined conditions
            try:
                eeg_epochs_condition = eeg_epochs[condition_name]
            except KeyError:
                p_logger.warning(f"No epochs found for condition '{condition_name}' in EEG data. Skipping PLV for this condition.")
                continue

            for i in range(len(eeg_epochs_condition)): 
                current_eeg_epoch = eeg_epochs_condition[i]
                eeg_trial_data_multichannel = current_eeg_epoch.get_data(picks=eeg_channels_for_plv_wp1) 
                if eeg_trial_data_multichannel.size == 0:
                    p_logger.warning(f"WP1 PLV: Empty EEG data for trial {i}, condition {condition_name}. Skipping trial.")
                    continue
                eeg_trial_data_avg = eeg_trial_data_multichannel.mean(axis=0).ravel() 
                eeg_trial_sfreq = current_eeg_epoch.info['sfreq']
                
                trial_start_time_sec = (current_eeg_epoch.events[0,0] / raw_eeg_proc.info['sfreq']) + current_eeg_epoch.tmin
                trial_end_time_sec = trial_start_time_sec + (len(eeg_trial_data_avg) / eeg_trial_sfreq)

                # EEG-HRV PLV for this trial
                if rpeaks_samples_overall is not None and ecg_sampling_rate is not None:
                    trial_rpeak_times_sec_abs = [r_idx / ecg_sampling_rate for r_idx in rpeaks_samples_overall]
                    trial_rpeaks_in_window_abs_times = [
                        t for t in trial_rpeak_times_sec_abs if trial_start_time_sec <= t < trial_end_time_sec
                    ]
                    if len(trial_rpeaks_in_window_abs_times) >= 2:
                        nn_intervals_trial_ms = np.diff(trial_rpeaks_in_window_abs_times) * 1000
                        _, cont_hrv_signal_trial = analysis_service.interpolate_nn_intervals(
                            nn_intervals_trial_ms, original_sfreq=ecg_sampling_rate, 
                            target_sfreq=config.PLV_RESAMPLE_SFREQ_AUTONOMIC
                        )
                        if cont_hrv_signal_trial is not None:
                            hrv_segment_for_plv = analysis_service.resample_signal(
                                cont_hrv_signal_trial, config.PLV_RESAMPLE_SFREQ_AUTONOMIC, eeg_trial_sfreq
                            )
                            if hrv_segment_for_plv is not None and len(hrv_segment_for_plv) >= len(eeg_trial_data_avg):
                                 hrv_segment_for_plv = hrv_segment_for_plv[:len(eeg_trial_data_avg)] 
                            elif hrv_segment_for_plv is not None: 
                                 hrv_segment_for_plv = np.pad(hrv_segment_for_plv, (0, len(eeg_trial_data_avg) - len(hrv_segment_for_plv)), 'edge')

                            if hrv_segment_for_plv is not None and len(hrv_segment_for_plv) == len(eeg_trial_data_avg):
                                for band_name, band_freqs in config.PLV_EEG_BANDS.items():
                                    plv_val = analysis_service.calculate_plv(eeg_trial_data_avg, hrv_segment_for_plv, eeg_trial_sfreq, band_freqs)
                                    if plv_val is not None:
                                        all_trial_plv_results.append({
                                            'participant_id': participant_id, 'condition': condition_name, 'trial': i,
                                            'modality_pair': 'EEG-HRV', 'eeg_band': band_name, 'plv': plv_val
                                        })
                # EEG-EDA PLV for this trial
                if phasic_eda_csv_path:
                    phasic_eda_df = pd.read_csv(phasic_eda_csv_path)
                    phasic_eda_full_signal = phasic_eda_df['EDA_Phasic'].values
                    eda_original_sfreq = eda_sampling_rate if eda_sampling_rate else config.EDA_SAMPLING_RATE_DEFAULT
                    
                    start_sample_eda_trial = int(trial_start_time_sec * eda_original_sfreq)
                    end_sample_eda_trial = int(trial_end_time_sec * eda_original_sfreq)

                    if start_sample_eda_trial < end_sample_eda_trial and end_sample_eda_trial <= len(phasic_eda_full_signal) and start_sample_eda_trial >=0 :
                        eda_trial_segment_raw = phasic_eda_full_signal[start_sample_eda_trial:end_sample_eda_trial]
                        
                        if eda_trial_segment_raw.size > 0:
                            eda_segment_for_plv = analysis_service.resample_signal(
                                eda_trial_segment_raw, eda_original_sfreq, eeg_trial_sfreq 
                            )
                            if eda_segment_for_plv is not None and len(eda_segment_for_plv) >= len(eeg_trial_data_avg):
                                eda_segment_for_plv = eda_segment_for_plv[:len(eeg_trial_data_avg)]
                            elif eda_segment_for_plv is not None:
                                eda_segment_for_plv = np.pad(eda_segment_for_plv, (0, len(eeg_trial_data_avg) - len(eda_segment_for_plv)), 'edge')

                            if eda_segment_for_plv is not None and len(eda_segment_for_plv) == len(eeg_trial_data_avg):
                                for band_name, band_freqs in config.PLV_EEG_BANDS.items():
                                    plv_val = analysis_service.calculate_plv(eeg_trial_data_avg, eda_segment_for_plv, eeg_trial_sfreq, band_freqs)
                                    if plv_val is not None:
                                        all_trial_plv_results.append({
                                            'participant_id': participant_id, 'condition': condition_name, 'trial': i,
                                            'modality_pair': 'EEG-EDA', 'eeg_band': band_name, 'plv': plv_val
                                        })
                        else:
                            p_logger.warning(f"WP1 PLV: EDA trial segment empty for trial {i}, condition {condition_name}.")
                    else:
                        p_logger.warning(f"WP1 PLV: EDA trial segment indices out of bounds for trial {i}, condition {condition_name}.")
        
        trial_plv_df = pd.DataFrame(all_trial_plv_results)
        processed_data_artifacts['analysis_outputs']['dataframes']['trial_plv_wp1'] = trial_plv_df
        if not trial_plv_df.empty:
            p_logger.info(f"Calculated {len(trial_plv_df)} trial-wise PLV values for WP1.")
            avg_plv_wp1_df = trial_plv_df.groupby(['participant_id', 'condition', 'modality_pair', 'eeg_band'])['plv'].mean().reset_index()
            processed_data_artifacts['analysis_outputs']['dataframes']['avg_plv_wp1'] = avg_plv_wp1_df
            
            plotting_service.plot_plv_results(participant_id, avg_plv_wp1_df, "wp1_avg_plv")
            p_logger.info("ANOVA for WP1 PLV will be performed at the group level.")


    # --- Work Package 2: Synchrony and Subjective Arousal ---
    p_logger.info("--- WP2: Synchrony and Subjective Arousal ---")
    survey_data_raw = processed_data_artifacts['analysis_outputs']['dataframes'].get('survey_data_raw')
    trial_plv_df_wp1 = processed_data_artifacts['analysis_outputs']['dataframes'].get('trial_plv_wp1')

    if survey_data_raw is not None and not survey_data_raw.empty and \
       trial_plv_df_wp1 is not None and not trial_plv_df_wp1.empty:
        
        # This merge requires survey_data_raw to have 'condition' and 'trial' (or equivalent)
        # to match with trial_plv_df_wp1.
        # Example: if survey has 'Condition' and 'TrialNum' and PLV df has 'condition' and 'trial'
        # merged_wp2_df = pd.merge(trial_plv_df_wp1, survey_data_raw, 
        #                          left_on=['condition', 'trial'], 
        #                          right_on=['Condition', 'TrialNum'], # Adjust column names
        #                          how='inner')
        # if not merged_wp2_df.empty and 'sam_arousal' in merged_wp2_df.columns:
        #     processed_data_artifacts['analysis_outputs']['dataframes']['merged_plv_arousal_wp2'] = merged_wp2_df
        #     p_logger.info(f"WP2: Merged PLV and SAM arousal data for {participant_id}. N={len(merged_wp2_df)}")
        # else:
        #     p_logger.warning(f"WP2: Could not merge PLV and SAM arousal data, or 'sam_arousal' missing. Check column names and trial identifiers.")
        
        # For now, storing the raw components for group-level merging and correlation
        if 'sam_arousal' in survey_data_raw.columns:
            processed_data_artifacts['analysis_outputs']['metrics']['wp2_has_sam_arousal'] = True
    else:
        p_logger.info("Skipping WP2 prep due to missing survey or PLV data for this participant.")


    # --- Work Package 3: Baseline Vagal Tone and Task-Related Synchrony ---
    p_logger.info("--- WP3: Baseline Vagal Tone and Task-Related Synchrony ---")
    nn_intervals_baseline = processed_data_artifacts['analysis_outputs']['metrics'].get('ecg_nn_intervals_ms_baseline')
    if nn_intervals_baseline is not None:
        baseline_rmssd = analysis_service.calculate_rmssd(nn_intervals_baseline)
        processed_data_artifacts['analysis_outputs']['metrics']['baseline_rmssd'] = baseline_rmssd
        if baseline_rmssd is not None:
            p_logger.info(f"Calculated baseline RMSSD: {baseline_rmssd:.2f} ms")
            avg_plv_df_wp1 = processed_data_artifacts['analysis_outputs']['dataframes'].get('avg_plv_wp1')
            if avg_plv_df_wp1 is not None and not avg_plv_df_wp1.empty:
                plv_negative_series = avg_plv_df_wp1[
                    (avg_plv_df_wp1['condition'] == 'Negative') &
                    (avg_plv_df_wp1['modality_pair'] == 'EEG-HRV') & 
                    (avg_plv_df_wp1['eeg_band'] == 'Alpha')         
                ]['plv']
                if not plv_negative_series.empty:
                    avg_plv_negative = plv_negative_series.mean() # This is per participant
                    processed_data_artifacts['analysis_outputs']['metrics']['wp3_avg_plv_negative_alpha_hrv'] = avg_plv_negative
                    p_logger.info(f"WP3: Participant {participant_id} - Baseline RMSSD: {baseline_rmssd:.2f}, Avg PLV (Alpha-HRV-Negative): {avg_plv_negative:.3f}")
    # Actual correlation for WP3 will be done at group level.


    # --- Work Package 4: Frontal Asymmetry and Branch-Specific Synchrony ---
    p_logger.info("--- WP4: Frontal Asymmetry and Branch-Specific Synchrony ---")
    fai_results_list = []
    if eeg_epochs:
        for left_elec, right_elec in FAI_ELECTRODE_PAIRS:
            if left_elec not in eeg_epochs.ch_names or right_elec not in eeg_epochs.ch_names:
                p_logger.warning(f"FAI: Electrodes {left_elec} or {right_elec} not found in EEG data. Skipping pair.")
                continue

            power_left_obj = analysis_service.calculate_band_power(eeg_epochs, config.FAI_ALPHA_BAND, picks=[left_elec])
            power_right_obj = analysis_service.calculate_band_power(eeg_epochs, config.FAI_ALPHA_BAND, picks=[right_elec])

            if power_left_obj is not None and power_right_obj is not None:
                power_left = power_left_obj[0] 
                power_right = power_right_obj[0]
                fai_value = analysis_service.calculate_fai(power_right, power_left)
                if fai_value is not None:
                    fai_results_list.append({'participant_id': participant_id, 'pair': f"{left_elec}-{right_elec}", 'fai': fai_value})
                    p_logger.info(f"FAI for {left_elec}-{right_elec}: {fai_value:.3f}")
        
        fai_df = pd.DataFrame(fai_results_list)
        processed_data_artifacts['analysis_outputs']['dataframes']['fai_wp4'] = fai_df
        if not fai_df.empty:
            p_logger.info("Calculated FAI for specified electrode pairs.")
            # Store FAI for F4-F3 pair for group correlation example
            fai_f4f3_series = fai_df[fai_df['pair'] == 'F4-F3']['fai']
            if not fai_f4f3_series.empty:
                 processed_data_artifacts['analysis_outputs']['metrics']['wp4_fai_f4f3'] = fai_f4f3_series.iloc[0]
    # Actual correlation for WP4 will be done at group level.

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
        participant_ids = [d for d in os.listdir(data_root_dir) if os.path.isdir(os.path.join(data_root_dir, d))]
        if not participant_ids:
            main_logger.error(f"No participant subdirectories found in {data_root_dir}.")
            return []
        main_logger.info(f"Found {len(participant_ids)} potential participants: {participant_ids}")
    
    all_participants_summary = []

    for p_id in participant_ids:
        main_logger.info(f"--- Processing participant: {p_id} ---")
        participant_output_dirs = create_output_directories(output_base_dir, p_id)
        
        p_log_manager = ParticipantLogger(participant_output_dirs['base_participant'], p_id, config.LOG_LEVEL)
        participant_logger_instance = p_log_manager.get_logger()
        
        try:
            temp_data_loader = DataLoader(participant_logger_instance) 
            participant_files = temp_data_loader.find_participant_files(p_id, data_root_dir, DATA_TYPES)
            
            if not any(val for val in participant_files.values() if val is not None):
                participant_logger_instance.warning(f"No data files found for participant {p_id} in {data_root_dir}. Skipping.")
                all_participants_summary.append({
                    'participant_id': p_id, 'status': 'no_data_found', 
                    'log_file': p_log_manager.log_file_path
                })
                continue

            processed_artifacts = process_participant_data(
                p_id, participant_files, participant_output_dirs, output_base_dir, participant_logger_instance
            )
            all_participants_summary.append(processed_artifacts) 
            participant_logger_instance.info(f"--- Successfully processed participant: {p_id} ---")
        except Exception as e:
            participant_logger_instance.error(f"--- CRITICAL ERROR processing participant {p_id}: {e} ---", exc_info=True)
            all_participants_summary.append({
                'participant_id': p_id, 'status': 'error', 'error_message': str(e),
                'log_file': p_log_manager.log_file_path
            })
        finally:
            p_log_manager.close_handlers() 

    main_logger.info("--- EmotiView Data Processor Finished ---")
    
    # --- Aggregate results across participants and Run Group-Level Analyses ---
    group_analysis_service = AnalysisService(main_logger)
    group_plotting_service = PlottingService(main_logger, os.path.join(output_base_dir, "_GROUP_PLOTS"))
    group_results_dir = os.path.join(output_base_dir, "_GROUP_RESULTS")
    os.makedirs(group_results_dir, exist_ok=True)

    # WP1: ANOVA on PLV
    all_avg_plv_wp1_dfs = [
        res['analysis_outputs']['dataframes']['avg_plv_wp1'] 
        for res in all_participants_summary 
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('avg_plv_wp1') is not None
    ]
    if all_avg_plv_wp1_dfs:
        group_plv_data_wp1 = pd.concat(all_avg_plv_wp1_dfs, ignore_index=True)
        if not group_plv_data_wp1.empty:
            main_logger.info(f"WP1: Aggregated PLV data from {group_plv_data_wp1['participant_id'].nunique()} participants for group ANOVA.")
            # Example ANOVA: PLV ~ condition, for Alpha band, EEG-HRV
            df_for_anova_wp1 = group_plv_data_wp1[
                (group_plv_data_wp1['eeg_band'] == 'Alpha') & 
                (group_plv_data_wp1['modality_pair'] == 'EEG-HRV')
            ].copy() # Use .copy() to avoid SettingWithCopyWarning
            
            if not df_for_anova_wp1.empty:
                anova_results_wp1 = group_analysis_service.run_repeated_measures_anova(
                   data_df=df_for_anova_wp1, dv='plv', 
                   within='condition', subject='participant_id'
                )
                main_logger.info(f"Group ANOVA results for PLV (WP1, Alpha, EEG-HRV):\n{anova_results_wp1}")
                if anova_results_wp1 is not None and not anova_results_wp1.empty:
                    # --- FDR Correction for ANOVA p-values ---
                    # Example: Correct p-values for the 'Source' column (main effects, interactions)
                    p_values_to_correct_wp1 = anova_results_wp1['p-unc'].dropna().tolist()
                    if p_values_to_correct_wp1:
                        reject_fdr_wp1, pval_corr_fdr_wp1 = apply_fdr_correction(p_values_to_correct_wp1, alpha=0.05)
                        # Add corrected p-values back to the DataFrame (aligning by original index)
                        fdr_series_pval = pd.Series(pval_corr_fdr_wp1, index=anova_results_wp1.dropna(subset=['p-unc']).index)
                        fdr_series_reject = pd.Series(reject_fdr_wp1, index=anova_results_wp1.dropna(subset=['p-unc']).index)
                        anova_results_wp1['p-corr-fdr'] = fdr_series_pval
                        anova_results_wp1['reject_fdr'] = fdr_series_reject
                        main_logger.info(f"WP1 ANOVA with FDR correction:\n{anova_results_wp1}")
                    
                    if 'p-unc' in anova_results_wp1.columns:
                        # Apply FDR to relevant p-values (e.g., for the 'condition' factor)
                        p_values_condition_effect = anova_results_wp1[anova_results_wp1['Source'] == 'condition']['p-unc'].dropna()
                        if not p_values_condition_effect.empty:
                            reject_fdr, pval_corr_fdr = apply_fdr_correction(p_values_condition_effect.tolist(), alpha=0.05)
                            # Add these back carefully if needed, or report separately
                            main_logger.info(f"WP1 ANOVA 'condition' effect (Source specific FDR): p-values={p_values_condition_effect.tolist()}, FDR corrected reject={reject_fdr}, p-corr={pval_corr_fdr}")
                    
                    anova_results_wp1.to_csv(os.path.join(group_results_dir, "group_anova_wp1_plv_alpha_hrv.csv"))
                    group_plotting_service.plot_anova_results("GROUP", anova_results_wp1, df_for_anova_wp1, 'plv', 'condition', "WP1: PLV (Alpha, HRV) by Condition", "wp1_plv_alpha_hrv")

    # WP2: Correlation (Synchrony vs. Subjective Arousal)
    # First, collect all trial-level PLV and survey data
    all_trial_plv_wp1_dfs = [
        res['analysis_outputs']['dataframes']['trial_plv_wp1']
        for res in all_participants_summary
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('trial_plv_wp1') is not None
    ]
    all_survey_dfs = [
        res['analysis_outputs']['dataframes']['survey_data_raw']
        for res in all_participants_summary
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('survey_data_raw') is not None
    ]

    if all_trial_plv_wp1_dfs and all_survey_dfs:
        group_trial_plv_df = pd.concat(all_trial_plv_wp1_dfs, ignore_index=True)
        group_survey_df = pd.concat(all_survey_dfs, ignore_index=True)

        if not group_trial_plv_df.empty and not group_survey_df.empty and 'sam_arousal' in group_survey_df.columns:
            # Merge requires common columns like 'participant_id', 'condition', 'trial'
            # Adjust column names in survey_df if they differ (e.g., 'Condition' vs 'condition')
            # Example: group_survey_df.rename(columns={'OldTrialCol': 'trial', 'OldCondCol': 'condition'}, inplace=True)
            try:
                merged_wp2_group_df = pd.merge(group_trial_plv_df, group_survey_df, 
                                               on=['participant_id', 'condition', 'trial'], # Adjust as needed
                                               how='inner')
                if not merged_wp2_group_df.empty:
                    # Select specific PLV for correlation, e.g., Alpha, EEG-HRV
                    df_for_corr_wp2 = merged_wp2_group_df[
                        (merged_wp2_group_df['eeg_band'] == 'Alpha') &
                        (merged_wp2_group_df['modality_pair'] == 'EEG-HRV')
                    ].copy()
                    if not df_for_corr_wp2.empty:
                        corr_wp2 = group_analysis_service.run_correlation_analysis(
                            df_for_corr_wp2['sam_arousal'], df_for_corr_wp2['plv'], 
                            name1='SAM_Arousal', name2='Trial_PLV_Alpha_HRV'
                        )
                        main_logger.info(f"Group Correlation (WP2 - Trial SAM Arousal vs Trial PLV):\n{corr_wp2}")
                        if corr_wp2 is not None: 
                            # If multiple correlations are run for WP2, collect their p-values for FDR
                            # For a single correlation, FDR is not applied to its own p-value in isolation.
                            # Example: if you had corr_wp2_hrv, corr_wp2_eda, etc.
                            # p_vals_wp2 = [corr_wp2_hrv['p-val'].iloc[0], corr_wp2_eda['p-val'].iloc[0]]
                            # _, p_corr_wp2 = apply_fdr_correction(p_vals_wp2)
                            # Then add p_corr_wp2 back to respective DataFrames or report.
                            corr_wp2.to_csv(os.path.join(group_results_dir, "group_corr_wp2_trial_arousal_vs_plv.csv")) # Save the full table
                            group_plotting_service.plot_correlation("GROUP", df_for_corr_wp2['sam_arousal'], df_for_corr_wp2['plv'], 
                                                                    "SAM Arousal (Trial)", "PLV (Alpha-HRV, Trial)", 
                                                                    "WP2: Trial Arousal vs PLV", "wp2_trial_arousal_plv", corr_wp2)
            except Exception as e_merge_wp2:
                 main_logger.error(f"WP2: Error merging PLV and survey data for group analysis: {e_merge_wp2}")


    # WP3: Correlation (Baseline RMSSD vs. Task PLV)
    wp3_data_list = [{'participant_id': r['participant_id'],
                      'baseline_rmssd': r['analysis_outputs']['metrics'].get('baseline_rmssd'),
                      'plv_negative_alpha_hrv': r['analysis_outputs']['metrics'].get('wp3_avg_plv_negative_alpha_hrv')}
                     for r in all_participants_summary if isinstance(r, dict) and r.get('analysis_outputs', {}).get('metrics', {}).get('baseline_rmssd') is not None]
    if wp3_data_list:
        wp3_group_df = pd.DataFrame(wp3_data_list).dropna()
        if not wp3_group_df.empty and len(wp3_group_df) >=3:
            corr_wp3 = group_analysis_service.run_correlation_analysis(wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_alpha_hrv'], name1='Baseline_RMSSD', name2='Avg_PLV_Alpha_HRV_Negative')
            main_logger.info(f"Group Correlation (WP3 - Baseline RMSSD vs Negative PLV):\n{corr_wp3}")
            if corr_wp3 is not None:
                 corr_wp3.to_csv(os.path.join(group_results_dir, "group_corr_wp3_rmssd_vs_plv_neg.csv")) # Save the full table
                 group_plotting_service.plot_correlation("GROUP", wp3_group_df['baseline_rmssd'], wp3_group_df['plv_negative_alpha_hrv'], "Baseline RMSSD (ms)", "Avg PLV (Alpha-HRV-Negative)", "WP3: RMSSD vs Negative PLV", "wp3_rmssd_plv_neg", corr_wp3)

    # WP4: Correlation (FAI vs. Branch-Specific PLV)
    wp4_fai_list = []
    for res in all_participants_summary:
        if isinstance(res, dict) and res.get('analysis_outputs', {}).get('dataframes', {}).get('fai_wp4') is not None:
            fai_df_participant = res['analysis_outputs']['dataframes']['fai_wp4']
            if not fai_df_participant.empty:
                 # Get FAI for F4-F3 pair
                 fai_f4f3_val = fai_df_participant[fai_df_participant['pair'] == 'F4-F3']['fai'].values
                 if len(fai_f4f3_val) > 0:
                     wp4_fai_list.append({'participant_id': res['participant_id'], 'fai_f4f3': fai_f4f3_val[0]})
    
    # Combine with relevant PLV data (e.g., average PLV for EEG-HRV and EEG-EDA separately)
    if wp4_fai_list and all_avg_plv_wp1_dfs: # Reusing all_avg_plv_wp1_dfs
        wp4_fai_group_df = pd.DataFrame(wp4_fai_list)
        group_plv_data_wp1_for_wp4 = pd.concat(all_avg_plv_wp1_dfs, ignore_index=True)
        
        # Example: Correlate FAI F4-F3 with Alpha EEG-HRV PLV (averaged across conditions for simplicity)
        plv_hrv_alpha_avg_wp4 = group_plv_data_wp1_for_wp4[
            (group_plv_data_wp1_for_wp4['eeg_band'] == 'Alpha') &
            (group_plv_data_wp1_for_wp4['modality_pair'] == 'EEG-HRV')
        ].groupby('participant_id')['plv'].mean().reset_index().rename(columns={'plv': 'avg_plv_hrv_alpha'})

        merged_wp4_hrv = pd.merge(wp4_fai_group_df, plv_hrv_alpha_avg_wp4, on='participant_id')
        if not merged_wp4_hrv.empty and len(merged_wp4_hrv) >= 3:
            corr_wp4_hrv = group_analysis_service.run_correlation_analysis(merged_wp4_hrv['fai_f4f3'], merged_wp4_hrv['avg_plv_hrv_alpha'], name1='FAI_F4F3', name2='Avg_PLV_Alpha_HRV')
            main_logger.info(f"Group Correlation (WP4 - FAI F4-F3 vs EEG-HRV PLV):\n{corr_wp4_hrv}")
            if corr_wp4_hrv is not None:
                corr_wp4_hrv.to_csv(os.path.join(group_results_dir, "group_corr_wp4_fai_vs_plv_hrv.csv")) # Save the full table
                group_plotting_service.plot_correlation("GROUP", merged_wp4_hrv['fai_f4f3'], merged_wp4_hrv['avg_plv_hrv_alpha'], "FAI (F4-F3)", "Avg PLV (Alpha-HRV)", "WP4: FAI vs EEG-HRV PLV", "wp4_fai_plv_hrv", corr_wp4_hrv)
        
        # Similarly for EEG-EDA PLV
        plv_eda_alpha_avg_wp4 = group_plv_data_wp1_for_wp4[
            (group_plv_data_wp1_for_wp4['eeg_band'] == 'Alpha') &
            (group_plv_data_wp1_for_wp4['modality_pair'] == 'EEG-EDA')
        ].groupby('participant_id')['plv'].mean().reset_index().rename(columns={'plv': 'avg_plv_eda_alpha'})
        merged_wp4_eda = pd.merge(wp4_fai_group_df, plv_eda_alpha_avg_wp4, on='participant_id')
        if not merged_wp4_eda.empty and len(merged_wp4_eda) >= 3:
            corr_wp4_eda = group_analysis_service.run_correlation_analysis(merged_wp4_eda['fai_f4f3'], merged_wp4_eda['avg_plv_eda_alpha'], name1='FAI_F4F3', name2='Avg_PLV_Alpha_EDA')
            main_logger.info(f"Group Correlation (WP4 - FAI F4-F3 vs EEG-EDA PLV):\n{corr_wp4_eda}")
            if corr_wp4_eda is not None:
                corr_wp4_eda.to_csv(os.path.join(group_results_dir, "group_corr_wp4_fai_vs_plv_eda.csv")) # Save the full table
                group_plotting_service.plot_correlation("GROUP", merged_wp4_eda['fai_f4f3'], merged_wp4_eda['avg_plv_eda_alpha'], "FAI (F4-F3)", "Avg PLV (Alpha-EDA)", "WP4: FAI vs EEG-EDA PLV", "wp4_fai_plv_eda", corr_wp4_eda)

    # --- Create Final Summary ---
    summary_list = []
    for r_idx, r_val in enumerate(all_participants_summary):
        if isinstance(r_val, dict):
            summary_list.append({
                'participant_id': r_val.get('participant_id', f'unknown_participant_{r_idx}'),
                'status': r_val.get('status', 'error' if 'error_message' in r_val else 'processed'),
                'log_file': r_val.get('log_file')
            })
        else: # Should not happen if processed_artifacts is always a dict
            summary_list.append({'participant_id': f'unknown_participant_{r_idx}', 'status': 'unknown_format', 'log_file': None})
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(os.path.join(output_base_dir, "processing_summary.csv"), index=False)
    main_logger.info(f"Processing summary saved to {os.path.join(output_base_dir, 'processing_summary.csv')}")

    return all_participants_summary

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) 
    
    default_data_root = os.path.join(project_root, "sample_data") 
    default_output_root = os.path.join(project_root, "EV_Processed_Data_Orchestrator_Full")

    print(f"Default Data Root: {default_data_root}")
    print(f"Default Output Root: {default_output_root}")

    if not os.path.exists(default_data_root):
        os.makedirs(default_data_root)
        print(f"Created dummy data root: {default_data_root}")
        # You might want to create dummy participant folders here for testing
        # e.g., os.makedirs(os.path.join(default_data_root, "participant_01"))
    
    main_orchestrator(default_data_root, default_output_root, participant_ids=None)