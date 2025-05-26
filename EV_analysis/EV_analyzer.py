# d:\repoShaggy\EmotiView\EV_analysis\EV_analyzer.py
import os
import time
import mne
import pandas as pd
import numpy as np
import pickle # For saving/loading artifact dictionaries
import logging # For main logger
import argparse # For command-line arguments
from . import EV_config # This is correct as EV_config.py is in the same directory
from psyModuleToolbox.preprocessing.eeg_preprocessor import EEGPreprocessor
from psyModuleToolbox.preprocessing.ecg_preprocessor import ECGPreprocessor
from psyModuleToolbox.preprocessing.eda_preprocessor import EDAPreprocessor
from psyModuleToolbox.preprocessing.fnirs_preprocessor import FNIRSPreprocessor
from psyModuleToolbox.analysis.analysis_service import AnalysisService
from psyModuleToolbox.reporting.plotting_service import PlottingService
from psyModuleToolbox.data_handling.data_loader import DataLoader
# from psyModuleToolbox.utils.event_processor import EventProcessor # Removed - likely used within DataLoader
from psyModuleToolbox.utils.participant_logger import ParticipantLogger
# Import helper functions from the new helpers.py file within the utils folder
from psyModuleToolbox.analysis.group_analyzer import GroupAnalyzer
from psyModuleToolbox.utils.parallel_runner import ParallelTaskRunner
from psyModuleToolbox.utils.git_handler import commit_and_sync_changes
from psyModuleToolbox.utils.helpers import select_eeg_channels_by_fnirs_rois, create_output_directories, create_mne_events_from_dataframe

# Constants for data types (These are fine as they are internal to the analyzer's logic)
DATA_TYPES = ['eeg', 'fnirs', 'ecg', 'eda', 'events', 'survey']

def process_participant_data(participant_id, participant_raw_xdf_data_dir, eprime_txt_file_path, output_dirs, global_output_base_dir, p_logger, ev_config):
    """
    Main processing pipeline for a single participant.
    """
    p_logger.info(f"Starting processing for participant: {participant_id}")

    preproc_results_dir = output_dirs['preprocessed_data']
    analysis_results_dir = output_dirs['analysis_results']

    data_loader = DataLoader(
        p_logger,
        baseline_marker_start_eprime=ev_config.BASELINE_MARKER_START_EPRIME,
        baseline_marker_end_eprime=ev_config.BASELINE_MARKER_END_EPRIME,
        event_duration_default=ev_config.EVENT_DURATION_DEFAULT
    )
    # Pass ev_config to AnalysisService if it needs to access general configs
    # or to pass them down to analyzers it instantiates.
    analysis_service = AnalysisService(p_logger, main_config=ev_config)
    plotting_service = PlottingService(
        p_logger,
        output_dirs['plots_root'],
        reporting_figure_format_config=ev_config.REPORTING_FIGURE_FORMAT,
        reporting_dpi_config=ev_config.REPORTING_DPI
    )

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
    if participant_raw_xdf_data_dir and os.path.exists(participant_raw_xdf_data_dir):
         loaded_physiological_data = data_loader.load_participant_streams(participant_id, participant_raw_xdf_data_dir, eprime_txt_file_path)
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

    survey_df_per_trial = data_loader.load_survey_data(eprime_txt_file_path, participant_id)
    processed_data_artifacts['analysis_outputs']['dataframes']['survey_data_per_trial'] = survey_df_per_trial

    events_df = loaded_physiological_data.get('events_df')

    if events_df is not None and not events_df.empty:
        p_logger.info("Using events_df provided by DataLoader (assumed synchronized).")
        if 'onset_time_sec' not in events_df.columns:
            p_logger.error("Critical: 'onset_time_sec' column missing in events_df from DataLoader.")
            processed_data_artifacts['status'] = 'error_event_timing_missing'
            return processed_data_artifacts

        processed_data_artifacts['event_times_df'] = events_df
        event_csv_path = os.path.join(preproc_results_dir, f"{participant_id}_event_times_synchronized.csv")
        events_df.to_csv(event_csv_path, index=False)
        processed_data_artifacts['preprocessed_data_paths']['event_times_csv'] = event_csv_path
        p_logger.info(f"Event times (from DataLoader) saved to: {event_csv_path}")

        if processed_data_artifacts['baseline_start_time_sec'] is None and 'onset_time_sec' in events_df.columns:
            baseline_start_event = events_df[events_df['condition'] == ev_config.BASELINE_MARKER_START_EPRIME]
            if not baseline_start_event.empty:
                processed_data_artifacts['baseline_start_time_sec'] = baseline_start_event['onset_time_sec'].iloc[0]
            baseline_end_event = events_df[events_df['condition'] == ev_config.BASELINE_MARKER_END_EPRIME]
            if not baseline_end_event.empty:
                processed_data_artifacts['baseline_end_time_sec'] = baseline_end_event['onset_time_sec'].iloc[0]

            if processed_data_artifacts['baseline_start_time_sec'] is not None and \
               processed_data_artifacts['baseline_end_time_sec'] is None:
                emotional_stim_after_baseline = events_df[
                    (events_df['condition'].isin(ev_config.EMOTIONAL_CONDITIONS)) & # Use ev_config for conditions
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

    p_logger.info("--- Preprocessing Stage ---")
    if raw_eeg_mne:
        eeg_preprocessor = EEGPreprocessor(p_logger)
        raw_eeg_processed = eeg_preprocessor.process(
            raw_eeg_mne.copy(),
            eeg_filter_band_config=ev_config.EEG_FILTER_BAND,
            ica_n_components_config=ev_config.ICA_N_COMPONENTS,
            ica_random_state_config=ev_config.ICA_RANDOM_STATE,
            ica_accept_labels_config=ev_config.ICA_ACCEPT_LABELS,
            ica_reject_threshold_config=ev_config.ICA_REJECT_THRESHOLD)
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
        fnirs_haemo_processed = fnirs_preprocessor.process(
            fnirs_od_mne.copy(),
            beer_lambert_ppf_config=ev_config.FNIRS_BEER_LAMBERT_PPF,
            short_channel_regression_config=ev_config.FNIRS_SHORT_CHANNEL_REGRESSION,
            motion_correction_method_config=ev_config.FNIRS_MOTION_CORRECTION_METHOD,
            filter_band_config=ev_config.FNIRS_FILTER_BAND)
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
            ecg_data_raw, ecg_sampling_rate, participant_id, preproc_results_dir,
            ecg_rpeak_method_config=ev_config.ECG_RPEAK_METHOD
        )
        if rpeak_times_path:
            processed_data_artifacts['preprocessed_data_paths']['ecg_rpeak_times_csv'] = rpeak_times_path
            processed_data_artifacts['preprocessed_data_paths']['ecg_nn_intervals_csv'] = nn_intervals_path
            processed_data_artifacts['analysis_outputs']['metrics']['ecg_nn_intervals_ms_overall'] = nn_intervals_ms_overall
            processed_data_artifacts['ecg_rpeaks_samples_overall'] = rpeaks_samples_overall

            baseline_start_sec_abs = processed_data_artifacts.get('baseline_start_time_sec')
            baseline_end_sec_abs = processed_data_artifacts.get('baseline_end_time_sec')

            if baseline_start_sec_abs is not None and baseline_end_sec_abs is not None:
                resting_rmssd = analysis_service.calculate_resting_state_rmssd(
                    ecg_data_raw, ecg_sampling_rate, ecg_times_abs,
                    baseline_start_sec_abs, baseline_end_sec_abs,
                    ecg_rpeak_method_config=ev_config.ECG_RPEAK_METHOD
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
        # Assuming EDA_CLEAN_METHOD will be added to EV_config.py
        # If not, EDAPreprocessor should handle a missing config or have a default.
        eda_clean_method = getattr(ev_config, 'EDA_CLEAN_METHOD', 'neurokit') # Default if not in config
        phasic_path, tonic_path, phasic_eda_full_signal, tonic_eda_full_signal = eda_preprocessor.preprocess_eda(
            eda_data_raw, eda_sampling_rate, participant_id, preproc_results_dir,
            eda_clean_method_config=eda_clean_method
        )
        if phasic_path:
            eda_original_sfreq_for_plv = eda_sampling_rate
            processed_data_artifacts['preprocessed_data_paths']['phasic_eda_csv'] = phasic_path
            processed_data_artifacts['preprocessed_data_paths']['tonic_eda_csv'] = tonic_path
            p_logger.info("EDA preprocessed. Phasic and Tonic components available.")
    else:
        p_logger.warning("No EDA data loaded or sampling rate missing. Skipping EDA preprocessing.")

    p_logger.info("--- Analysis Stage ---")
    ref_sfreq_for_events = eeg_sampling_rate if eeg_sampling_rate else (fnirs_sampling_rate if fnirs_sampling_rate else None)
    if 'onset_sample' not in current_events_df.columns and 'onset_time_sec' in current_events_df.columns and ref_sfreq_for_events:
        pass
    elif 'onset_sample' not in current_events_df.columns:
        p_logger.error("Cannot determine 'onset_sample' for events. Missing 'onset_time_sec' or reference sfreq.")
        return processed_data_artifacts

    mne_events_array, event_id_map, trial_id_eprime_map = create_mne_events_from_dataframe(
        current_events_df, ev_config.EMOTIONAL_CONDITIONS, ref_sfreq_for_events, p_logger # Use ev_config for conditions
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
                                    tmin=ev_config.ANALYSIS_EPOCH_TIMES[0], tmax=ev_config.ANALYSIS_EPOCH_TIMES[1],
                                    baseline=ev_config.ANALYSIS_BASELINE_TIMES, preload=True, verbose=False,
                                    picks='eeg', on_missing='warning')
            processed_data_artifacts['mne_objects']['epochs']['eeg'] = epochs_eeg
            p_logger.info(f"EEG data epoched: {len(epochs_eeg)} epochs created across {len(epochs_eeg.event_id)} conditions.")
        except Exception as e:
            p_logger.error(f"Failed to epoch EEG data: {e}", exc_info=True)

    raw_fnirs_proc = processed_data_artifacts['mne_objects']['raw'].get('fnirs_haemo_processed')
    if raw_fnirs_proc:
        try:
            epochs_fnirs = mne.Epochs(raw_fnirs_proc, mne_events_array, event_id=event_id_map,
                                      tmin=ev_config.ANALYSIS_EPOCH_TIMES[0], tmax=ev_config.ANALYSIS_EPOCH_TIMES[1],
                                      baseline=ev_config.ANALYSIS_BASELINE_TIMES, preload=True, verbose=False,
                                      on_missing='warning')
            processed_data_artifacts['mne_objects']['epochs']['fnirs'] = epochs_fnirs
            p_logger.info(f"fNIRS data epoched: {len(epochs_fnirs)} epochs created across {len(epochs_fnirs.event_id)} conditions.")
        except Exception as e:
            p_logger.error(f"Failed to epoch fNIRS data: {e}", exc_info=True)

    p_logger.info("--- WP1: Emotional Modulation of Synchrony ---")
    fnirs_epochs = processed_data_artifacts['mne_objects']['epochs'].get('fnirs')
    eeg_epochs = processed_data_artifacts['mne_objects']['epochs'].get('eeg')

    active_fnirs_roi_names_for_eeg_guidance = []
    if fnirs_epochs and raw_eeg_proc:
        event_id_map_for_glm = processed_data_artifacts['analysis_outputs']['metadata'].get('event_id_map_mne', {})
        fnirs_glm_output = analysis_service.run_fnirs_glm_and_contrasts(
            glm_hrf_model=ev_config.FNIRS_HRF_MODEL,
            glm_contrasts_config=ev_config.FNIRS_CONTRASTS,
            glm_rois_config=ev_config.FNIRS_ROIS,
            glm_activation_p_threshold=ev_config.FNIRS_ACTIVATION_P_THRESHOLD,
            fnirs_epochs=fnirs_epochs, # Pass epochs directly
            participant_id=participant_id,
            analysis_results_dir=analysis_results_dir
        )
        processed_data_artifacts['analysis_outputs']['stats']['fnirs_glm_output'] = fnirs_glm_output
        if fnirs_glm_output and 'contrast_results' in fnirs_glm_output:
            for contrast_name_plot, contrast_df_plot in fnirs_glm_output['contrast_results'].items():
                if contrast_df_plot is not None and not contrast_df_plot.empty:
                    plotting_service.plot_fnirs_contrast_results(participant_id, contrast_name_plot, contrast_df_plot, f"participant_fnirs_contrast_{contrast_name_plot}")
        active_fnirs_roi_names_for_eeg_guidance = fnirs_glm_output.get('active_rois_for_eeg_guidance', [])

        eeg_channels_for_plv_wp1 = select_eeg_channels_by_fnirs_rois(
            raw_eeg_proc.info, active_fnirs_roi_names_for_eeg_guidance,
            ev_config.FNIRS_ROI_TO_EEG_CHANNELS_MAP,
            ev_config.DEFAULT_EEG_CHANNELS_FOR_PLV,
            ev_config.EEG_CHANNEL_SELECTION_STRATEGY_FOR_PLV,
            p_logger
        )
        if not eeg_channels_for_plv_wp1:
            eeg_channels_for_plv_wp1 = [ch for ch in ev_config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in raw_eeg_proc.ch_names]
        p_logger.info(f"Using EEG channels for WP1 PLV: {eeg_channels_for_plv_wp1}")
    elif raw_eeg_proc:
        eeg_channels_for_plv_wp1 = [ch for ch in ev_config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in raw_eeg_proc.ch_names]
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
                rpeaks_samples_overall, ecg_sampling_rate, ev_config.PLV_RESAMPLE_SFREQ_AUTONOMIC
            )
            hrv_sfreq_for_plv = ev_config.PLV_RESAMPLE_SFREQ_AUTONOMIC
            processed_data_artifacts['analysis_outputs']['metadata']['hrv_sfreq_for_plv'] = hrv_sfreq_for_plv

        trial_plv_df_participant = analysis_service.calculate_trial_plv(
            eeg_epochs, eeg_channels_for_plv_wp1,
            continuous_hrv_signal_for_plv, hrv_sfreq_for_plv,
            phasic_eda_full_signal, eda_original_sfreq_for_plv,
            plv_eeg_bands_config=ev_config.PLV_EEG_BANDS,
            participant_id=participant_id,
            raw_eeg_sfreq_for_event_timing=raw_eeg_proc.info['sfreq'],
            trial_id_eprime_map=processed_data_artifacts['analysis_outputs']['metadata'].get('trial_id_eprime_map', {})
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

    p_logger.info("--- WP2: Synchrony and Subjective Arousal ---")
    trial_plv_df_wp1_for_wp2 = processed_data_artifacts['analysis_outputs']['dataframes'].get('trial_plv_wp1')

    if survey_df_per_trial is not None and not survey_df_per_trial.empty and \
       trial_plv_df_wp1_for_wp2 is not None and not trial_plv_df_wp1_for_wp2.empty:
        if 'sam_arousal' in survey_df_per_trial.columns and 'trial_identifier_eprime' in survey_df_per_trial.columns:
            processed_data_artifacts['analysis_outputs']['metadata']['wp2_has_sam_arousal_and_ids'] = True
            p_logger.info(f"WP2: Participant {participant_id} has SAM arousal and trial identifiers for survey data.")
        else:
            p_logger.warning(f"WP2: Participant {participant_id} survey data missing 'sam_arousal' or 'trial_identifier_eprime'.")
        # Calculate EDA features per condition if data is available
        if raw_eeg_proc and phasic_eda_full_signal is not None and eda_original_sfreq_for_plv is not None:
            analysis_service.calculate_eda_features_per_condition(
                raw_eeg_proc, phasic_eda_full_signal, eda_original_sfreq_for_plv,
                ev_config.STIMULUS_DURATION_SECONDS,
                processed_data_artifacts['analysis_outputs']['metrics']
            )
    else:
        p_logger.info("Skipping WP2 prep due to missing survey or PLV data for this participant.")

    p_logger.info("--- WP3: Baseline Vagal Tone and Task-Related Synchrony ---")
    baseline_rmssd = processed_data_artifacts['analysis_outputs']['metrics'].get('baseline_rmssd')
    if baseline_rmssd is not None and not np.isnan(baseline_rmssd):
        p_logger.info(f"Calculated baseline RMSSD: {baseline_rmssd:.2f} ms")
        avg_plv_df_wp1_for_wp3 = processed_data_artifacts['analysis_outputs']['dataframes'].get('avg_plv_wp1')
        if avg_plv_df_wp1_for_wp3 is not None and not avg_plv_df_wp1_for_wp3.empty:
            plv_negative_series = avg_plv_df_wp1_for_wp3[
                (avg_plv_df_wp1_for_wp3['condition'] == 'Negative') &
                (avg_plv_df_wp1_for_wp3['modality_pair'] == 'EEG-HRV') &
                (avg_plv_df_wp1_for_wp3['eeg_band'] == ev_config.PLV_PRIMARY_EEG_BAND_FOR_WP3)
            ]['plv']
            if not plv_negative_series.empty:
                avg_plv_negative = plv_negative_series.mean()
                processed_data_artifacts['analysis_outputs']['metrics']['wp3_avg_plv_negative_specific'] = avg_plv_negative
                p_logger.info(f"WP3: P{participant_id} - Baseline RMSSD: {baseline_rmssd:.2f}, Avg PLV (Negative, {ev_config.PLV_PRIMARY_EEG_BAND_FOR_WP3}, HRV): {avg_plv_negative:.3f}")
    else:
        p_logger.warning(f"WP3: Baseline RMSSD not available or NaN for P{participant_id}.")

    p_logger.info("--- WP4: Frontal Asymmetry and Branch-Specific Synchrony ---")
    if eeg_epochs:
        psd_results, fai_results_per_condition = analysis_service.calculate_psd_and_fai(
            raw_eeg_proc, mne_events_array, event_id_map,
            fai_alpha_band_config=ev_config.FAI_ALPHA_BAND,
            eeg_bands_config_for_beta=ev_config.PLV_EEG_BANDS,
            fai_electrode_pairs_config=ev_config.FAI_ELECTRODE_PAIRS,
            analysis_epoch_tmax_config=ev_config.ANALYSIS_EPOCH_TIMES[1]
        )
        processed_data_artifacts['analysis_outputs']['metrics']['fai_results_per_condition_wp4'] = fai_results_per_condition

        avg_fai_f4f3_list = []
        target_fai_pair_wp4 = f"{ev_config.FAI_ELECTRODE_PAIRS_FOR_WP4[1]}_vs_{ev_config.FAI_ELECTRODE_PAIRS_FOR_WP4[0]}"

        if fai_results_per_condition:
            for cond_name_fai, fai_pairs_data in fai_results_per_condition.items():
                if cond_name_fai in ev_config.EMOTIONAL_CONDITIONS: # Use ev_config for conditions
                    if target_fai_pair_wp4 in fai_pairs_data:
                        avg_fai_f4f3_list.append(fai_pairs_data[target_fai_pair_wp4])

            if avg_fai_f4f3_list:
                processed_data_artifacts['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional'] = np.nanmean(avg_fai_f4f3_list)
                p_logger.info(f"WP4: P{participant_id} - Avg FAI ({target_fai_pair_wp4}) over emotional conditions: {processed_data_artifacts['analysis_outputs']['metrics']['wp4_avg_fai_f4f3_emotional']:.3f}")
    else:
        p_logger.warning(f"WP4: No EEG epochs for P{participant_id}, cannot calculate FAI.")

    p_logger.info("--- Analysis Stage Complete for Participant ---")
    return processed_data_artifacts

def _process_single_participant_task(task_config, ev_config): # Add ev_config as a parameter
    """
    Worker function to process a single participant.
    This function will be executed by each thread.
    It includes logger setup and teardown for the participant.
    Args:
        task_config (dict): A dictionary containing all necessary parameters for this task.
        ev_config (module): The EV_config module, passed from ParallelTaskRunner.
    """
    p_id_raw = task_config['p_id_raw']
    p_id = task_config['p_id']
    data_root_dir = task_config['data_root_dir']
    output_base_dir = task_config['output_base_dir']
    main_logger_name = task_config['main_logger_name']

    main_logger = logging.getLogger(main_logger_name)
    main_logger.info(f"Thread starting for participant: {p_id}")

    participant_output_dirs = create_output_directories(output_base_dir, p_id)
    p_log_manager = ParticipantLogger(participant_output_dirs['base_participant'], p_id, ev_config.LOG_LEVEL)
    participant_logger_instance = p_log_manager.get_logger()

    processed_artifacts = None
    try:
        participant_raw_data_path = os.path.join(data_root_dir, p_id_raw)

        eprime_txt_file = None
        if os.path.exists(participant_raw_data_path):
            for f_name in os.listdir(participant_raw_data_path):
                if p_id in f_name and f_name.lower().endswith('.txt'):
                    eprime_txt_file = os.path.join(participant_raw_data_path, f_name)
                    break

        if not os.path.exists(participant_raw_data_path) and not eprime_txt_file: # Corrected logic
            participant_logger_instance.warning(f"No data directory or E-Prime .txt file found for participant {p_id} in {participant_raw_data_path}. Skipping.")
            return {'participant_id': p_id, 'status': 'no_data_found', 'log_file': p_log_manager.log_file_path}

        # Call process_participant_data with ev_config
        processed_artifacts = process_participant_data(
            p_id,
            participant_raw_data_path,
            eprime_txt_file,
            participant_output_dirs,
            output_base_dir,
            participant_logger_instance,
            ev_config # Pass ev_config here
        )
        if processed_artifacts and isinstance(processed_artifacts, dict):
            processed_artifacts.setdefault('status', 'success')
            processed_artifacts['participant_output_path'] = participant_output_dirs['base_participant']
            processed_artifacts.setdefault('log_file', p_log_manager.log_file_path)

            if processed_artifacts.get('status') == 'success':
                artifact_save_path = os.path.join(participant_output_dirs['base_participant'], f"{p_id}_summary_artifact.pkl")
                try:
                    with open(artifact_save_path, 'wb') as f_pkl:
                        pickle.dump(processed_artifacts, f_pkl)
                    processed_artifacts['artifact_pickle_path'] = artifact_save_path
                    participant_logger_instance.info(f"Saved detailed summary artifact to: {artifact_save_path}")
                except Exception as e_save:
                     participant_logger_instance.error(f"Failed to save artifact pickle for {p_id}: {e_save}", exc_info=True)
                     processed_artifacts['status'] = 'error_save_artifact'
                     processed_artifacts['error_message'] = f"Failed to save artifact: {e_save}"

            if processed_artifacts.get('status') == 'success':
                 participant_logger_instance.info(f"--- Successfully processed participant: {p_id} (in thread) ---")

        elif processed_artifacts is None:
             processed_artifacts = {'participant_id': p_id, 'status': 'error_no_artifact', 'log_file': p_log_manager.log_file_path, 'participant_output_path': participant_output_dirs['base_participant']}
    except Exception as e:
        participant_logger_instance.error(f"--- CRITICAL ERROR processing participant {p_id} (in thread): {e} ---", exc_info=True)
        processed_artifacts = {'participant_id': p_id, 'status': 'error', 'error_message': str(e), 'log_file': p_log_manager.log_file_path, 'participant_output_path': participant_output_dirs['base_participant']}
    finally:
        p_log_manager.close_handlers()
    return processed_artifacts

def main_analyzer_run(data_root_dir, output_base_dir, project_root_dir, participant_ids_to_process=None, main_logger_name="MainAnalyzer"):
    main_logger = logging.getLogger(main_logger_name)
    if not main_logger.handlers:
        logging.basicConfig(level=getattr(logging, EV_config.LOG_LEVEL.upper(), logging.INFO), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        main_logger = logging.getLogger(main_logger_name)

    main_logger.info("--- EmotiView Data Analyzer Starting Run ---")
    main_logger.info(f"Data root directory: {data_root_dir}")
    main_logger.info(f"Output base directory: {output_base_dir}")

    summary_csv_path = os.path.join(output_base_dir, EV_config.PROCESSING_SUMMARY_FILENAME)
    existing_summary_df = pd.DataFrame()
    if os.path.exists(summary_csv_path):
        try:
            existing_summary_df = pd.read_csv(summary_csv_path)
            main_logger.info(f"Loaded existing processing summary from {summary_csv_path}")
        except Exception as e:
            main_logger.warning(f"Could not load existing processing summary from {summary_csv_path}: {e}. Starting with an empty summary.")

    if not participant_ids_to_process:
        try:
            all_p_ids = [d for d in os.listdir(data_root_dir)
                               if os.path.isdir(os.path.join(data_root_dir, d)) and
                               (d.startswith("P") or d.startswith("EV_P"))]
        except FileNotFoundError:
            main_logger.error(f"Data root directory not found: {data_root_dir}")
            return []

        if not all_p_ids:
            main_logger.error(f"No participant subdirectories matching pattern found in {data_root_dir}.")
            return []
        main_logger.info(f"Found {len(all_p_ids)} potential participants: {all_p_ids}")
        participant_ids_to_process = all_p_ids

    main_logger.info(f"Participants to process in this run: {participant_ids_to_process}")

    if not participant_ids_to_process:
        main_logger.info("No participants found or specified to process in this run.")
        return []

    max_workers = getattr(EV_config, 'MAX_PARALLEL_PARTICIPANTS', os.cpu_count() or 4)
    main_logger.info(f"Using up to {max_workers} threads for parallel participant processing.")

    participant_task_configs = []
    processed_participant_ids_in_run = set()
    for p_id_raw in participant_ids_to_process:
        p_id = p_id_raw
        if p_id in processed_participant_ids_in_run:
            main_logger.warning(f"Participant {p_id} was already scheduled in this run. Skipping duplicate.")
            continue
        processed_participant_ids_in_run.add(p_id)
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
        task_function_kwargs={'ev_config': EV_config},
        main_logger_name=main_logger.name
    )
    all_participants_summary_artifacts = runner.run()

    main_logger.info("--- EmotiView Data Analyzer Finished Individual Participant Processing for this batch ---")
    main_logger.info("--- Attempting Git Sync for Successful Participants ---")

    for artifact in all_participants_summary_artifacts:
        if isinstance(artifact, dict) and artifact.get('status') == 'success':
            p_id_processed = artifact.get('participant_id')
            output_path_processed = artifact.get('participant_output_path')
            if p_id_processed and output_path_processed:
                main_logger.info(f"Attempting Git sync for successfully processed participant: {p_id_processed}")
                commit_msg = f"feat: Processed data for participant {p_id_processed}"
                sync_successful = commit_and_sync_changes(
                    repository_path=project_root_dir,
                    paths_to_add=[output_path_processed],
                    commit_message=commit_msg,
                    logger=main_logger
                )
                main_logger.info(f"Git sync for {p_id_processed} {'succeeded' if sync_successful else 'failed or was skipped'}.")

    summary_list = []
    for r_idx, r_val in enumerate(all_participants_summary_artifacts):
        if isinstance(r_val, dict):
            summary_list.append({
                'participant_id': r_val.get('participant_id', f'unknown_participant_{r_idx}'),
                'status': r_val.get('status', 'error' if 'error_message' in r_val else 'processed'),
                'log_file': r_val.get('log_file'),
                'artifact_pickle_path': r_val.get('artifact_pickle_path')
            })
        else:
            summary_list.append({'participant_id': f'unknown_participant_{r_idx}', 'status': 'unknown_format', 'log_file': None})
    summary_df = pd.DataFrame(summary_list)

    if not existing_summary_df.empty and 'participant_id' in existing_summary_df.columns:
        combined_summary_df = pd.concat([existing_summary_df, summary_df], ignore_index=True)
        combined_summary_df = combined_summary_df.drop_duplicates(subset=['participant_id'], keep='last')
        main_logger.info(f"Merged current batch summary ({len(summary_df)} entries) with existing summary ({len(existing_summary_df)} entries). Total unique participants: {len(combined_summary_df)}")
    else:
        combined_summary_df = summary_df
        main_logger.info(f"No existing summary found or it was invalid. Saving current batch summary ({len(summary_df)} entries).")

    combined_summary_df.to_csv(summary_csv_path, index=False)
    main_logger.info(f"Updated processing summary saved to {summary_csv_path}")

    full_summary_df_for_group_analysis = pd.DataFrame()
    if os.path.exists(summary_csv_path):
        try:
            full_summary_df_for_group_analysis = pd.read_csv(summary_csv_path)
        except Exception as e:
            main_logger.error(f"Could not load full processing summary from {summary_csv_path} for group analysis: {e}")

    all_successful_artifacts_for_group = []
    if not full_summary_df_for_group_analysis.empty:
        successful_participants_info = full_summary_df_for_group_analysis[full_summary_df_for_group_analysis['status'] == 'success']
        main_logger.info(f"Found {len(successful_participants_info)} successful participants in total for cumulative group analysis.")

        for _, row in successful_participants_info.iterrows():
            p_id_group = row['participant_id']
            artifact_pkl_path = row.get('artifact_pickle_path')

            if artifact_pkl_path and os.path.exists(artifact_pkl_path):
                try:
                    with open(artifact_pkl_path, 'rb') as f_pkl:
                        loaded_artifact = pickle.load(f_pkl)
                        loaded_artifact['participant_id'] = p_id_group
                        loaded_artifact['artifact_pickle_path'] = artifact_pkl_path
                        all_successful_artifacts_for_group.append(loaded_artifact)
                except Exception as e_load:
                    main_logger.error(f"Failed to load artifact pickle for {p_id_group} from {artifact_pkl_path}: {e_load}")
            else:
                 fallback_pkl_path = os.path.join(output_base_dir, p_id_group, f"{p_id_group}_summary_artifact.pkl")
                 if os.path.exists(fallback_pkl_path):
                     try:
                         with open(fallback_pkl_path, 'rb') as f_pkl:
                             loaded_artifact = pickle.load(f_pkl)
                             loaded_artifact['participant_id'] = p_id_group
                             loaded_artifact['artifact_pickle_path'] = fallback_pkl_path
                             all_successful_artifacts_for_group.append(loaded_artifact)
                             main_logger.warning(f"Used fallback path for artifact pickle for {p_id_group}: {fallback_pkl_path}")
                     except Exception as e_load_fallback:
                         main_logger.error(f"Failed to load artifact pickle for {p_id_group} from fallback path {fallback_pkl_path}: {e_load_fallback}")
                 else:
                    main_logger.warning(f"Artifact pickle file not found for successful participant {p_id_group} at expected path ({artifact_pkl_path if artifact_pkl_path else 'N/A'}) or fallback path. Skipping for group analysis.")

    if all_successful_artifacts_for_group:
        main_logger.info(f"Running group analysis with {len(all_successful_artifacts_for_group)} successful participants.")
        group_orchestrator = GroupAnalyzer(
            main_logger, output_base_dir,
            emotional_conditions_config=ev_config.EMOTIONAL_CONDITIONS,
            plv_eeg_bands_config=ev_config.PLV_EEG_BANDS,
            plv_primary_eeg_band_for_wp3_config=ev_config.PLV_PRIMARY_EEG_BAND_FOR_WP3,
            figure_format_config=ev_config.REPORTING_FIGURE_FORMAT, # Pass plotting format
            figure_dpi_config=ev_config.REPORTING_DPI)              # Pass plotting DPI
        group_orchestrator.run_group_analysis(all_successful_artifacts_for_group)
    else:
        main_logger.warning("No successful participant artifacts (neither current nor historical) available for group analysis.")

    main_logger.info("--- EmotiView Data Analyzer Finished Run ---")
    return all_participants_summary_artifacts

def get_processed_participants(output_base_dir):
    if not os.path.isdir(output_base_dir):
        return set()
    processed = set()
    for item in os.listdir(output_base_dir):
        if os.path.isdir(os.path.join(output_base_dir, item)):
            processed.add(item)
    return processed

def find_new_participants(data_root_dir, output_base_dir, logger):
    try:
        potential_participants = set([
            p_id for p_id in os.listdir(data_root_dir)
            if os.path.isdir(os.path.join(data_root_dir, p_id)) and (p_id.startswith("P") or p_id.startswith("EV_P"))
        ])
    except FileNotFoundError:
        logger.error(f"Data root directory not found: {data_root_dir}")
        return []

    processed_participants = get_processed_participants(output_base_dir)
    new_participants = sorted(list(potential_participants - processed_participants))

    if new_participants:
        logger.info(f"Found new participants to process: {new_participants}")
    else:
        logger.info("No new participants found to process.")
    return new_participants

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) # Adjusted to go up two levels

    parser = argparse.ArgumentParser(description="EmotiView Data Analyzer")
    parser.add_argument(
        "--mode",
        choices=['automated', 'testrun'],
        required=True,
        help="Operation mode: 'automated' to run indefinitely and wait for new participants, 'testrun' to process specified participants once."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(project_root, EV_config.DEFAULT_INPUT_DATA_DIR),
        help="Root directory containing participant data."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, EV_config.DEFAULT_OUTPUT_RESULTS_DIR),
        help="Base directory to save processed results."
    )
    parser.add_argument(
        "--project-dir",
        default=project_root,
        help="Root directory of the project (for Git operations)."
    )
    parser.add_argument(
        "--participant-ids",
        nargs='*',
        default=None,
        help="Required for 'testrun' mode: Specific participant IDs to process."
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=300,
        help="Interval in seconds to check for new participants in 'automated' mode."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main_log_file = os.path.join(args.output_dir, EV_config.ANALYZER_MAIN_LOG_FILENAME)

    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, EV_config.LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(main_log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger("MainAnalyzer")

    main_logger.info(f"Starting EmotiView Analyzer with arguments: {args}")

    if args.mode == 'automated':
        main_logger.info(f"Running in 'automated' mode. Checking for new participants every {args.check_interval} seconds.")
        try:
            while True:
                new_p_ids = find_new_participants(args.data_dir, args.output_dir, main_logger)
                if new_p_ids:
                    main_logger.info(f"Automated run: Processing new participants - {new_p_ids}")
                    main_analyzer_run(args.data_dir, args.output_dir, args.project_dir, participant_ids_to_process=new_p_ids, main_logger_name="MainAnalyzer")
                main_logger.info(f"Sleeping for {args.check_interval} seconds...")
                time.sleep(args.check_interval)
        except KeyboardInterrupt:
            main_logger.info("Automated mode interrupted by user. Exiting.")
        except Exception as e:
            main_logger.critical(f"Unhandled exception in automated mode: {e}", exc_info=True)

    elif args.mode == 'testrun':
        main_logger.info("Running in 'testrun' mode.")
        if not args.participant_ids:
            main_logger.error("For 'testrun' mode, --participant-ids must be specified.")
            parser.print_help()
        else:
            main_logger.info(f"Test run: Processing specified participants - {args.participant_ids}")
            main_analyzer_run(args.data_dir, args.output_dir, args.project_dir, participant_ids_to_process=args.participant_ids, main_logger_name="MainAnalyzer")

    main_logger.info("EmotiView Analyzer finished its current task.")
