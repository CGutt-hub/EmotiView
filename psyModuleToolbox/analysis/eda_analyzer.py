import mne
import neurokit2 as nk
import numpy as np
import pandas as pd

class EDAAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDAAnalyzer initialized.")

    def calculate_eda_features_per_condition(self, raw_eeg_with_events, # Used for event timings
                                             phasic_eda_full_signal_array, # Full phasic EDA signal as numpy array
                                             eda_original_sfreq, # Original sampling rate of EDA
                                             stimulus_duration_seconds, # Duration of the stimulus/epoch
                                             analysis_metrics):
        """
        Calculates EDA features (e.g., mean phasic amplitude, SCR count) per condition.
        Args:
            raw_eeg_with_events (mne.io.Raw): Raw EEG object containing event annotations.
            phasic_eda_full_signal_array (np.ndarray): The full preprocessed phasic EDA signal.
            eda_original_sfreq (float): Original sampling rate of the EDA signal.
            stimulus_duration_seconds (float): The duration of the stimulus epoch in seconds.
            analysis_metrics (dict): Dictionary to store results.
        """
        if raw_eeg_with_events is None:
            self.logger.warning("EDAAnalyzer - No EEG data with events provided. Skipping EDA feature extraction.")
            return
        if phasic_eda_full_signal_array is None or eda_original_sfreq is None:
            self.logger.warning("EDAAnalyzer - Phasic EDA signal array or original sampling rate not provided. Skipping.")
            return
        if stimulus_duration_seconds is None or stimulus_duration_seconds <= 0:
            self.logger.warning("EDAAnalyzer - Invalid stimulus_duration_seconds provided. Skipping.")
            return

        self.logger.info("EDAAnalyzer - Calculating EDA features per condition.")
        try:
            events, event_id_map = mne.events_from_annotations(raw_eeg_with_events, verbose=False)
            
            if not events.size:
                self.logger.warning("EDAAnalyzer - No events found in EEG data. Cannot extract condition-specific EDA features.")
                return

            for condition_name, event_code in event_id_map.items():
                self.logger.debug(f"EDAAnalyzer - Processing condition: {condition_name}")
                condition_event_indices = events[events[:, 2] == event_code, 0] # Get sample onsets

                condition_scr_counts = []
                condition_phasic_means = []

                for onset_sample in condition_event_indices:
                    start_time_sec = onset_sample / raw_eeg_with_events.info['sfreq']
                    end_time_sec = start_time_sec + stimulus_duration_seconds

                    # Extract corresponding segment from phasic_eda_full
                    start_idx_eda = int(start_time_sec * eda_original_sfreq)
                    end_idx_eda = int(end_time_sec * eda_original_sfreq)

                    if start_idx_eda < end_idx_eda and end_idx_eda <= len(phasic_eda_full_signal_array):
                        eda_epoch = phasic_eda_full_signal_array[start_idx_eda:end_idx_eda]
                        if len(eda_epoch) > 0:
                            # Example features using NeuroKit2
                            # For mean phasic, directly average the phasic component
                            condition_phasic_means.append(np.mean(eda_epoch))
                            
                            # Calculate SCR count for the epoch
                            # Note: nk.eda_peaks is typically run on the raw EDA or cleaned EDA, not just phasic.
                            # For this example, we'll use it on the phasic component, but this might need refinement
                            # depending on the quality of the phasic signal and desired sensitivity.
                            # A minimum amplitude threshold for SCRs is also common.
                            signals_peaks, info_peaks = nk.eda_peaks(eda_epoch, sampling_rate=eda_original_sfreq, method="neurokit", amplitude_min=0.01) # amplitude_min is example
                            scr_count_epoch = len(info_peaks["SCR_Onsets"])
                            condition_scr_counts.append(scr_count_epoch)
                 
                if condition_phasic_means:
                    analysis_metrics[f'eda_phasic_mean_{condition_name}'] = np.nanmean(condition_phasic_means)
                if condition_scr_counts:
                    analysis_metrics[f'eda_scr_count_{condition_name}'] = np.nanmean(condition_scr_counts) # Average SCR count per trial
            
            self.logger.info("EDAAnalyzer - Condition-specific EDA feature calculation completed.")
        except Exception as e:
            self.logger.error(f"EDAAnalyzer - Error calculating EDA features: {e}", exc_info=True)

    # Placeholder method if you want overall EDA metrics (e.g., mean SCL across recording)
    # def calculate_overall_eda_metrics(self, tonic_eda_signal_path, eda_original_sfreq, analysis_metrics):
    #     pass