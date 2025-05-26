import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert
# from scipy.interpolate import interp1d # Not directly used here anymore
from ..orchestrators import config 

# Utility function, can stay here or move to utils.py if generally useful
def _calculate_plv_segment(phase_sig1, phase_sig2, logger_obj): 
    min_len = min(len(phase_sig1), len(phase_sig2))
    if min_len == 0: 
        logger_obj.debug("PLV calc: Zero length signal provided to calculate_plv_segment.")
        return np.nan
    phase_diff = phase_sig1[:min_len] - phase_sig2[:min_len]
    return np.abs(np.mean(np.exp(1j * phase_diff)))

class ConnectivityAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ConnectivityAnalyzer initialized.")

    def _get_autonomic_segment_for_epoch(self, eeg_epoch_start_time_abs, eeg_epoch_duration_sec,
                                         autonomic_signal_full, autonomic_signal_sfreq_orig,
                                         target_eeg_epoch_len_samples, target_eeg_sfreq,
                                         autonomic_signal_name):
        """
        Extracts, resamples, and pads/truncates an autonomic signal segment 
        to match an EEG epoch's timing and length.
        """
        if autonomic_signal_full is None or autonomic_signal_sfreq_orig is None:
            self.logger.debug(f"ConnectivityAnalyzer - Autonomic signal ({autonomic_signal_name}) or its sfreq is None.")
            return None

        start_sample_autonomic_orig = int(eeg_epoch_start_time_abs * autonomic_signal_sfreq_orig)
        # Ensure end sample does not exceed signal length
        end_sample_autonomic_orig = min(
            int((eeg_epoch_start_time_abs + eeg_epoch_duration_sec) * autonomic_signal_sfreq_orig),
            len(autonomic_signal_full) 
        )


        if start_sample_autonomic_orig < 0 or start_sample_autonomic_orig >= end_sample_autonomic_orig:
            self.logger.debug(f"ConnectivityAnalyzer - Autonomic segment indices out of bounds or invalid for {autonomic_signal_name}. Epoch Start: {eeg_epoch_start_time_abs:.2f}s, Duration: {eeg_epoch_duration_sec:.2f}s. Indices: {start_sample_autonomic_orig}-{end_sample_autonomic_orig}, SigLen: {len(autonomic_signal_full)}")
            return None
        
        autonomic_epoch_raw = autonomic_signal_full[start_sample_autonomic_orig:end_sample_autonomic_orig]
        if autonomic_epoch_raw.size == 0:
            self.logger.debug(f"ConnectivityAnalyzer - Extracted autonomic segment for {autonomic_signal_name} is empty.")
            return None

        # Resample this specific epoch to the EEG sampling frequency
        # MNE resample needs float64
        resampled_autonomic_epoch = mne.filter.resample(autonomic_epoch_raw.astype(np.float64), 
                                                        up=target_eeg_sfreq, 
                                                        down=autonomic_signal_sfreq_orig, 
                                                        npad='auto', verbose=False)
        
        # Pad or truncate to match the EEG epoch length
        if len(resampled_autonomic_epoch) > target_eeg_epoch_len_samples:
            autonomic_segment_final = resampled_autonomic_epoch[:target_eeg_epoch_len_samples]
        elif len(resampled_autonomic_epoch) < target_eeg_epoch_len_samples:
            pad_width = target_eeg_epoch_len_samples - len(resampled_autonomic_epoch)
            autonomic_segment_final = np.pad(resampled_autonomic_epoch, (0, pad_width), 'edge')
        else:
            autonomic_segment_final = resampled_autonomic_epoch
        
        return autonomic_segment_final


    def calculate_trial_plv(self, eeg_epochs, eeg_channels_for_plv, 
                            continuous_hrv_signal, hrv_sfreq,
                            phasic_eda_signal, eda_sfreq,
                            participant_id, raw_eeg_sfreq_for_event_timing,
                            trial_id_eprime_map=None): # Add map, default to None
        """
        Calculates trial-wise PLV between specified EEG channels (averaged) and autonomic signals.

        Args:
            eeg_epochs (mne.Epochs): Epoched EEG data.
            eeg_channels_for_plv (list): List of EEG channel names to use for PLV.
            continuous_hrv_signal (np.ndarray): Full continuous HRV signal (e.g., interpolated NNIs).
            hrv_sfreq (float): Sampling frequency of the continuous_hrv_signal.
            phasic_eda_signal (np.ndarray): Full continuous phasic EDA signal.
            eda_sfreq (float): Sampling frequency of the phasic_eda_signal.
            participant_id (str): Participant ID.
            raw_eeg_sfreq_for_event_timing (float): Original sampling rate of the raw EEG from which events were derived.
                                                   Used to convert epoch event samples to absolute time.
            trial_id_eprime_map (dict, optional): Map from numeric e-prime trial ID to string identifier.
        Returns:
            pd.DataFrame: DataFrame with trial-wise PLV results.
        """
        if eeg_epochs is None or not eeg_channels_for_plv:
            self.logger.warning("ConnectivityAnalyzer - EEG epochs or channels for PLV not provided. Skipping PLV.")
            return pd.DataFrame()

        self.logger.info(f"ConnectivityAnalyzer - Starting trial-wise PLV for P:{participant_id}, Channels: {eeg_channels_for_plv}")
        
        all_trial_plv_results = []
        eeg_epoch_sfreq = eeg_epochs.info['sfreq']
    
        if trial_id_eprime_map is None: # Ensure it's a dict for safe lookup
            trial_id_eprime_map = {}

        for i, epoch in enumerate(eeg_epochs): # Iterate through each trial
            # MNE Epochs object has an event_id attribute which is a dict mapping condition names to int codes.
            # epoch.event_id here will be the integer code for the condition of this specific epoch.
            # We need to map this code back to the condition name.
            # The eeg_epochs.event_id is the map like {'Positive':1, 'Negative':2}
            # epoch.events[0,2] is the integer code for this epoch's condition
            condition_code = epoch.events[0,2]
            condition_name = "UnknownCondition"
            for name, code in eeg_epochs.event_id.items():
                if code == condition_code:
                    condition_name = name
                    break
            
            eeg_trial_data_multichannel = epoch.get_data(picks=eeg_channels_for_plv)
            if eeg_trial_data_multichannel.size == 0:
                self.logger.warning(f"ConnectivityAnalyzer - Empty EEG data for trial {i}, cond {condition_name}. Skipping.")
                continue
            
            eeg_trial_data_avg = eeg_trial_data_multichannel.mean(axis=0).ravel() # Average selected channels
            target_eeg_epoch_len_samples = len(eeg_trial_data_avg)

            # Determine absolute start and end times of the EEG epoch
            event_sample_in_raw = epoch.events[0,0] 
            epoch_tmin_from_event = epoch.tmin
            
            trial_identifier_eprime_str = "N/A_EPRIME_ID"
            # Check if the 4th column (index 3) exists in epoch.events
            if epoch.events.shape[1] > 3: 
                eprime_trial_id_numeric = epoch.events[0, 3] 
                # Find key by value in the map
                trial_identifier_eprime_str = next((k for k, v in trial_id_eprime_map.items() if v == eprime_trial_id_numeric), f"UNKNOWN_EPRIME_ID_NUMERIC_{eprime_trial_id_numeric}")
            else:
                self.logger.warning(f"ConnectivityAnalyzer - E-Prime trial ID (4th event column) not found in epoch event info for P:{participant_id}, trial_idx_in_epoch_obj {i}, cond {condition_name}.")


            trial_start_time_abs = (event_sample_in_raw / raw_eeg_sfreq_for_event_timing) + epoch_tmin_from_event
            trial_duration_sec = target_eeg_epoch_len_samples / eeg_epoch_sfreq

            # --- EEG-HRV PLV ---
            if continuous_hrv_signal is not None and hrv_sfreq is not None:
                hrv_segment_for_plv = self._get_autonomic_segment_for_epoch(
                    trial_start_time_abs, trial_duration_sec,
                    continuous_hrv_signal, hrv_sfreq,
                    target_eeg_epoch_len_samples, eeg_epoch_sfreq, "HRV"
                )
                if hrv_segment_for_plv is not None:
                    phase_hrv_epoch = np.angle(hilbert(hrv_segment_for_plv - np.mean(hrv_segment_for_plv)))
                    for band_name, band_freqs in config.PLV_EEG_BANDS.items():
                        eeg_filtered_band = mne.filter.filter_data(eeg_trial_data_avg.astype(np.float64), eeg_epoch_sfreq, 
                                                                   l_freq=band_freqs[0], h_freq=band_freqs[1], 
                                                                   verbose=False, fir_design='firwin')
                        phase_eeg_epoch_band = np.angle(hilbert(eeg_filtered_band))
                        plv_val = _calculate_plv_segment(phase_eeg_epoch_band, phase_hrv_epoch, self.logger)
                        if not np.isnan(plv_val):
                            all_trial_plv_results.append({ 
                                'participant_id': participant_id, 'condition': condition_name, 
                                'trial_index_in_condition_epoch': i, # Index within this specific condition's epochs
                                'trial_identifier_eprime': trial_identifier_eprime_str,
                                'modality_pair': 'EEG-HRV', 'eeg_band': band_name, 'plv': plv_val
                            })
            
            # --- EEG-EDA PLV ---
            if phasic_eda_signal is not None and eda_sfreq is not None:
                eda_segment_for_plv = self._get_autonomic_segment_for_epoch(
                    trial_start_time_abs, trial_duration_sec,
                    phasic_eda_signal, eda_sfreq,
                    target_eeg_epoch_len_samples, eeg_epoch_sfreq, "EDA"
                )
                if eda_segment_for_plv is not None:
                    phase_eda_epoch = np.angle(hilbert(eda_segment_for_plv - np.mean(eda_segment_for_plv)))
                    for band_name, band_freqs in config.PLV_EEG_BANDS.items():
                        eeg_filtered_band = mne.filter.filter_data(eeg_trial_data_avg.astype(np.float64), eeg_epoch_sfreq,
                                                                   l_freq=band_freqs[0], h_freq=band_freqs[1],
                                                                   verbose=False, fir_design='firwin')
                        phase_eeg_epoch_band = np.angle(hilbert(eeg_filtered_band))
                        plv_val = _calculate_plv_segment(phase_eeg_epoch_band, phase_eda_epoch, self.logger)
                        if not np.isnan(plv_val): 
                            all_trial_plv_results.append({ 
                                'participant_id': participant_id, 'condition': condition_name, 
                                'trial_index_in_condition_epoch': i,
                                'trial_identifier_eprime': trial_identifier_eprime_str,
                                'modality_pair': 'EEG-EDA', 'eeg_band': band_name, 'plv': plv_val
                            })
        
        self.logger.info(f"ConnectivityAnalyzer - Trial-wise PLV calculation completed for P:{participant_id}. Found {len(all_trial_plv_results)} PLV values.")
        return pd.DataFrame(all_trial_plv_results)