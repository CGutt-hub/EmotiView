import mne
import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import interp1d # If EDA needs resampling here
from ... import config # Relative import
from ..utils import calculate_plv # Assuming calculate_plv is moved to utils or kept here

class ConnectivityAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ConnectivityAnalyzer initialized.")

    def _get_autonomic_phase_segment(self, eeg_epoch_start_time, eeg_epoch_end_time,
                                     autonomic_phase_signal, autonomic_time_vector):
        """Helper to extract aligned autonomic phase segment."""
        if autonomic_phase_signal is None or autonomic_time_vector is None or not len(autonomic_time_vector):
            return None
        start_idx = np.argmin(np.abs(autonomic_time_vector - eeg_epoch_start_time))
        end_idx = np.argmin(np.abs(autonomic_time_vector - eeg_epoch_end_time))
        if start_idx < end_idx and end_idx <= len(autonomic_phase_signal):
            return autonomic_phase_signal[start_idx:end_idx]
        self.logger.debug(f"ConnectivityAnalyzer - Could not get valid autonomic segment for {eeg_epoch_start_time}-{eeg_epoch_end_time}")
        return None

    def calculate_all_plv(self, raw_eeg, # Preprocessed EEG Raw object
                          phase_hrv, target_time_hrv, # From HRVAnalyzer
                          phasic_eda_signal, eda_sampling_rate, # From EDAPreprocessor output & config/loaded
                          analysis_metrics,
                          active_fnirs_rois_per_condition=None): # New parameter
        """
        Calculates PLV between EEG bands (alpha, beta) and autonomic signals (HRV, EDA).
        Args:
            raw_eeg (mne.io.Raw): Preprocessed EEG data.
            phase_hrv (np.ndarray): Phase of the continuous HRV signal.
            target_time_hrv (np.ndarray): Time vector for phase_hrv.
            phasic_eda_signal (np.ndarray): Phasic EDA signal (needs resampling and phase extraction).
            eda_sampling_rate (float): Original sampling rate of EDA before resampling for PLV.
            analysis_metrics (dict): Dictionary to store results.
            active_fnirs_rois_per_condition (dict): Dict mapping condition names to lists of active fNIRS ROI names.
        """
        if raw_eeg is None:
            self.logger.warning("ConnectivityAnalyzer - No EEG data provided. Skipping PLV.")
            return

        self.logger.info("ConnectivityAnalyzer - Starting PLV calculations.")
        try:
            eeg_sfreq = raw_eeg.info['sfreq']
            events_eeg, event_id_eeg = mne.events_from_annotations(raw_eeg, verbose=False)
            if not events_eeg.size:
                self.logger.warning("ConnectivityAnalyzer - No events in EEG data. Skipping PLV.")
                return

            # Prepare EDA phase signal (resample and get phase)
            phase_eda, target_time_eda = None, None
            if phasic_eda_signal is not None and eda_sampling_rate is not None:
                original_eda_time = np.arange(len(phasic_eda_signal)) / eda_sampling_rate
                if len(original_eda_time) > 1:
                    interp_func_eda = interp1d(original_eda_time, phasic_eda_signal, kind='linear', fill_value="extrapolate")
                    # Align with EEG time if possible
                    t_start = max(original_eda_time[0], raw_eeg.times[0])
                    t_end = min(original_eda_time[-1], raw_eeg.times[-1])
                    target_time_eda = np.arange(t_start, t_end, 1.0 / config.AUTONOMIC_RESAMPLE_SFREQ)
                    if len(target_time_eda) > 0:
                        resampled_phasic_eda = interp_func_eda(target_time_eda)
                        phase_eda = np.angle(hilbert(resampled_phasic_eda - np.mean(resampled_phasic_eda)))
                        self.logger.info(f"ConnectivityAnalyzer - EDA phase signal generated (length: {len(phase_eda)}).")

            # Determine conditions from event_id_eeg keys
            conditions = list(event_id_eeg.keys())
            self.logger.info(f"ConnectivityAnalyzer - Processing PLV for conditions: {conditions}")

            for cond_name in conditions:
                if cond_name not in event_id_eeg: continue
                
                cond_event_onsets_samples = events_eeg[events_eeg[:, 2] == event_id_eeg[cond_name], 0]
                if not cond_event_onsets_samples.size: continue

                # Determine EEG channels for PLV for this condition
                current_plv_eeg_channels_map = {} # Will map ROI name or channel name to its EEG channels
                use_fnirs_guided_channels = False
                if active_fnirs_rois_per_condition and cond_name in active_fnirs_rois_per_condition:
                    active_rois = active_fnirs_rois_per_condition[cond_name]
                    if active_rois:
                        self.logger.info(f"ConnectivityAnalyzer - Using fNIRS-guided EEG channels for condition '{cond_name}' based on active ROIs: {active_rois}")
                        for roi_name in active_rois:
                            if roi_name in config.FNIRS_TO_EEG_ROI_MAP:
                                eeg_chs_for_roi = [ch for ch in config.FNIRS_TO_EEG_ROI_MAP[roi_name] if ch in raw_eeg.ch_names]
                                if eeg_chs_for_roi:
                                    current_plv_eeg_channels_map[roi_name] = eeg_chs_for_roi
                                    use_fnirs_guided_channels = True
                        if not use_fnirs_guided_channels:
                             self.logger.warning(f"ConnectivityAnalyzer - fNIRS guidance for '{cond_name}' yielded no usable EEG channels. Falling back to default.")
                    else:
                        self.logger.info(f"ConnectivityAnalyzer - No fNIRS ROIs active for condition '{cond_name}'. Falling back to default EEG channels.")
                else:
                    self.logger.info(f"ConnectivityAnalyzer - No fNIRS guidance available for condition '{cond_name}'. Using default EEG channels.")

                if not use_fnirs_guided_channels or not current_plv_eeg_channels_map: # Fallback
                    default_eeg_chs = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_PLV if ch in raw_eeg.ch_names]
                    if not default_eeg_chs:
                        self.logger.warning(f"ConnectivityAnalyzer - No default PLV EEG channels found in data for condition '{cond_name}'. Skipping PLV for this condition.")
                        continue
                    current_plv_eeg_channels_map = {ch_name: [ch_name] for ch_name in default_eeg_chs} # Each channel is its own "ROI"
                    self.logger.info(f"ConnectivityAnalyzer - Using default EEG channels for PLV for condition '{cond_name}': {default_eeg_chs}")

                # Get all unique EEG channels needed across the selected ROIs/channels
                all_unique_eeg_picks_for_cond = sorted(list(set(ch for ch_list in current_plv_eeg_channels_map.values() for ch in ch_list)))
                if not all_unique_eeg_picks_for_cond:
                    self.logger.warning(f"ConnectivityAnalyzer - No EEG channels selected for PLV for condition '{cond_name}'. Skipping.")
                    continue
                
                # Prepare EEG phase signals for the selected channels
                eeg_alpha_filtered = raw_eeg.copy().filter(8, 13, picks=all_unique_eeg_picks_for_cond, verbose=False).get_data()
                eeg_beta_filtered = raw_eeg.copy().filter(13, 30, picks=all_unique_eeg_picks_for_cond, verbose=False).get_data()
                phase_eeg_alpha_allchans_picked = np.angle(hilbert(eeg_alpha_filtered))
                phase_eeg_beta_allchans_picked = np.angle(hilbert(eeg_beta_filtered))

                for group_name, eeg_channels_in_group in current_plv_eeg_channels_map.items(): # group_name is ROI or single EEG channel
                    plvs_alpha_hrv_group, plvs_beta_hrv_group = [], []
                    plvs_alpha_eda_group, plvs_beta_eda_group = [], []

                    for eeg_ch_name_in_group in eeg_channels_in_group:
                        eeg_ch_idx_in_picked = all_unique_eeg_picks_for_cond.index(eeg_ch_name_in_group)
                        
                        for onset_samp in cond_event_onsets_samples:
                            start_time = (onset_samp / eeg_sfreq) + config.PLV_EPOCH_TMIN_RELATIVE_TO_ONSET
                            end_time = (onset_samp / eeg_sfreq) + config.PLV_EPOCH_TMAX_RELATIVE_TO_ONSET
                            eeg_start_idx = int(start_time * eeg_sfreq)
                            eeg_end_idx = int(end_time * eeg_sfreq)
                            if eeg_start_idx < 0 or eeg_end_idx > phase_eeg_alpha_allchans_picked.shape[1]: continue

                            eeg_phase_alpha_epoch = phase_eeg_alpha_allchans_picked[eeg_ch_idx_in_picked, eeg_start_idx:eeg_end_idx]
                            eeg_phase_beta_epoch = phase_eeg_beta_allchans_picked[eeg_ch_idx_in_picked, eeg_start_idx:eeg_end_idx]

                            hrv_phase_segment = self._get_autonomic_phase_segment(start_time, end_time, phase_hrv, target_time_hrv)
                            if hrv_phase_segment is not None:
                                plvs_alpha_hrv_group.append(calculate_plv(eeg_phase_alpha_epoch, hrv_phase_segment, self.logger))
                                plvs_beta_hrv_group.append(calculate_plv(eeg_phase_beta_epoch, hrv_phase_segment, self.logger))
                            
                            eda_phase_segment = self._get_autonomic_phase_segment(start_time, end_time, phase_eda, target_time_eda)
                            if eda_phase_segment is not None:
                                plvs_alpha_eda_group.append(calculate_plv(eeg_phase_alpha_epoch, eda_phase_segment, self.logger))
                                plvs_beta_eda_group.append(calculate_plv(eeg_phase_beta_epoch, eda_phase_segment, self.logger))

                    # Average PLV for the group (ROI or single channel)
                    if plvs_alpha_hrv_group: analysis_metrics[f'plv_avg_alpha_{group_name}_hrv_{cond_name}'] = np.nanmean(plvs_alpha_hrv_group)
                    if plvs_beta_hrv_group: analysis_metrics[f'plv_avg_beta_{group_name}_hrv_{cond_name}'] = np.nanmean(plvs_beta_hrv_group)
                    if plvs_alpha_eda_group: analysis_metrics[f'plv_avg_alpha_{group_name}_eda_{cond_name}'] = np.nanmean(plvs_alpha_eda_group)
                    if plvs_beta_eda_group: analysis_metrics[f'plv_avg_beta_{group_name}_eda_{cond_name}'] = np.nanmean(plvs_beta_eda_group)

            self.logger.info("ConnectivityAnalyzer - PLV calculations completed.")
        except Exception as e:
            self.logger.error(f"ConnectivityAnalyzer - Error during PLV calculation: {e}", exc_info=True)