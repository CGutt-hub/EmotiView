import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch
from ..orchestrators import config # Relative import

class PSDAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("PSDAnalyzer initialized.")

    def _calculate_psd_for_epochs(self, epochs, sfreq, band_freqs, condition_name_logging=""):
        """Helper to compute PSD for given epochs and band."""
        try:
            psds, freqs = epochs.compute_psd(
                method='welch', fmin=band_freqs[0], fmax=band_freqs[1],
                n_fft=int(sfreq), n_overlap=int(sfreq * 0.5), # Adjust n_fft and n_overlap as needed
                average='mean', verbose=False).get_data(return_freqs=True)
            
            # psds shape is (n_epochs_averaged_to_1, n_channels, n_freqs)
            # We want mean power per channel across the frequency band
            mean_power_per_channel = np.mean(psds, axis=2).squeeze() # Squeeze to remove first dim if 1
            return mean_power_per_channel
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Error computing PSD for band {band_freqs} {condition_name_logging}: {e}", exc_info=True)
            return None


    def calculate_psd_and_fai(self, raw_eeg_processed, events, event_id_map):
        """
        Calculates Power Spectral Density (PSD) for alpha and beta bands,
        and Frontal Asymmetry Index (FAI) for alpha band for each condition.

        Args:
            raw_eeg_processed (mne.io.Raw): Processed MNE Raw object for EEG.
            events (np.ndarray): MNE events array from mne.events_from_annotations.
            event_id_map (dict): Mapping from condition names to event codes.

        Returns:
            tuple: (psd_results, fai_results)
                   psd_results (dict): {'condition': {'band': {'channel': power}}}
                   fai_results (dict): {'condition': {'pair_name': fai_value}}
        """
        if raw_eeg_processed is None:
            self.logger.warning("PSDAnalyzer - No processed EEG data provided. Skipping PSD and FAI calculation.")
            return {}, {}
        if events is None or not events.size or event_id_map is None or not event_id_map:
            self.logger.warning("PSDAnalyzer - Events or event_id_map not provided or empty. Skipping PSD and FAI calculation.")
            return {}, {}

        self.logger.info("PSDAnalyzer - Calculating PSD and FAI.")
        sfreq = raw_eeg_processed.info['sfreq']
        
        # Use bands from config
        alpha_band = config.FAI_ALPHA_BAND # Assuming FAI alpha band is primary alpha
        beta_band = config.PLV_EEG_BANDS.get('Beta', (13.0, 30.0)) # Get Beta band from PLV config or default

        fai_pairs_config = config.FAI_ELECTRODE_PAIRS # e.g., [('Fp1', 'Fp2'), ('F3', 'F4')]

        psd_results = {} # To store all PSD values: condition -> band -> channel -> power
        fai_results = {} # To store FAI values: condition -> pair_name -> fai

        # Channels to pick for PSD calculation (superset for FAI and general PSD)
        all_fai_channels = list(set(ch for pair in fai_pairs_config for ch in pair))
        # You might want a broader set of channels for general PSD reporting if needed
        # For now, focus on channels relevant to FAI
        psd_picks = [ch for ch in all_fai_channels if ch in raw_eeg_processed.ch_names]
        
        if not psd_picks:
            self.logger.warning(f"PSDAnalyzer - None of the FAI channels found in EEG data: {all_fai_channels}")
            return {}, {}
        self.logger.info(f"PSDAnalyzer - Will calculate PSD for FAI-relevant channels: {psd_picks}")

        try:
            for condition_name, event_code in event_id_map.items():
                if condition_name.lower() in ["bad_stim", "boundary", "edge"]:
                    self.logger.debug(f"PSDAnalyzer - Skipping non-experimental condition '{condition_name}'.")
                    continue
                
                self.logger.debug(f"PSDAnalyzer - Processing PSD/FAI for condition: {condition_name}")
                psd_results[condition_name] = {'Alpha': {}, 'Beta': {}}
                fai_results[condition_name] = {}

                try:
                    epochs = mne.Epochs(raw_eeg_processed, events, event_id={condition_name: event_code},
                                        tmin=0.0, tmax=config.ANALYSIS_EPOCH_TIMES[1], # Use full epoch duration
                                        baseline=None, preload=True, picks=psd_picks, verbose=False)

                    if len(epochs) == 0:
                        self.logger.info(f"PSDAnalyzer - No epochs for condition '{condition_name}'.")
                        continue

                    # Alpha PSD
                    alpha_power_per_channel = self._calculate_psd_for_epochs(epochs, sfreq, alpha_band, f"Alpha, Cond: {condition_name}")
                    if alpha_power_per_channel is not None:
                        for ch_idx, ch_name in enumerate(epochs.ch_names):
                            psd_results[condition_name]['Alpha'][ch_name] = alpha_power_per_channel[ch_idx]
                        
                        # Calculate FAI using these alpha powers
                        for ch_left, ch_right in fai_pairs_config: # Iterate through configured pairs
                            pair_name = f"{ch_right}_vs_{ch_left}" # e.g., Fp2_vs_Fp1
                            power_left = psd_results[condition_name]['Alpha'].get(ch_left)
                            power_right = psd_results[condition_name]['Alpha'].get(ch_right)

                            if power_left is not None and power_right is not None:
                                if power_left > 1e-12 and power_right > 1e-12: # Avoid log(0) or log(negative)
                                    fai_val = np.log(power_right) - np.log(power_left)
                                    fai_results[condition_name][pair_name] = fai_val
                                    self.logger.info(f"PSDAnalyzer - FAI {pair_name} for {condition_name}: {fai_val:.4f}")
                                else:
                                    self.logger.warning(f"PSDAnalyzer - Zero/tiny power for FAI {pair_name}, {condition_name}. FAI is NaN.")
                                    fai_results[condition_name][pair_name] = np.nan
                            else:
                                self.logger.warning(f"PSDAnalyzer - Missing alpha power for FAI {pair_name}, {condition_name}. FAI is NaN.")
                                fai_results[condition_name][pair_name] = np.nan
                    
                    # Beta PSD
                    beta_power_per_channel = self._calculate_psd_for_epochs(epochs, sfreq, beta_band, f"Beta, Cond: {condition_name}")
                    if beta_power_per_channel is not None:
                        for ch_idx, ch_name in enumerate(epochs.ch_names):
                            psd_results[condition_name]['Beta'][ch_name] = beta_power_per_channel[ch_idx]

                except Exception as e_cond:
                    self.logger.error(f"PSDAnalyzer - Error processing condition '{condition_name}': {e_cond}", exc_info=True)

            self.logger.info("PSDAnalyzer - PSD and FAI calculation completed.")
            return psd_results, fai_results
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Error calculating PSD/FAI: {e}", exc_info=True)
            return {}, {}