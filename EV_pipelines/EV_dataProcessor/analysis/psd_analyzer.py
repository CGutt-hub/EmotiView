import mne
import numpy as np
from .. import config # Relative import

class PSDAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("PSDAnalyzer initialized.")

    def calculate_psd_and_fai(self, raw_eeg, analysis_metrics):
        """
        Calculates Power Spectral Density (PSD) for alpha and beta bands,
        and Frontal Asymmetry Index (FAI) for alpha band.
        Updates the analysis_metrics dictionary.
        Args:
            raw_eeg (mne.io.Raw): Preprocessed EEG raw object.
            analysis_metrics (dict): Dictionary to store results.
        """
        if raw_eeg is None:
            self.logger.warning("PSDAnalyzer - No EEG data provided. Skipping PSD and FAI.")
            return

        self.logger.info("PSDAnalyzer - Calculating PSD and FAI.")
        try:
            # Assuming events are annotated for epoching, or use fixed-length epochs
            events, event_id = mne.events_from_annotations(raw_eeg, verbose=False)
            
            psd_picks = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_FAI_PSD if ch in raw_eeg.ch_names]
            if not psd_picks:
                self.logger.warning(f"PSDAnalyzer - None of the default FAI/PSD channels found in EEG data: {config.DEFAULT_EEG_CHANNELS_FOR_FAI_PSD}")
                return

            self.logger.info(f"PSDAnalyzer - Calculating PSD for channels: {psd_picks}")

            # Iterate through conditions to calculate condition-specific PSD and FAI
            for condition_name, event_code in event_id.items():
                self.logger.debug(f"PSDAnalyzer - Processing condition: {condition_name}")
                try:
                    # Create epochs for the specific condition
                    condition_epochs = mne.Epochs(raw_eeg, events, event_id={condition_name: event_code},
                                                  tmin=-0.2, tmax=config.STIMULUS_DURATION_SECONDS, baseline=(None, 0),
                                                  preload=True, verbose=False)
                    if len(condition_epochs) == 0:
                        self.logger.info(f"PSDAnalyzer - No epochs found for condition '{condition_name}'. Skipping PSD/FAI for this condition.")
                        continue

                    psds_alpha_cond, _ = condition_epochs.compute_psd(fmin=8, fmax=13, method='welch', picks=psd_picks, verbose=False).get_data(return_freqs=True)
                    psds_beta_cond, _ = condition_epochs.compute_psd(fmin=13, fmax=30, method='welch', picks=psd_picks, verbose=False).get_data(return_freqs=True)

                    # Store mean power per channel for this condition
                    condition_alpha_powers = {}
                    for ch_idx, ch_name_picked in enumerate(psd_picks): # Iterate over actual picked channels
                        # psds_alpha_cond shape: (n_epochs, n_channels, n_freqs)
                        mean_alpha_power = np.mean(psds_alpha_cond[:, ch_idx, :])
                        mean_beta_power = np.mean(psds_beta_cond[:, ch_idx, :])
                        analysis_metrics[f'alpha_power_{ch_name_picked}_{condition_name}'] = mean_alpha_power
                        analysis_metrics[f'beta_power_{ch_name_picked}_{condition_name}'] = mean_beta_power
                        condition_alpha_powers[ch_name_picked] = mean_alpha_power

                    # FAI Calculation (Alpha band) for this condition
                    for pair in [('F4','F3'), ('Fp2','Fp1'), ('AF4', 'AF3')]: # Added AF pair
                        r_ch, l_ch = pair
                        r_pow = condition_alpha_powers.get(r_ch, np.nan)
                        l_pow = condition_alpha_powers.get(l_ch, np.nan)
                        if not np.isnan(r_pow) and not np.isnan(l_pow) and l_pow > 1e-9 and r_pow > 1e-9: # ensure powers are positive for log
                            analysis_metrics[f'fai_alpha_{r_ch}_{l_ch}_{condition_name}'] = np.log(r_pow) - np.log(l_pow)
                        else:
                            analysis_metrics[f'fai_alpha_{r_ch}_{l_ch}_{condition_name}'] = np.nan
                except Exception as e_cond:
                    self.logger.error(f"PSDAnalyzer - Error processing condition '{condition_name}': {e_cond}", exc_info=True)

            self.logger.info("PSDAnalyzer - PSD and FAI calculation completed.")
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Error calculating PSD/FAI: {e}", exc_info=True)