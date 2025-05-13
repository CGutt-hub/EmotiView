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
            epochs = mne.Epochs(raw_eeg, events, event_id=event_id, 
                                tmin=-0.2, tmax=0.8, baseline=(None, 0), 
                                preload=True, verbose=False) # Adjust epoching as needed

            psd_picks = [ch for ch in config.DEFAULT_EEG_CHANNELS_FOR_FAI_PSD if ch in raw_eeg.ch_names]
            if not psd_picks:
                self.logger.warning(f"PSDAnalyzer - None of the default FAI/PSD channels found in EEG data: {config.DEFAULT_EEG_CHANNELS_FOR_FAI_PSD}")
                return

            self.logger.info(f"PSDAnalyzer - Calculating PSD for channels: {psd_picks}")
            psds_alpha, _ = epochs.compute_psd(fmin=8, fmax=13, method='welch', picks=psd_picks, verbose=False).get_data(return_freqs=True)
            psds_beta, _ = epochs.compute_psd(fmin=13, fmax=30, method='welch', picks=psd_picks, verbose=False).get_data(return_freqs=True)

            for ch_name in config.DEFAULT_EEG_CHANNELS_FOR_FAI_PSD:
                if ch_name in psd_picks:
                    ch_idx = psd_picks.index(ch_name)
                    analysis_metrics[f'alpha_power_{ch_name}_mean'] = np.mean(psds_alpha[:, ch_idx, :])
                    analysis_metrics[f'beta_power_{ch_name}_mean'] = np.mean(psds_beta[:, ch_idx, :])

            # FAI Calculation (Alpha band)
            for pair in [('F4','F3'), ('Fp2','Fp1')]:
                r_ch, l_ch = pair
                r_pow = analysis_metrics.get(f'alpha_power_{r_ch}_mean', np.nan)
                l_pow = analysis_metrics.get(f'alpha_power_{l_ch}_mean', np.nan)
                if not np.isnan(r_pow) and not np.isnan(l_pow) and l_pow > 0: # ensure l_pow is not zero for log
                    analysis_metrics[f'fai_alpha_{r_ch}_{l_ch}'] = np.log(r_pow + 1e-9) - np.log(l_pow + 1e-9) # Add epsilon
                else:
                    analysis_metrics[f'fai_alpha_{r_ch}_{l_ch}'] = np.nan
            self.logger.info("PSDAnalyzer - PSD and FAI calculation completed.")
        except Exception as e:
            self.logger.error(f"PSDAnalyzer - Error calculating PSD/FAI: {e}", exc_info=True)