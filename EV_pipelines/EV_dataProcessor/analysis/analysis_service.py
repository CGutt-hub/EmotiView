import numpy as np
import pandas as pd
from scipy.signal import hilbert, resample
from scipy.interpolate import interp1d
import mne
from .. import config
# For stats, you might use:
# import pingouin as pg
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm

class AnalysisService:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("AnalysisService initialized.")

    def calculate_rmssd(self, nn_intervals_ms):
        """
        Calculates RMSSD from NN intervals.
        Args:
            nn_intervals_ms (np.ndarray): Array of NN intervals in milliseconds.
        Returns:
            float: RMSSD value, or None if input is insufficient.
        """
        if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
            self.logger.warning("RMSSD calculation: Not enough NN intervals.")
            return None
        try:
            diff_nn = np.diff(nn_intervals_ms)
            rmssd = np.sqrt(np.mean(diff_nn ** 2))
            self.logger.info(f"RMSSD calculated: {rmssd:.2f} ms")
            return rmssd
        except Exception as e:
            self.logger.error(f"Error calculating RMSSD: {e}", exc_info=True)
            return None

    def interpolate_nn_intervals(self, nn_intervals_ms, original_sfreq, target_sfreq):
        """
        Interpolates NN intervals to create a continuous HRV signal.
        Args:
            nn_intervals_ms (np.ndarray): NN intervals in milliseconds.
            original_sfreq (float): Approximate original sampling frequency of R-peaks.
            target_sfreq (float): Target sampling frequency for the continuous signal.
        Returns:
            tuple(np.ndarray, np.ndarray): time_vector, interpolated_signal or (None, None)
        """
        if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
            self.logger.warning("NN interval interpolation: Insufficient data.")
            return None, None
        try:
            # Create a time vector for NN intervals (cumulative sum of intervals)
            nn_times_sec = np.cumsum(nn_intervals_ms) / 1000.0
            nn_times_sec = np.insert(nn_times_sec, 0, 0) # Start at t=0
            
            # Use the interval values at the time of the *second* R-peak of the pair
            # For interpolation, we need values at specific time points.
            # Let's use the nn_intervals themselves as the values at nn_times_sec[1:]
            
            if len(nn_times_sec[1:]) != len(nn_intervals_ms): # Should match
                self.logger.error("NN interval interpolation: Mismatch in time and interval array lengths.")
                return None, None

            # Interpolation function
            interp_func = interp1d(nn_times_sec[1:], nn_intervals_ms, kind='cubic', fill_value="extrapolate")
            
            # Create new time vector at target_sfreq
            max_time = nn_times_sec[-1]
            new_time_vector = np.arange(0, max_time, 1.0/target_sfreq)
            
            interpolated_signal = interp_func(new_time_vector)
            self.logger.info(f"NN intervals interpolated to {target_sfreq} Hz continuous signal.")
            return new_time_vector, interpolated_signal
        except Exception as e:
            self.logger.error(f"Error interpolating NN intervals: {e}", exc_info=True)
            return None, None

    def resample_signal(self, signal, original_sfreq, target_sfreq):
        """Resamples a signal to a target sampling frequency."""
        if signal is None:
            return None
        try:
            num_samples_target = int(len(signal) * target_sfreq / original_sfreq)
            resampled_sig = resample(signal, num_samples_target)
            self.logger.info(f"Signal resampled from {original_sfreq} Hz to {target_sfreq} Hz.")
            return resampled_sig
        except Exception as e:
            self.logger.error(f"Error resampling signal: {e}", exc_info=True)
            return None

    def calculate_plv(self, signal1, signal2, sfreq, bandpass_freqs=None):
        """
        Calculates Phase Locking Value (PLV) between two signals.
        Args:
            signal1 (np.ndarray): First signal.
            signal2 (np.ndarray): Second signal.
            sfreq (float): Sampling frequency of the signals.
            bandpass_freqs (tuple, optional): (l_freq, h_freq) for bandpass filtering signal1 (e.g., EEG).
        Returns:
            float: PLV value (0 to 1), or None if error.
        """
        if signal1 is None or signal2 is None or len(signal1) != len(signal2) or len(signal1) == 0:
            self.logger.warning("PLV calculation: Invalid input signals.")
            return None
        try:
            # Optional: Filter signal1 (e.g., EEG)
            s1_filtered = signal1.copy()
            if bandpass_freqs:
                s1_filtered = mne.filter.filter_data(s1_filtered.astype(np.float64), sfreq, 
                                                     l_freq=bandpass_freqs[0], h_freq=bandpass_freqs[1],
                                                     verbose=False, fir_design='firwin')

            # Get analytic signal and phase using Hilbert transform
            phase1 = np.angle(hilbert(s1_filtered))
            phase2 = np.angle(hilbert(signal2)) # Autonomic signal usually not filtered again here

            # Calculate phase difference
            phase_diff = phase1 - phase2

            # Calculate PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            self.logger.info(f"PLV calculated: {plv:.3f}" + (f" for band {bandpass_freqs}" if bandpass_freqs else ""))
            return plv
        except Exception as e:
            self.logger.error(f"Error calculating PLV: {e}", exc_info=True)
            return None

    def calculate_band_power(self, epochs_mne, freq_band, picks=None):
        """Calculates power in a specific frequency band from MNE Epochs."""
        if epochs_mne is None:
            return None
        try:
            # Using Welch's method for power spectral density
            spectrum = epochs_mne.compute_psd(method='welch', fmin=freq_band[0], fmax=freq_band[1],
                                              picks=picks, verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)
            
            # Average power across the band and epochs (if multiple)
            # psds shape is (n_epochs, n_channels, n_freqs) or (n_channels, n_freqs) if averaged
            if psds.ndim == 3: # epochs, channels, freqs
                band_power = np.mean(np.mean(psds, axis=2), axis=0) # Avg over freqs, then epochs -> (n_channels,)
            elif psds.ndim == 2: # channels, freqs (already averaged over epochs)
                band_power = np.mean(psds, axis=1) # Avg over freqs -> (n_channels,)
            else: # Should not happen with compute_psd
                band_power = np.mean(psds)

            self.logger.info(f"Calculated power for band {freq_band} Hz.")
            return band_power # Returns array of power per channel
        except Exception as e:
            self.logger.error(f"Error calculating band power for {freq_band} Hz: {e}", exc_info=True)
            return None

    def calculate_fai(self, alpha_power_right, alpha_power_left):
        """
        Calculates Frontal Asymmetry Index (FAI).
        FAI = ln(Power_Right) - ln(Power_Left)
        Args:
            alpha_power_right (float): Alpha power at a right frontal electrode.
            alpha_power_left (float): Alpha power at a homologous left frontal electrode.
        Returns:
            float: FAI value, or None if error.
        """
        if alpha_power_right is None or alpha_power_left is None or alpha_power_right <= 0 or alpha_power_left <= 0:
            self.logger.warning("FAI calculation: Invalid alpha power values (must be > 0).")
            return None
        try:
            fai = np.log(alpha_power_right) - np.log(alpha_power_left)
            self.logger.info(f"FAI calculated: {fai:.3f}")
            return fai
        except Exception as e:
            self.logger.error(f"Error calculating FAI: {e}", exc_info=True)
            return None

    def run_fnirs_glm(self, fnirs_epochs_mne, events_df):
        """Placeholder for fNIRS GLM analysis."""
        # This would involve creating a design matrix based on events_df (conditions)
        # and running mne_nirs.statistics.run_glm or similar.
        self.logger.info("Placeholder: Running fNIRS GLM analysis...")
        # Example output: dictionary of ROI names to t-values or significance
        # For now, let's simulate finding some ROIs
        # In reality, this would come from statistical maps.
        significant_rois_info = {
            'simulated_dlpfc_l_channels': ['S1_D1', 'S1_D2'], # Example channel names
            'simulated_vmpfc_channels': ['S3_D4', 'S3_D5']   # Example channel names
        }
        self.logger.info(f"Placeholder: fNIRS GLM identified ROIs: {list(significant_rois_info.keys())}")
        return significant_rois_info

    def run_repeated_measures_anova(self, data_df, dv, within_subject_factor, subject_id_col):
        """Placeholder for Repeated Measures ANOVA."""
        # Example using pingouin:
        # aov = pg.rm_anova(data=data_df, dv=dv, within=within_subject_factor, subject=subject_id_col, detailed=True)
        # self.logger.info(f"Repeated Measures ANOVA results for '{dv}':\n{aov}")
        self.logger.info(f"Placeholder: Running RM ANOVA for DV '{dv}' with factor '{within_subject_factor}'.")
        # Return the ANOVA table or a summary
        return pd.DataFrame({'source': [within_subject_factor], 'p-unc': [0.04]}) # Dummy result

    def run_correlation_analysis(self, series1, series2):
        """Placeholder for Pearson correlation."""
        # Example using pingouin:
        # corr_result = pg.corr(series1, series2)
        # self.logger.info(f"Correlation results:\n{corr_result}")
        self.logger.info(f"Placeholder: Running Pearson correlation.")
        # Return correlation coefficient and p-value
        return 0.5, 0.03 # Dummy result (r, p)