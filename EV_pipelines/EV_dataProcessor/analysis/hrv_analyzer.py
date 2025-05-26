import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.interpolate import interp1d # For continuous HRV signal
from ..orchestrators import config # Relative import

class HRVAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("HRVAnalyzer initialized.")

    def calculate_rmssd_from_nni_array(self, nn_intervals_ms):
        """Calculates RMSSD directly from an array of NN intervals in milliseconds."""
        if nn_intervals_ms is None or len(nn_intervals_ms) < 2:
            self.logger.warning("HRVAnalyzer (array) - Not enough NN intervals for RMSSD.")
            return np.nan
        try:
            diff_nn = np.diff(nn_intervals_ms)
            rmssd = np.sqrt(np.mean(diff_nn ** 2))
            self.logger.info(f"HRVAnalyzer (array) - Calculated RMSSD: {rmssd:.2f} ms")
            return rmssd
        except Exception as e:
            self.logger.error(f"HRVAnalyzer (array) - Error calculating RMSSD: {e}", exc_info=True)
            return np.nan

    def calculate_hrv_metrics_from_nni_file(self, nn_intervals_path):
        """
        Calculates overall HRV metrics (e.g., RMSSD) from NN intervals file.
        Returns a dictionary of metrics.
        """
        hrv_metrics = {}
        if nn_intervals_path is None or not os.path.exists(nn_intervals_path):
            self.logger.warning("HRVAnalyzer - NN intervals file not found. Skipping overall HRV calculation.")
            hrv_metrics['hrv_rmssd_Overall'] = np.nan
            return hrv_metrics

        self.logger.info("HRVAnalyzer - Calculating overall HRV metrics from file.")
        try:
            nn_intervals_df = pd.read_csv(nn_intervals_path)
            nn_intervals_ms = nn_intervals_df['NN_Interval_ms'].dropna().values # Assuming ms column

            if len(nn_intervals_ms) < 2:
                 self.logger.warning("HRVAnalyzer - Not enough valid NN intervals to calculate overall HRV. Skipping.")
                 hrv_metrics['hrv_rmssd_Overall'] = np.nan
                 return hrv_metrics
            
            hrv_metrics['hrv_rmssd_Overall'] = self.calculate_rmssd_from_nni_array(nn_intervals_ms)
            self.logger.info("HRVAnalyzer - Overall HRV calculation from file completed.")
            return hrv_metrics
        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error calculating overall HRV from file: {e}", exc_info=True)
            hrv_metrics['hrv_rmssd_Overall'] = np.nan
            return hrv_metrics

    def calculate_resting_state_hrv(self, ecg_signal, ecg_sfreq, ecg_times, 
                                    baseline_start_time_abs, baseline_end_time_abs):
        """
        Calculates resting-state HRV metrics (e.g., RMSSD) from the baseline period.
        Returns a dictionary of metrics.
        """
        resting_hrv_metrics = {'resting_state_rmssd': np.nan}
        if ecg_signal is None or ecg_sfreq is None or ecg_times is None or \
           baseline_start_time_abs is None or baseline_end_time_abs is None:
            self.logger.warning("HRVAnalyzer - Insufficient data or timing info for resting-state HRV. Skipping.")
            return resting_hrv_metrics

        self.logger.info(f"HRVAnalyzer - Calculating resting-state HRV ({baseline_start_time_abs:.2f}s to {baseline_end_time_abs:.2f}s).")
        try:
            baseline_indices = np.where((ecg_times >= baseline_start_time_abs) & (ecg_times < baseline_end_time_abs))[0]

            if len(baseline_indices) < ecg_sfreq * 5: # Require at least 5s of data
                self.logger.warning(f"HRVAnalyzer - Baseline ECG segment too short ({len(baseline_indices)/ecg_sfreq:.2f}s). Skipping resting-state HRV.")
                return resting_hrv_metrics

            baseline_ecg_signal = ecg_signal[baseline_indices]
            signals, info = nk.ecg_peaks(baseline_ecg_signal, sampling_rate=ecg_sfreq, method=config.ECG_PEAK_DETECTION_METHOD)
            rpeaks_baseline = info["ECG_R_Peaks"]

            if len(rpeaks_baseline) < 2:
                self.logger.warning("HRVAnalyzer - Less than 2 R-peaks in baseline. Cannot calculate RMSSD.")
                return resting_hrv_metrics

            nn_intervals_baseline_ms = np.diff(rpeaks_baseline) / ecg_sfreq * 1000
            resting_hrv_metrics['resting_state_rmssd'] = self.calculate_rmssd_from_nni_array(nn_intervals_baseline_ms)
            return resting_hrv_metrics
        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error calculating resting-state HRV: {e}", exc_info=True)
            return resting_hrv_metrics
    
    def get_continuous_hrv_signal(self, rpeaks_samples, original_sfreq, target_sfreq):
        """
        Generates a continuous, interpolated HRV signal (e.g., NN intervals) from R-peak samples.
        Args:
            rpeaks_samples (np.ndarray): Array of R-peak sample indices.
            original_sfreq (float): Sampling frequency of the signal from which R-peaks were derived.
            target_sfreq (float): Target sampling frequency for the continuous HRV signal.
        Returns:
            tuple: (continuous_hrv_signal, time_vector_for_hrv_signal) or (None, None)
        """
        if rpeaks_samples is None or len(rpeaks_samples) < 2:
            self.logger.warning("HRVAnalyzer - Not enough R-peaks to generate continuous HRV signal.")
            return None, None
        
        try:
            nn_intervals_ms = np.diff(rpeaks_samples) / original_sfreq * 1000
            rpeak_times_sec = rpeaks_samples / original_sfreq # Absolute times of R-peaks

            # We need times corresponding to the NN intervals for interpolation.
            # An NNI value is typically associated with the time of the second R-peak that forms it.
            nn_interval_times_sec = rpeak_times_sec[1:]

            if len(nn_interval_times_sec) != len(nn_intervals_ms):
                self.logger.error("HRVAnalyzer - Mismatch between NN interval times and values. Cannot interpolate.")
                return None, None
            
            # Check for NaNs or Infs in the data
            if np.any(np.isnan(nn_intervals_ms)) or np.any(np.isinf(nn_intervals_ms)):
                self.logger.error("HRVAnalyzer - NaN or Inf found in NN interval values. Cannot interpolate.")
                return None, None
            
            if np.any(np.isnan(nn_interval_times_sec)) or np.any(np.isinf(nn_interval_times_sec)):
                self.logger.error("HRVAnalyzer - NaN or Inf found in NN interval times. Cannot interpolate.")
                return None, None

            # Check for strict monotonicity in nn_interval_times_sec
            if len(nn_interval_times_sec) > 1 and not np.all(np.diff(nn_interval_times_sec) > 0):
                self.logger.error("HRVAnalyzer - NN interval times are not strictly monotonically increasing. Cannot interpolate.")
                return None, None
            
            num_nni_points = len(nn_intervals_ms)
            if num_nni_points < 2 : # Need at least two points to interpolate
                 self.logger.warning("HRVAnalyzer - Less than 2 NN intervals. Cannot interpolate.")
                 return None, None

            # Determine interpolation kind based on number of points
            if num_nni_points >= 4:
                interpolation_kind = 'cubic'
            elif num_nni_points == 3:
                interpolation_kind = 'quadratic'
            else: # num_nni_points == 2
                interpolation_kind = 'linear'
            
            self.logger.info(f"HRVAnalyzer - Using '{interpolation_kind}' interpolation for {num_nni_points} NNI points.")
            interp_func = interp1d(nn_interval_times_sec, nn_intervals_ms, kind=interpolation_kind, fill_value="extrapolate")

            # Create new time vector from the first NNI time to the last
            min_time = nn_interval_times_sec[0]
            max_time = nn_interval_times_sec[-1]
            
            if max_time <= min_time: # Should not happen if len(nn_intervals_ms) >=1
                self.logger.warning("HRVAnalyzer - Max time not greater than min time for interpolation.")
                return None, None

            new_time_vector = np.arange(min_time, max_time, 1.0/target_sfreq)
            if len(new_time_vector) == 0:
                # If the duration is too short for the target_sfreq, new_time_vector can be empty.
                self.logger.warning("HRVAnalyzer - Interpolated time vector is empty. Check duration and target sfreq.")
                return None, None # Or handle differently, e.g., return raw NNIs if short

            interpolated_signal_ms = interp_func(new_time_vector)
            self.logger.info(f"HRVAnalyzer - Generated continuous HRV signal (NNIs in ms) at {target_sfreq} Hz.")
            return interpolated_signal_ms, new_time_vector

        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error generating continuous HRV signal: {e}", exc_info=True)
            return None, None

    # calculate_hrv_metrics_per_condition and get_hrv_phase_signal might be too specific
    # or better handled by the orchestrator/connectivity_analyzer using the continuous HRV signal.
    # For now, keeping them out of the primary delegation path from AnalysisService.