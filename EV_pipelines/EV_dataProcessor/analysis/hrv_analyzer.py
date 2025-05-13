import os
import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from ... import config # Relative import

class HRVAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("HRVAnalyzer initialized.")

    def calculate_hrv_metrics(self, nn_intervals_path, analysis_metrics):
        """
        Calculates time-domain HRV metrics from NN-intervals.
        Args:
            nn_intervals_path (str): Path to the CSV file containing NN-intervals (in ms).
            analysis_metrics (dict): Dictionary to store/update results.
        """
        if nn_intervals_path is None or not os.path.exists(nn_intervals_path):
            self.logger.warning(f"HRVAnalyzer - NN-intervals file not found or not provided: {nn_intervals_path}. Skipping HRV metrics.")
            analysis_metrics['rmssd_overall'] = np.nan # Ensure key exists if expected
            return

        self.logger.info(f"HRVAnalyzer - Calculating HRV metrics from {nn_intervals_path}.")
        try:
            nn_intervals_df = pd.read_csv(nn_intervals_path)
            if 'NN_ms' not in nn_intervals_df.columns or nn_intervals_df['NN_ms'].isnull().all():
                self.logger.warning("HRVAnalyzer - 'NN_ms' column not found or all NaN in NN-intervals file.")
                analysis_metrics['rmssd_overall'] = np.nan
                return

            # NeuroKit2 expects NN-intervals in milliseconds.
            hrv_time_domain = nk.hrv_time(nn_intervals_df['NN_ms'].dropna(), sampling_rate=1000) # sampling_rate is for internal processing if needed
            analysis_metrics['rmssd_overall'] = hrv_time_domain['HRV_RMSSD'].iloc[0]
            # Add other metrics as needed: analysis_metrics['sdnn_overall'] = hrv_time_domain['HRV_SDNN'].iloc[0]
            self.logger.info(f"HRVAnalyzer - RMSSD calculated: {analysis_metrics['rmssd_overall']:.2f} ms.")
        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error calculating HRV metrics: {e}", exc_info=True)
            analysis_metrics['rmssd_overall'] = np.nan

    def get_hrv_phase_signal(self, rpeak_times_path, nn_intervals_path):
        """
        Creates a continuous HRV signal and extracts its phase.
        Args:
            rpeak_times_path (str): Path to CSV file with R-peak times in seconds.
            nn_intervals_path (str): Path to CSV file with NN-intervals in ms.
        Returns:
            tuple: (phase_hrv_signal, target_time_vector_hrv) or (None, None) if error.
        """
        if not rpeak_times_path or not os.path.exists(rpeak_times_path) or \
           not nn_intervals_path or not os.path.exists(nn_intervals_path):
            self.logger.warning("HRVAnalyzer - R-peak times or NN-intervals file missing for phase signal generation.")
            return None, None
        try:
            rpeak_times_sec = pd.read_csv(rpeak_times_path)['R_Peak_Time_s'].values
            nn_values_ms = pd.read_csv(nn_intervals_path)['NN_ms'].values
            if len(rpeak_times_sec) < 2 or len(nn_values_ms) < 1 or len(rpeak_times_sec) != len(nn_values_ms) + 1:
                self.logger.warning("HRVAnalyzer - Data length mismatch or insufficient data for HRV interpolation.")
                return None, None
            nn_interp_times = rpeak_times_sec[:-1] + np.diff(rpeak_times_sec) / 2
            interp_func_hrv = interp1d(nn_interp_times, nn_values_ms, kind='cubic', fill_value="extrapolate")
            target_time_hrv = np.arange(nn_interp_times[0], nn_interp_times[-1], 1.0 / config.AUTONOMIC_RESAMPLE_SFREQ)
            continuous_hrv_signal = interp_func_hrv(target_time_hrv)
            phase_hrv = np.angle(hilbert(continuous_hrv_signal - np.mean(continuous_hrv_signal)))
            self.logger.info(f"HRVAnalyzer - HRV phase signal generated (length: {len(phase_hrv)}).")
            return phase_hrv, target_time_hrv
        except Exception as e:
            self.logger.error(f"HRVAnalyzer - Error generating HRV phase signal: {e}", exc_info=True)
            return None, None