import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from ... import config # Relative import

class ECGPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ECGPreprocessor initialized.")

    def process_and_save(self, ecg_signal_raw, ecg_sampling_rate, participant_id, output_dir):
        """
        Processes raw ECG signal to extract R-peaks and NN-intervals, then saves them.
        Args:
            ecg_signal_raw (np.ndarray): The raw ECG signal.
            ecg_sampling_rate (float): The sampling rate of the ECG signal.
            participant_id (str): The participant ID.
            output_dir (str): Directory to save the output files.
        Returns:
            tuple: (path_to_nn_intervals_csv, path_to_rpeaks_csv) or (None, None) if error.
        """
        if ecg_signal_raw is None or ecg_sampling_rate is None:
            self.logger.warning("ECGPreprocessor - Raw ECG signal or sampling rate not provided. Skipping.")
            return None, None

        self.logger.info(f"ECGPreprocessor - Processing ECG for {participant_id}.")
        try:
            # Clean ECG signal
            ecg_cleaned = nk.ecg_clean(ecg_signal_raw, sampling_rate=ecg_sampling_rate)
            self.logger.debug("ECGPreprocessor - ECG signal cleaned.")

            # Detect R-peaks
            # Use a robust method, correct_artifacts can be helpful
            peaks_info, _ = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_sampling_rate, 
                                         method="neurokit", correct_artifacts=True)
            rpeaks_indices = peaks_info['ECG_R_Peaks']
            self.logger.info(f"ECGPreprocessor - Detected {len(rpeaks_indices)} R-peaks.")

            if len(rpeaks_indices) < 2:
                self.logger.warning("ECGPreprocessor - Less than 2 R-peaks detected. Cannot compute NN-intervals.")
                return None, None

            # Calculate NN-intervals (in ms)
            nn_intervals_ms = np.diff(rpeaks_indices) / ecg_sampling_rate * 1000
            
            # Save NN-intervals
            nn_intervals_path = os.path.join(output_dir, f"{participant_id}_nn_intervals.csv")
            pd.DataFrame(nn_intervals_ms, columns=['NN_ms']).to_csv(nn_intervals_path, index=False)
            self.logger.info(f"ECGPreprocessor - NN-intervals saved to {nn_intervals_path}")

            # Save R-peak times (in seconds)
            rpeaks_times_sec = rpeaks_indices / ecg_sampling_rate
            rpeaks_path = os.path.join(output_dir, f"{participant_id}_rpeaks_times_sec.csv")
            pd.DataFrame(rpeaks_times_sec, columns=['R_Peak_Time_s']).to_csv(rpeaks_path, index=False)
            self.logger.info(f"ECGPreprocessor - R-peak times saved to {rpeaks_path}")

            return nn_intervals_path, rpeaks_path
        except Exception as e:
            self.logger.error(f"ECGPreprocessor - Error processing ECG for {participant_id}: {e}", exc_info=True)
            return None, None