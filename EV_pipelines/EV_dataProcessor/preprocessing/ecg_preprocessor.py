import os
import numpy as np
import pandas as pd
import neurokit2 as nk
from ..orchestrators import config # Relative import

class ECGPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("ECGPreprocessor initialized.")

    def preprocess_ecg(self, ecg_signal, ecg_sfreq, participant_id, preproc_results_dir):
        """
        Preprocesses ECG data to detect R-peaks and calculate NN intervals.

        Args:
            ecg_signal (np.ndarray): The raw ECG signal.
            ecg_sfreq (float): The sampling frequency of the ECG signal.
            participant_id (str): The ID of the participant.
            preproc_results_dir (str): Directory to save intermediate results.

        Returns:
            tuple: (rpeak_times_path, nn_intervals_path, rpeaks_samples_array, nn_intervals_ms_array)
                   Returns (None, None, None, None) if preprocessing fails.
        """
        if ecg_signal is None or ecg_sfreq is None:
            self.logger.warning("ECGPreprocessor - No ECG signal or sampling frequency provided. Skipping preprocessing.")
            return None, None

        self.logger.info(f"ECGPreprocessor - Starting preprocessing for {participant_id}.")

        try:
            # 1. Filter the ECG signal
            # NeuroKit2's processing functions often include filtering internally,
            # but an explicit filter can be added if needed before peak detection.
            # For now, rely on NeuroKit2's internal filtering within processing.

            # 2. Find R-peaks
            # Use NeuroKit2's ecg_peaks function
            self.logger.info("ECGPreprocessor - Detecting R-peaks...")
            # The `method` parameter uses the specified algorithm from config
            signals, info = nk.ecg_peaks(ecg_signal, sampling_rate=ecg_sfreq, method=config.ECG_RPEAK_METHOD)

            rpeaks = info["ECG_R_Peaks"]
            self.logger.info(f"ECGPreprocessor - Found {len(rpeaks)} R-peaks.")

            if len(rpeaks) < 2:
                self.logger.warning("ECGPreprocessor - Less than 2 R-peaks found. Cannot calculate NN intervals. Skipping.")
                return None, None, None, None

            # 3. Calculate NN intervals
            # NeuroKit2's hrv_time function can derive NN intervals from R-peaks
            # Or we can calculate them manually from peak indices
            nn_intervals_ms = np.diff(rpeaks) / ecg_sfreq * 1000 # Difference in samples, convert to seconds, then ms
            nn_intervals_s = nn_intervals_ms / 1000 # Keep in seconds for consistency with MNE/timing

            # Get R-peak times in seconds relative to the start of the signal
            rpeak_times_s = rpeaks / ecg_sfreq

            # 4. Save results
            rpeak_times_df = pd.DataFrame({'R_Peak_Time_s': rpeak_times_s})
            nn_intervals_df = pd.DataFrame({'NN_Interval_ms': nn_intervals_ms, 'NN_Interval_s': nn_intervals_s})

            rpeak_times_path = os.path.join(preproc_results_dir, f'{participant_id}_ecg_rpeak_times.csv')
            nn_intervals_path = os.path.join(preproc_results_dir, f'{participant_id}_ecg_nn_intervals.csv')

            rpeak_times_df.to_csv(rpeak_times_path, index=False)
            nn_intervals_df.to_csv(nn_intervals_path, index=False)

            self.logger.info(f"ECGPreprocessor - R-peak times saved to {rpeak_times_path}")
            self.logger.info(f"ECGPreprocessor - NN intervals saved to {nn_intervals_path}")

            self.logger.info(f"ECGPreprocessor - Preprocessing completed for {participant_id}.")
            return rpeak_times_path, nn_intervals_path, rpeaks, nn_intervals_ms

        except Exception as e:
            self.logger.error(f"ECGPreprocessor - Error during preprocessing for {participant_id}: {e}", exc_info=True)
            return None, None, None, None