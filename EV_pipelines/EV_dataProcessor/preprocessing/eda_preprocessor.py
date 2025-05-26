import neurokit2 as nk
import pandas as pd
import os
import numpy as np
from ..orchestrators import config

class EDAPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDAPreprocessor initialized.")

    def preprocess_eda(self, eda_signal_raw, eda_sampling_rate, participant_id, output_dir):
        """
        Processes raw EDA signal to extract and save phasic and tonic components.
        Args:
            eda_signal_raw (np.ndarray): The raw EDA signal.
            eda_sampling_rate (float): The sampling rate of the EDA signal.
            participant_id (str): The participant ID.
            output_dir (str): Directory to save the output file.
        Returns:
            tuple: (phasic_eda_path, tonic_eda_path, phasic_eda_array, tonic_eda_array)
                   Returns (None, None, None, None) if error.
        """
        if eda_signal_raw is None or eda_sampling_rate is None:
            self.logger.warning("EDAPreprocessor - Raw EDA signal or sampling rate not provided. Skipping.")
            return None, None, None, None
        
        if eda_sampling_rate <= 0:
            self.logger.error(f"EDAPreprocessor - Invalid EDA sampling rate: {eda_sampling_rate}. Skipping.")
            return None, None, None, None

        self.logger.info(f"EDAPreprocessor - Processing EDA for {participant_id}.")
        try:
            # Ensure eda_signal_raw is a 1D numpy array
            if isinstance(eda_signal_raw, pd.Series):
                eda_signal_raw = eda_signal_raw.values
            elif isinstance(eda_signal_raw, list):
                eda_signal_raw = np.array(eda_signal_raw)
            
            if eda_signal_raw.ndim > 1:
                if eda_signal_raw.shape[0] == 1 or eda_signal_raw.shape[1] == 1:
                    eda_signal_raw = eda_signal_raw.ravel() # Flatten if it's a row or column vector
                else:
                    self.logger.error("EDAPreprocessor - EDA signal is not 1D. Cannot process.")
                    return None, None, None, None

            eda_signals, _ = nk.eda_process(eda_signal_raw, sampling_rate=int(eda_sampling_rate))
            phasic_eda = eda_signals['EDA_Phasic'].values # Ensure it's a numpy array
            tonic_eda = eda_signals['EDA_Tonic'].values   # Ensure it's a numpy array
            self.logger.debug("EDAPreprocessor - EDA signal decomposed, phasic and tonic components extracted.")

            phasic_eda_path = os.path.join(output_dir, f"{participant_id}_phasic_eda.csv")
            pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Phasic EDA saved to {phasic_eda_path}")

            tonic_eda_path = os.path.join(output_dir, f"{participant_id}_tonic_eda.csv")
            pd.DataFrame(tonic_eda, columns=['EDA_Tonic']).to_csv(tonic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Tonic EDA saved to {tonic_eda_path}")

            return phasic_eda_path, tonic_eda_path, phasic_eda, tonic_eda
        except Exception as e:
            self.logger.error(f"EDAPreprocessor - Error processing EDA for {participant_id}: {e}", exc_info=True)
            return None, None, None, None