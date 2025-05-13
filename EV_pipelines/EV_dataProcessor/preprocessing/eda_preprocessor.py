import os
import pandas as pd
import neurokit2 as nk
from ... import config # Relative import

class EDAPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EDAPreprocessor initialized.")

    def process_and_save(self, eda_signal_raw, eda_sampling_rate, participant_id, output_dir):
        """
        Processes raw EDA signal to extract phasic component and saves it.
        Args:
            eda_signal_raw (np.ndarray): The raw EDA signal.
            eda_sampling_rate (float): The sampling rate of the EDA signal.
            participant_id (str): The participant ID.
            output_dir (str): Directory to save the output file.
        Returns:
            str: Path to the saved phasic EDA CSV file, or None if error.
        """
        if eda_signal_raw is None or eda_sampling_rate is None:
            self.logger.warning("EDAPreprocessor - Raw EDA signal or sampling rate not provided. Skipping.")
            return None

        self.logger.info(f"EDAPreprocessor - Processing EDA for {participant_id}.")
        try:
            eda_signals, _ = nk.eda_process(eda_signal_raw, sampling_rate=eda_sampling_rate)
            phasic_eda = eda_signals['EDA_Phasic'].values # Ensure it's a numpy array
            self.logger.debug("EDAPreprocessor - EDA signal decomposed, phasic component extracted.")

            phasic_eda_path = os.path.join(output_dir, f"{participant_id}_phasic_eda.csv")
            pd.DataFrame(phasic_eda, columns=['EDA_Phasic']).to_csv(phasic_eda_path, index=False)
            self.logger.info(f"EDAPreprocessor - Phasic EDA saved to {phasic_eda_path}")
            return phasic_eda_path
        except Exception as e:
            self.logger.error(f"EDAPreprocessor - Error processing EDA for {participant_id}: {e}", exc_info=True)
            return None