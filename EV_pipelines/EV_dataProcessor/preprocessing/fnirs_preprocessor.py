import mne
import mne_nirs
from .. import config # Relative import

class FNIRSPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSPreprocessor initialized.")

    def process(self, fnirs_raw_od):
        """
        Processes raw fNIRS optical density data.
        Args:
            fnirs_raw_od (mne.io.Raw): Raw fNIRS optical density data.
        Returns:
            mne.io.Raw: Preprocessed haemoglobin concentration data, or None if error.
        """
        if fnirs_raw_od is None:
            self.logger.warning("FNIRSPreprocessor - No raw fNIRS OD data provided. Skipping.")
            return None

        try:
            self.logger.info("FNIRSPreprocessor - Starting fNIRS preprocessing.")
            if hasattr(fnirs_raw_od, '_data') and fnirs_raw_od._data is None and fnirs_raw_od.preload is False:
                fnirs_raw_od.load_data(verbose=False)

            self.logger.info(f"FNIRSPreprocessor - Applying Beer-Lambert Law (PPF={config.FNIRS_BEER_LAMBERT_PPF}).")
            raw_haemo = mne_nirs.beer_lambert_law(fnirs_raw_od, ppf=config.FNIRS_BEER_LAMBERT_PPF)
            
            self.logger.info("FNIRSPreprocessor - Applying TDDR motion artifact correction.")
            corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            
            self.logger.info(f"FNIRSPreprocessor - Applying band-pass filter ({config.FNIRS_FILTER_LPASS_HZ}-{config.FNIRS_FILTER_HPASS_HZ} Hz).")
            corrected_haemo.filter(config.FNIRS_FILTER_LPASS_HZ, config.FNIRS_FILTER_HPASS_HZ,
                                   h_trans_bandwidth=0.02,
                                   l_trans_bandwidth=0.002,
                                   fir_design='firwin', verbose=False)
            
            self.logger.info("FNIRSPreprocessor - fNIRS preprocessing completed successfully.")
            return corrected_haemo
            
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error during fNIRS preprocessing: {e}", exc_info=True)
            return None

    def save_preprocessed_data(self, processed_fnirs_data, participant_id, output_dir):
        if processed_fnirs_data is None:
            self.logger.warning("FNIRSPreprocessor - No processed fNIRS data to save.")
            return None
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{participant_id}_fnirs_haemo_preprocessed.fif"
            filepath = os.path.join(output_dir, filename)
            processed_fnirs_data.save(filepath, overwrite=True, verbose=False)
            self.logger.info(f"FNIRSPreprocessor - Preprocessed fNIRS data saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error saving preprocessed fNIRS data: {e}", exc_info=True)
            return None