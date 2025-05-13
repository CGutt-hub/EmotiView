import mne
from .. import config # Relative import

class EEGPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def process(self, raw_eeg, eeg_sampling_rate):
        """
        Preprocesses raw EEG data.
        Args:
            raw_eeg (mne.io.Raw): Raw EEG data.
            eeg_sampling_rate (float): Sampling rate of the EEG data.
        Returns:
            mne.io.Raw: Preprocessed EEG data, or None if input is None or error.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None
        if eeg_sampling_rate is None:
            self.logger.error("EEGPreprocessor - EEG sampling rate not provided. Cannot preprocess.")
            return None
            
        self.logger.info("EEGPreprocessor - Starting EEG preprocessing.")
        try:
            # Ensure data is loaded if it's a preloaded Raw object from a file
            if hasattr(raw_eeg, '_data') and raw_eeg._data is None and raw_eeg.preload is False:
                 raw_eeg.load_data(verbose=False)

            self.logger.info(f"EEGPreprocessor - Filtering EEG: {config.EEG_LPASS_HZ}-{config.EEG_HPASS_HZ} Hz.")
            raw_eeg.filter(l_freq=config.EEG_LPASS_HZ, h_freq=config.EEG_HPASS_HZ, 
                           fir_design='firwin', verbose=False)

            self.logger.info("EEGPreprocessor - Setting average reference.")
            raw_eeg.set_eeg_reference('average', projection=True, verbose=False)

            self.logger.info(f"EEGPreprocessor - Fitting ICA with {config.ICA_N_COMPONENTS} components.")
            ica = mne.preprocessing.ICA(n_components=config.ICA_N_COMPONENTS, 
                                        random_state=97, max_iter='auto')
            ica.fit(raw_eeg, verbose=False)

            # Automatic artifact labeling (optional, requires mne_icalabel)
            try:
                from mne_icalabel import label_components
                label_components(raw_eeg, ica, method='iclabel')
                ica.exclude = [idx for idx, label in enumerate(ica.labels_) if label not in ['brain', 'other']]
                self.logger.info(f"EEGPreprocessor - ICLabel excluded components: {ica.exclude} (Labels: {[ica.labels_[i] for i in ica.exclude]})")
            except ImportError:
                self.logger.warning("EEGPreprocessor - mne_icalabel not found. Skipping automatic ICA component labeling.")
            except Exception as e_icalabel:
                self.logger.error(f"EEGPreprocessor - Error during ICLabel: {e_icalabel}", exc_info=True)

            ica.apply(raw_eeg, verbose=False)
            self.logger.info(f"EEGPreprocessor - Applied ICA (excluded: {ica.exclude}).")
            return raw_eeg
        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error during EEG preprocessing: {e}", exc_info=True)
            return None