import mne
from ..orchestrators import config
from mne_icalabel import label_components # For automatic ICA component labeling

class EEGPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def process(self, raw_eeg):
        """
        Preprocesses raw EEG data.
        Args:
            raw_eeg (mne.io.Raw): Raw EEG data.
        Returns:
            mne.io.Raw: Preprocessed EEG data, or None if input is None or error.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None
            
        self.logger.info("EEGPreprocessor - Starting EEG preprocessing.")
        try:
            # Ensure data is loaded if it's not already
            if hasattr(raw_eeg, '_data') and raw_eeg._data is None and raw_eeg.preload is False:
                 raw_eeg.load_data(verbose=False)

            self.logger.info(f"EEGPreprocessor - Filtering EEG: {config.EEG_FILTER_BAND[0]}-{config.EEG_FILTER_BAND[1]} Hz.")
            raw_eeg.filter(l_freq=config.EEG_FILTER_BAND[0], h_freq=config.EEG_FILTER_BAND[1], 
                           fir_design='firwin', verbose=False)

            self.logger.info("EEGPreprocessor - Setting average reference.")
            raw_eeg.set_eeg_reference('average', projection=True, verbose=False)

            self.logger.info(f"EEGPreprocessor - Fitting ICA with {config.ICA_N_COMPONENTS} components.")
            ica = mne.preprocessing.ICA(n_components=config.ICA_N_COMPONENTS, random_state=config.ICA_RANDOM_STATE, max_iter='auto')
            ica.fit(raw_eeg, verbose=False)

            # Automatic artifact labeling (optional, requires mne_icalabel)
            self.logger.info("EEGPreprocessor - Attempting automatic ICA component labeling.")
            try:
                component_labels = label_components(raw_eeg, ica, method='iclabel')
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]
                
                exclude_idx = [
                    idx for idx, label in enumerate(labels)
                    if label not in config.ICA_ACCEPT_LABELS and 
                       probabilities[idx, list(component_labels['classes']).index(label)] > config.ICA_REJECT_THRESHOLD
                ]
                
                self.logger.info(f"EEGPreprocessor - Automatically identified {len(exclude_idx)} ICA components to exclude: {exclude_idx}")
                if exclude_idx: # Only apply if there are components to exclude
                    ica.exclude = exclude_idx
                    ica.apply(raw_eeg, verbose=False) # Apply ICA to the raw data
                    self.logger.info("EEGPreprocessor - ICA applied to remove artifact components.")
                else:
                    self.logger.info("EEGPreprocessor - No ICA components met criteria for automatic exclusion.")

            except Exception as e_icalabel:
                self.logger.warning(f"EEGPreprocessor - Automatic ICA labeling failed: {e_icalabel}. ICA components not automatically excluded. Manual inspection might be needed.", exc_info=True)

            self.logger.info("EEGPreprocessor - EEG preprocessing completed.")
            return raw_eeg
        except Exception as e:
            self.logger.error(f"EEGPreprocessor - Error during EEG preprocessing: {e}", exc_info=True)
            return None