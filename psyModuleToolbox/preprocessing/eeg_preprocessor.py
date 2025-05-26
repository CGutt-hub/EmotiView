import mne
from mne_icalabel import label_components # For automatic ICA component labeling

class EEGPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("EEGPreprocessor initialized.")

    def process(self, raw_eeg, 
                  eeg_filter_band_config, 
                  ica_n_components_config, 
                  ica_random_state_config,
                  ica_accept_labels_config,
                  ica_reject_threshold_config):
        """
        Preprocesses raw EEG data.
        Args:
            raw_eeg (mne.io.Raw): Raw EEG data.
            eeg_filter_band_config (tuple): Low and high cut-off frequencies for filtering (e.g., (0.5, 40.0)).
            ica_n_components_config (int or float or None): Number of ICA components.
            ica_random_state_config (int or None): Random state for ICA.
            ica_accept_labels_config (list): List of ICA component labels to keep (e.g., ['brain', 'other']).
            ica_reject_threshold_config (float): Probability threshold to reject components not in accept_labels.
        Returns:
            mne.io.Raw: Preprocessed EEG data, or None if input is None or error.
        """
        if raw_eeg is None:
            self.logger.warning("EEGPreprocessor - No raw EEG data provided. Skipping.")
            return None
        
        if not all([eeg_filter_band_config, ica_n_components_config is not None, 
                      ica_random_state_config is not None, ica_accept_labels_config, 
                      ica_reject_threshold_config is not None]):
            self.logger.warning("EEGPreprocessor - One or more critical EEG preprocessing configurations not provided. Skipping.")
            return None
            
        self.logger.info("EEGPreprocessor - Starting EEG preprocessing.")
        try:
            # Ensure data is loaded if it's not already
            if hasattr(raw_eeg, '_data') and raw_eeg._data is None and raw_eeg.preload is False:
                 raw_eeg.load_data(verbose=False)
            self.logger.info(f"EEGPreprocessor - Filtering EEG: {eeg_filter_band_config[0]}-{eeg_filter_band_config[1]} Hz.")
            raw_eeg.filter(l_freq=eeg_filter_band_config[0], h_freq=eeg_filter_band_config[1], 
                           fir_design='firwin', verbose=False)

            self.logger.info("EEGPreprocessor - Setting average reference.")
            raw_eeg.set_eeg_reference('average', projection=True, verbose=False)
            self.logger.info(f"EEGPreprocessor - Fitting ICA with {ica_n_components_config} components.")
            ica = mne.preprocessing.ICA(n_components=ica_n_components_config, random_state=ica_random_state_config, max_iter='auto')
            ica.fit(raw_eeg, verbose=False)

            # Automatic artifact labeling (optional, requires mne_icalabel)
            self.logger.info("EEGPreprocessor - Attempting automatic ICA component labeling.")
            try:
                component_labels = label_components(raw_eeg, ica, method='iclabel')
                labels = component_labels["labels"]
                probabilities = component_labels["y_pred_proba"]
                
                exclude_idx = [
                    idx for idx, label in enumerate(labels)
                    if label not in ica_accept_labels_config and 
                       probabilities[idx, list(component_labels['classes']).index(label)] > ica_reject_threshold_config
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