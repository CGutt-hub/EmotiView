import mne
import mne_nirs
from mne_nirs.signal_enhancement import short_channel_regression
from mne_nirs.channels import get_long_channels, get_short_channels
import os
import numpy as np

class FNIRSPreprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("FNIRSPreprocessor initialized.")

    def process(self, fnirs_raw_od,
                  beer_lambert_ppf_config,
                  short_channel_regression_config,
                  motion_correction_method_config,
                  filter_band_config):
        """
        Preprocesses raw fNIRS optical density data to haemoglobin concentration.
        Args:
            fnirs_raw_od (mne.io.Raw): Raw fNIRS optical density data.
            beer_lambert_ppf_config (float or tuple): Partial pathlength factor(s) for Beer-Lambert law.
            short_channel_regression_config (bool): Whether to apply short-channel regression.
            motion_correction_method_config (str): Method for motion correction (e.g., 'tddr', 'none').
            filter_band_config (tuple): Low and high cut-off frequencies for filtering (e.g., (0.01, 0.1)).
        Returns:
            mne.io.Raw: Preprocessed fNIRS haemoglobin concentration data, or None if error.
        """
        if fnirs_raw_od is None:
            self.logger.warning("FNIRSPreprocessor - No raw fNIRS OD data provided. Skipping.")
            return None
        
        if not all([beer_lambert_ppf_config is not None, 
                      short_channel_regression_config is not None, 
                      motion_correction_method_config is not None, 
                      filter_band_config is not None]):
            self.logger.warning("FNIRSPreprocessor - One or more critical fNIRS preprocessing configurations not provided. Skipping.")
            return None

        self.logger.info("FNIRSPreprocessor - Starting fNIRS preprocessing.")
        try:
            # Ensure data is loaded
            if hasattr(fnirs_raw_od, '_data') and fnirs_raw_od._data is None and fnirs_raw_od.preload is False:
                fnirs_raw_od.load_data(verbose=False)
            self.logger.info(f"FNIRSPreprocessor - Applying Beer-Lambert Law (PPF={beer_lambert_ppf_config}).")
            raw_haemo = mne_nirs.beer_lambert_law(fnirs_raw_od.copy(), ppf=beer_lambert_ppf_config, remove_od=True)
            
            if short_channel_regression_config:
                try:
                    # Check if there are any short channels defined
                    short_chs = get_short_channels(raw_haemo)
                    long_chs = get_long_channels(raw_haemo)
                    if not short_chs.empty and not long_chs.empty:
                        self.logger.info(f"FNIRSPreprocessor - Found {len(short_chs)} short channels. Applying short-channel regression.")
                        # Ensure raw_haemo includes both long and short for regression
                        raw_haemo_corrected_short = short_channel_regression(raw_haemo.copy())
                        raw_haemo = raw_haemo_corrected_short
                    else:
                        self.logger.info("FNIRSPreprocessor - No short channels found or defined. Skipping short-channel regression.")
                except Exception as e_scr:
                    self.logger.warning(f"FNIRSPreprocessor - Short-channel regression failed: {e_scr}. Continuing without it.", exc_info=True)
            
            if motion_correction_method_config == 'tddr':
                self.logger.info("FNIRSPreprocessor - Applying TDDR motion artifact correction.")
                # TDDR might be sensitive to NaNs or Infs if present
                if np.any(np.isnan(raw_haemo.get_data())) or np.any(np.isinf(raw_haemo.get_data())):
                    self.logger.warning("FNIRSPreprocessor - NaN or Inf values found in data before TDDR. Attempting to interpolate.")
                    raw_haemo.interpolate_bads(reset_bads=True, mode='accurate', verbose=False) # General interpolation
                
                corrected_haemo = mne_nirs.temporal_derivative_distribution_repair(raw_haemo.copy())
            # Add other motion correction methods like 'savgol' if needed
            # elif motion_correction_method_config == 'savgol':
            #     self.logger.info("FNIRSPreprocessor - Applying Savitzky-Golay filter for motion artifact correction.")
            #     # Example: corrected_haemo = raw_haemo.copy().filter(..., method='savgol', h_freq=None, l_freq=None, **some_savgol_params_config)
            #     corrected_haemo = raw_haemo # Placeholder if savgol not fully implemented here
            else:
                self.logger.info("FNIRSPreprocessor - No specific motion correction method applied beyond initial processing.")
                corrected_haemo = raw_haemo # Pass through if no method or 'none'
            self.logger.info(f"FNIRSPreprocessor - Applying band-pass filter ({filter_band_config[0]}-{filter_band_config[1]} Hz).")
            corrected_haemo.filter(l_freq=filter_band_config[0], h_freq=filter_band_config[1],
                                   h_trans_bandwidth='auto', # MNE default or specify
                                   l_trans_bandwidth='auto', # MNE default or specify
                                   fir_design='firwin', verbose=False)
            
            self.logger.info("FNIRSPreprocessor - fNIRS preprocessing completed.")
            return corrected_haemo
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error during fNIRS preprocessing: {e}", exc_info=True)
            return None

    def save_preprocessed_data(self, fnirs_haemo_obj, participant_id, output_dir):
        """Saves the preprocessed fNIRS haemoglobin data to a .fif file."""
        if fnirs_haemo_obj is None:
            self.logger.warning("FNIRSPreprocessor - No processed fNIRS object to save.")
            return None
        
        try:
            output_path = os.path.join(output_dir, f"{participant_id}_fnirs_haemo_proc_raw.fif")
            fnirs_haemo_obj.save(output_path, overwrite=True, verbose=False)
            self.logger.info(f"FNIRSPreprocessor - Processed fNIRS data saved to: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"FNIRSPreprocessor - Error saving processed fNIRS data for {participant_id}: {e}", exc_info=True)
            return None