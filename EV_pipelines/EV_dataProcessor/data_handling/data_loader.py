import os
import mne
from .. import config # Relative import to access config.py

class DataLoader:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("DataLoader initialized.")

    def load_participant_streams(self, participant_id, participant_raw_data_path):
        """
        Loads raw data streams for a participant.
        Currently supports a combined LSL (BrainVision) and a separate NIRx fNIRS file.
        Returns:
            dict: Contains MNE Raw objects for 'eeg', 'ecg_signal', 'eda_signal', 'fnirs_od',
                  and their respective 'sfreq' if found. Values are None if not found.
        """
        streams = {
            'eeg': None, 'eeg_sfreq': None,
            'ecg_signal': None, 'ecg_sfreq': None,
            'eda_signal': None, 'eda_sfreq': None,
            'fnirs_od': None, 'fnirs_sfreq': None
        }

        # Load combined LSL stream (e.g., BrainVision .vhdr)
        combined_file = os.path.join(participant_raw_data_path, f"{participant_id}{config.COMBINED_LSL_STREAM_FILENAME_SUFFIX}")
        if os.path.exists(combined_file):
            self.logger.info(f"DataLoader - Loading combined LSL stream: {combined_file}")
            try:
                raw_combined = mne.io.read_raw_brainvision(combined_file, preload=True, verbose=False)
                sfreq = raw_combined.info['sfreq']
                
                # Extract EEG
                if len(raw_combined.ch_names) > config.COMBINED_EEG_START_CHANNEL_INDEX:
                    eeg_ch_names = raw_combined.ch_names[config.COMBINED_EEG_START_CHANNEL_INDEX:]
                    if eeg_ch_names:
                        streams['eeg'] = raw_combined.copy().pick_channels(eeg_ch_names, verbose=False)
                        streams['eeg_sfreq'] = sfreq
                        self.logger.info(f"DataLoader - Extracted EEG ({len(eeg_ch_names)} chans) at {sfreq} Hz.")
                
                # Extract ECG
                if len(raw_combined.ch_names) > config.COMBINED_ECG_CHANNEL_INDEX:
                    ecg_ch_name = raw_combined.ch_names[config.COMBINED_ECG_CHANNEL_INDEX]
                    streams['ecg_signal'] = raw_combined.get_data(picks=[ecg_ch_name])[0]
                    streams['ecg_sfreq'] = sfreq
                    self.logger.info(f"DataLoader - Extracted ECG ('{ecg_ch_name}') at {sfreq} Hz.")

                # Extract EDA
                if len(raw_combined.ch_names) > config.COMBINED_EDA_CHANNEL_INDEX:
                    eda_ch_name = raw_combined.ch_names[config.COMBINED_EDA_CHANNEL_INDEX]
                    streams['eda_signal'] = raw_combined.get_data(picks=[eda_ch_name])[0]
                    streams['eda_sfreq'] = sfreq
                    self.logger.info(f"DataLoader - Extracted EDA ('{eda_ch_name}') at {sfreq} Hz.")

            except Exception as e:
                self.logger.error(f"DataLoader - Error loading or parsing combined LSL file {combined_file}: {e}", exc_info=True)
        else:
            self.logger.warning(f"DataLoader - Combined LSL stream file not found: {combined_file}")

        # Load fNIRS stream (e.g., .nirs file)
        fnirs_file = os.path.join(participant_raw_data_path, f"{participant_id}{config.FNIRS_RAW_FILENAME_SUFFIX}")
        if os.path.exists(fnirs_file):
            self.logger.info(f"DataLoader - Loading fNIRS stream: {fnirs_file}")
            try:
                streams['fnirs_od'] = mne.io.read_raw_nirx(fnirs_file, preload=True, verbose=False)
                streams['fnirs_sfreq'] = streams['fnirs_od'].info['sfreq']
                self.logger.info(f"DataLoader - Loaded fNIRS OD at {streams['fnirs_sfreq']} Hz.")
            except Exception as e:
                self.logger.error(f"DataLoader - Error loading fNIRS file {fnirs_file}: {e}", exc_info=True)
                streams['fnirs_od'] = None # Ensure it's None on error
                streams['fnirs_sfreq'] = None
        else:
            self.logger.warning(f"DataLoader - fNIRS stream file not found: {fnirs_file}")
            
        # Log missing streams
        for stream_name, stream_data in streams.items():
            if stream_data is None and not stream_name.endswith('_sfreq'):
                 self.logger.warning(f"DataLoader - Stream '{stream_name}' could not be loaded or was not found.")

        return streams