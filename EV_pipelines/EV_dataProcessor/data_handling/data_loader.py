import os
import re
import pyxdf
import mne
import numpy as np
import pandas as pd
from .. import config # Relative import

class DataLoader:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("DataLoader initialized.")

    def load_participant_streams(self, participant_id, participant_raw_data_path):
        """
        Loads relevant data streams (EEG, ECG, EDA, fNIRS, Markers) from XDF files
        for a given participant.

        Args:
            participant_id (str): The ID of the participant.
            participant_raw_data_path (str): The path to the participant's raw data directory.

        Returns:
            dict: A dictionary containing loaded MNE Raw objects or signal arrays,
                  and their sampling frequencies. Returns None for modalities not found.
        """
        self.logger.info(f"DataLoader - Loading data for participant {participant_id} from {participant_raw_data_path}")
        loaded_data = {}

        # Find the main XDF file(s) - assuming one per participant for all streams
        # You might need to adjust this if data is split across multiple files
        xdf_files = [f for f in os.listdir(participant_raw_data_path) if f.lower().endswith('.xdf')]

        if not xdf_files:
            self.logger.error(f"DataLoader - No XDF files found in {participant_raw_data_path}")
            return loaded_data # Return empty dict

        # Assuming all streams are in the first found XDF file for simplicity
        # If streams are in multiple files, you'd need more complex logic here
        xdf_path = os.path.join(participant_raw_data_path, xdf_files[0])
        self.logger.info(f"DataLoader - Found and attempting to load XDF file: {xdf_path}")

        try:
            streams, header = pyxdf.load_xdf(xdf_path)
            self.logger.info(f"DataLoader - Successfully loaded XDF file. Found {len(streams)} streams.")

            stream_map = {stream['info']['name'][0]: stream for stream in streams}

            # --- Load EEG ---
            if config.EEG_STREAM_NAME in stream_map:
                self.logger.info(f"DataLoader - Loading EEG stream: {config.EEG_STREAM_NAME}")
                stream = stream_map[config.EEG_STREAM_NAME]
                try:
                    # Assuming EEG data is in stream['time_series'] and is float/double
                    # Assuming channel names are in stream['info']['desc'][0]['channels'][0]['channel']
                    data = np.array(stream['time_series']).T # Transpose to be channels x samples
                    sfreq = float(stream['info']['nominal_srate'][0])
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['eeg'] * len(ch_names)

                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                    raw_eeg = mne.io.RawArray(data, info, verbose=False)

                    # Add annotations from marker stream if available
                    if config.MARKER_STREAM_NAME in stream_map:
                         marker_stream = stream_map[config.MARKER_STREAM_NAME]
                         marker_times = marker_stream['time_stamps']
                         marker_values = [val[0] for val in marker_stream['time_series']] # Assuming markers are strings

                         # Convert marker times to EEG time points
                         eeg_start_time = raw_eeg.times[0] + stream['time_stamps'][0] # Absolute start time of EEG stream
                         annotations = []
                         for marker_time, marker_value in zip(marker_times, marker_values):
                             # Calculate onset relative to the start of the EEG Raw object
                             onset = marker_time - eeg_start_time
                             if onset >= 0: # Only add markers that occur after the EEG recording starts
                                 annotations.append([onset, 0, marker_value]) # onset, duration (0 for point events), description

                         if annotations:
                             annotations = mne.Annotations(
                                 [a[0] for a in annotations],
                                 [a[1] for a in annotations],
                                 [a[2] for a in annotations],
                                 orig_time=eeg_start_time # Store original time for accurate alignment
                             )
                             raw_eeg.set_annotations(annotations)
                             self.logger.info(f"DataLoader - Added {len(annotations)} annotations from marker stream to EEG.")
                         else:
                             self.logger.warning("DataLoader - No valid annotations found or all markers occurred before EEG start.")

                    loaded_data['eeg'] = raw_eeg
                    self.logger.info(f"DataLoader - Loaded EEG data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing EEG stream: {e}", exc_info=True)
                    loaded_data['eeg'] = None
            else:
                self.logger.warning(f"DataLoader - EEG stream '{config.EEG_STREAM_NAME}' not found.")
                loaded_data['eeg'] = None

            # --- Load ECG ---
            if config.ECG_STREAM_NAME in stream_map:
                self.logger.info(f"DataLoader - Loading ECG stream: {config.ECG_STREAM_NAME}")
                stream = stream_map[config.ECG_STREAM_NAME]
                try:
                    # Assuming ECG is a single channel time series
                    ecg_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    ecg_sfreq = float(stream['info']['nominal_srate'][0])
                    ecg_times = stream['time_stamps'] # Absolute timestamps

                    loaded_data['ecg_signal'] = ecg_signal
                    loaded_data['ecg_sfreq'] = ecg_sfreq
                    loaded_data['ecg_times'] = ecg_times # Store absolute times for alignment
                    self.logger.info(f"DataLoader - Loaded ECG data at {ecg_sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing ECG stream: {e}", exc_info=True)
                    loaded_data['ecg_signal'] = None
                    loaded_data['ecg_sfreq'] = None
                    loaded_data['ecg_times'] = None
            else:
                self.logger.warning(f"DataLoader - ECG stream '{config.ECG_STREAM_NAME}' not found.")
                loaded_data['ecg_signal'] = None
                loaded_data['ecg_sfreq'] = None
                loaded_data['ecg_times'] = None


            # --- Load EDA ---
            if config.EDA_STREAM_NAME in stream_map:
                self.logger.info(f"DataLoader - Loading EDA stream: {config.EDA_STREAM_NAME}")
                stream = stream_map[config.EDA_STREAM_NAME]
                try:
                    # Assuming EDA is a single channel time series
                    eda_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    eda_sfreq = float(stream['info']['nominal_srate'][0])
                    eda_times = stream['time_stamps'] # Absolute timestamps

                    loaded_data['eda_signal'] = eda_signal
                    loaded_data['eda_sfreq'] = eda_sfreq
                    loaded_data['eda_times'] = eda_times # Store absolute times for alignment
                    self.logger.info(f"DataLoader - Loaded EDA data at {eda_sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing EDA stream: {e}", exc_info=True)
                    loaded_data['eda_signal'] = None
                    loaded_data['eda_sfreq'] = None
                    loaded_data['eda_times'] = None
            else:
                self.logger.warning(f"DataLoader - EDA stream '{config.EDA_STREAM_NAME}' not found.")
                loaded_data['eda_signal'] = None
                loaded_data['eda_sfreq'] = None
                loaded_data['eda_times'] = None

            # --- Load fNIRS ---
            if config.FNIRS_STREAM_NAME in stream_map:
                self.logger.info(f"DataLoader - Loading fNIRS stream: {config.FNIRS_STREAM_NAME}")
                stream = stream_map[config.FNIRS_STREAM_NAME]
                try:
                    # Assuming fNIRS data is in stream['time_series'] (samples x channels)
                    # Assuming channel names are in stream['info']['desc'][0]['channels'][0]['channel']
                    data = np.array(stream['time_series']).T # Transpose to be channels x samples
                    sfreq = float(stream['info']['nominal_srate'][0])
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    # fNIRS channels are typically 'hbo' and 'hbr' after conversion, but raw is optical density
                    # We'll treat them as 'fnirs_od' for now, preprocessing will convert
                    ch_types = ['fnirs_od'] * len(ch_names)

                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                    raw_fnirs_od = mne.io.RawArray(data, info, verbose=False)

                    # Add annotations from marker stream if available (same as EEG)
                    if config.MARKER_STREAM_NAME in stream_map and raw_eeg is None: # Only add if not already added to EEG
                         marker_stream = stream_map[config.MARKER_STREAM_NAME]
                         marker_times = marker_stream['time_stamps']
                         marker_values = [val[0] for val in marker_stream['time_series']]

                         fnirs_start_time = raw_fnirs_od.times[0] + stream['time_stamps'][0]
                         annotations = []
                         for marker_time, marker_value in zip(marker_times, marker_values):
                             onset = marker_time - fnirs_start_time
                             if onset >= 0:
                                 annotations.append([onset, 0, marker_value])

                         if annotations:
                             annotations = mne.Annotations(
                                 [a[0] for a in annotations],
                                 [a[1] for a in annotations],
                                 [a[2] for a in annotations],
                                 orig_time=fnirs_start_time
                             )
                             raw_fnirs_od.set_annotations(annotations)
                             self.logger.info(f"DataLoader - Added {len(annotations)} annotations from marker stream to fNIRS.")
                         else:
                             self.logger.warning("DataLoader - No valid annotations found or all markers occurred before fNIRS start.")


                    loaded_data['fnirs_od'] = raw_fnirs_od
                    self.logger.info(f"DataLoader - Loaded fNIRS data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing fNIRS stream: {e}", exc_info=True)
                    loaded_data['fnirs_od'] = None
            else:
                self.logger.warning(f"DataLoader - fNIRS stream '{config.FNIRS_STREAM_NAME}' not found.")
                loaded_data['fnirs_od'] = None

            # --- Handle Marker Stream (if not already attached to EEG/fNIRS) ---
            # If EEG and fNIRS were not loaded, but markers exist, we might still need them
            # for timing other modalities. However, the current structure relies on EEG/fNIRS
            # annotations for epoching. Let's ensure the marker stream is available if needed.
            if config.MARKER_STREAM_NAME in stream_map:
                 marker_stream = stream_map[config.MARKER_STREAM_NAME]
                 loaded_data['marker_times'] = marker_stream['time_stamps']
                 loaded_data['marker_values'] = [val[0] for val in marker_stream['time_series']]
                 loaded_data['marker_sfreq'] = float(marker_stream['info']['nominal_srate'][0]) # Markers usually have nominal_srate 0, but keep for consistency
                 self.logger.info(f"DataLoader - Loaded Marker stream.")


        except Exception as e:
            self.logger.error(f"DataLoader - Critical error loading or parsing XDF file {xdf_path}: {e}", exc_info=True)
            # Return empty dict if XDF loading fails completely
            return {}

        # --- Identify Baseline Period ---
        # This assumes a 'Baseline_Start' marker exists and is the first event of the session
        # Or that the baseline is a fixed duration before the first stimulus marker
        baseline_start_time = None
        baseline_end_time = None
        first_stimulus_time = None

        if 'eeg' in loaded_data and loaded_data['eeg'] is not None:
            raw_obj = loaded_data['eeg']
        elif 'fnirs_od' in loaded_data and loaded_data['fnirs_od'] is not None:
             raw_obj = loaded_data['fnirs_od']
        else:
             raw_obj = None
             self.logger.warning("DataLoader - No EEG or fNIRS data loaded, cannot determine baseline times from annotations.")


        if raw_obj is not None and raw_obj.annotations:
            events, event_id = mne.events_from_annotations(raw_obj, verbose=False)
            event_id_inv = {v: k for k, v in event_id.items()}

            # Find the first 'Baseline_Start' marker
            baseline_start_marker_name = config.BASELINE_MARKER_START
            if baseline_start_marker_name in event_id_inv:
                baseline_events = events[events[:, 2] == event_id_inv[baseline_start_marker_name]]
                if baseline_events.size > 0:
                    # Onset is in samples, raw_obj.times converts it to seconds relative to raw_obj start
                    baseline_start_time = raw_obj.times[baseline_events[0, 0]]
                    self.logger.info(f"DataLoader - Identified baseline start from marker at {baseline_start_time:.2f} s.")

            # Find the first stimulus marker
            stimulus_event_ids = [config.EVENT_ID[cond] for cond in config.MOVIE_CONDITIONS if cond in config.EVENT_ID_TO_CONDITION.values()]
            stimulus_events = events[np.isin(events[:, 2], stimulus_event_ids)]

            if stimulus_events.size > 0:
                # Onset is in samples, convert to seconds relative to raw_obj start
                first_stimulus_time = raw_obj.times[stimulus_events[0, 0]]
                self.logger.info(f"DataLoader - Identified first stimulus start at {first_stimulus_time:.2f} s.")

            # Determine baseline end time
            baseline_end_marker_name = config.BASELINE_MARKER_END
            if baseline_end_marker_name in event_id_inv:
                baseline_end_events = events[events[:, 2] == event_id_inv[baseline_end_marker_name]]
                if baseline_end_events.size > 0:
                    baseline_end_time = raw_obj.times[baseline_end_events[0, 0]]
                    self.logger.info(f"DataLoader - Identified baseline end from marker at {baseline_end_time:.2f} s.")
            
            if baseline_start_time is not None and baseline_end_time is None and first_stimulus_time is not None:
                 baseline_end_time = first_stimulus_time # Ends at first stimulus if no end marker
                 self.logger.info(f"DataLoader - Baseline end set to first stimulus time: {baseline_end_time:.2f} s.")
            elif first_stimulus_time is not None:
                 # If no explicit baseline start marker, assume baseline is fixed duration before first stimulus
                 baseline_end_time = first_stimulus_time
                 baseline_start_time = max(0, first_stimulus_time - config.BASELINE_DURATION_FALLBACK_SEC)
                 self.logger.info(f"DataLoader - Assuming baseline period from {baseline_start_time:.2f} s to {baseline_end_time:.2f} s (fixed duration before first stimulus).")
            else:
                 self.logger.warning("DataLoader - Could not identify baseline period. No 'Baseline_Start' marker or stimulus markers found.")


        # Store baseline times relative to the start of the recording (time=0 in MNE Raw)
        if baseline_start_time is not None and baseline_end_time is not None:
             loaded_data['baseline_start_time_sec'] = baseline_start_time
             loaded_data['baseline_end_time_sec'] = baseline_end_time
             self.logger.info(f"DataLoader - Final baseline period: {baseline_start_time:.2f}s to {baseline_end_time:.2f}s")
        else:
             loaded_data['baseline_start_time_sec'] = None
             loaded_data['baseline_end_time_sec'] = None


        self.logger.info(f"DataLoader - Finished loading data for participant {participant_id}.")
        return loaded_data

    def load_survey_data(self, survey_file_path, participant_id):
        """
        Loads survey data from a CSV file.
        Assumes columns like 'participant_id', 'trial_id' (or 'condition' + 'trial_num'), 'sam_valence', 'sam_arousal'.
        """
        if survey_file_path is None or not os.path.exists(survey_file_path):
            self.logger.warning(f"Survey file not found: {survey_file_path}. Skipping survey data loading.")
            return None
        try:
            df = pd.read_csv(survey_file_path)
            # Filter for current participant if survey file contains multiple participants
            if 'participant_id' in df.columns:
                df = df[df['participant_id'] == participant_id]
            
            if df.empty:
                self.logger.warning(f"No survey data found for participant {participant_id} in {survey_file_path}.")
                return None
            self.logger.info(f"Successfully loaded survey data for {participant_id} from {survey_file_path}.")
            return df
        except Exception as e:
            self.logger.error(f"Error loading survey data from {survey_file_path}: {e}", exc_info=True)
            return None