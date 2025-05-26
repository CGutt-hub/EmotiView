import os
import re
import pyxdf
import mne
import numpy as np
import pandas as pd
from ..utils.event_processor import EventProcessor # Import EventProcessor
# No direct import of EV_config here; config values will be passed in.

class DataLoader:
    def __init__(self, logger,
                 # Config values passed from EV_analyzer.py (sourced from EV_config.py)
                 eeg_stream_name="DefaultEEGStreamName", # Provide defaults or make them required
                 fnirs_stream_name="DefaultFNIRSStreamName",
                 ecg_stream_name="DefaultECGStreamName",
                 eda_stream_name="DefaultEDAStreamName",
                 marker_stream_name="DefaultMarkerStreamName",
                 baseline_marker_start_eprime="Baseline_Start", # For EventProcessor
                 baseline_marker_end_eprime="Baseline_End",     # For EventProcessor
                 event_duration_default=4.0,                  # For EventProcessor
                 emotional_conditions_list=None,
                 baseline_duration_fallback_sec=60.0
                ):
        self.logger = logger
        # Store configuration
        self._eeg_stream_name = eeg_stream_name
        self._fnirs_stream_name = fnirs_stream_name
        self._ecg_stream_name = ecg_stream_name
        self._eda_stream_name = eda_stream_name
        self._marker_stream_name = marker_stream_name
        self._baseline_marker_start_eprime = baseline_marker_start_eprime
        self._baseline_marker_end_eprime = baseline_marker_end_eprime
        self._event_duration_default = event_duration_default
        self._emotional_conditions_list = emotional_conditions_list if emotional_conditions_list is not None else ['Positive', 'Negative', 'Neutral']
        self._baseline_duration_fallback_sec = baseline_duration_fallback_sec
        self.logger.info("DataLoader initialized.")
    def load_participant_streams(self, participant_id, participant_raw_data_path, eprime_txt_file_path=None):
        """
        Loads relevant data streams (EEG, ECG, EDA, fNIRS, Markers) from XDF files
        for a given participant.

        Args:
            participant_id (str): The ID of the participant.
            participant_raw_data_path (str): The path to the participant's raw data directory.
            eprime_txt_file_path (str, optional): Path to the E-Prime .txt file for metadata.

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
            raw_eeg = None # Initialize to handle cases where EEG is not loaded first

            # --- Load EEG ---
            if self._eeg_stream_name in stream_map:
                self.logger.info(f"DataLoader - Loading EEG stream: {self._eeg_stream_name}")
                stream = stream_map[self._eeg_stream_name]
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
                    if self._marker_stream_name in stream_map:
                         marker_stream = stream_map[self._marker_stream_name]
                         marker_times = marker_stream['time_stamps']
                         marker_values = [val[0] for val in marker_stream['time_series']] # Assuming markers are strings

                         # Convert marker times to EEG time points
                         # XDF timestamps are absolute. MNE annotations need to be relative to the raw object's start.
                         # The first_samp of the raw object is 0. Its orig_time is the absolute start time of the stream.
                         # We need to find the absolute start time of the EEG stream from XDF.
                         eeg_stream_start_time_xdf = stream['time_stamps'][0] # Absolute start time of this EEG stream
                         loaded_data['eeg_stream_start_time_xdf'] = eeg_stream_start_time_xdf

                         annotations_list = []
                         for marker_time, marker_value in zip(marker_times, marker_values):
                             # Calculate onset relative to the start of the EEG Raw object
                             onset_in_eeg_time = marker_time - eeg_stream_start_time_xdf
                             if onset_in_eeg_time >= 0: # Only add markers that occur at or after the EEG recording starts
                                 annotations_list.append([onset_in_eeg_time, 0, marker_value]) # onset, duration (0 for point events), description
                             else:
                                 self.logger.debug(f"Marker '{marker_value}' at {marker_time} occurred before EEG stream start ({eeg_stream_start_time_xdf}). Skipping.")
                         
                         if annotations_list:
                             onsets = [a[0] for a in annotations_list]
                             durations = [a[1] for a in annotations_list]
                             descriptions = [a[2] for a in annotations_list]
                             
                             # MNE Annotations orig_time should be the time of the first sample in the raw data,
                             # which is effectively 0 for RawArray unless you set first_samp.
                             # The onsets are already relative to the start of this EEG stream.
                             mne_annotations = mne.Annotations(
                                 onset=onsets,
                                 duration=durations,
                                 description=descriptions
                                 # orig_time can be set if you want to keep track of absolute XDF time,
                                 # but MNE operations typically use relative times.
                                 # For RawArray, if first_samp is 0, onsets are relative to t=0 of the Raw.
                             )
                             raw_eeg.set_annotations(mne_annotations)
                             self.logger.info(f"DataLoader - Added {len(mne_annotations)} annotations from marker stream to EEG.")
                         else:
                             self.logger.warning("DataLoader - No valid annotations found or all markers occurred before EEG start.")

                    loaded_data['eeg'] = raw_eeg
                    self.logger.info(f"DataLoader - Loaded EEG data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing EEG stream: {e}", exc_info=True)
                    loaded_data['eeg'] = None
            else:
                self.logger.warning(f"DataLoader - EEG stream '{self._eeg_stream_name}' not found.")
                loaded_data['eeg'] = None

            # --- Load ECG ---
            if self._ecg_stream_name in stream_map:
                self.logger.info(f"DataLoader - Loading ECG stream: {self._ecg_stream_name}")
                stream = stream_map[self._ecg_stream_name]
                try:
                    # Assuming ECG is a single channel time series
                    ecg_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    ecg_sfreq = float(stream['info']['nominal_srate'][0])
                    ecg_times = stream['time_stamps'] # Absolute timestamps from XDF

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
                self.logger.warning(f"DataLoader - ECG stream '{self._ecg_stream_name}' not found.")
                loaded_data['ecg_signal'] = None
                loaded_data['ecg_sfreq'] = None
                loaded_data['ecg_times'] = None


            # --- Load EDA ---
            if self._eda_stream_name in stream_map:
                self.logger.info(f"DataLoader - Loading EDA stream: {self._eda_stream_name}")
                stream = stream_map[self._eda_stream_name]
                try:
                    # Assuming EDA is a single channel time series
                    eda_signal = np.array(stream['time_series']).flatten() # Ensure 1D
                    eda_sfreq = float(stream['info']['nominal_srate'][0])
                    eda_times = stream['time_stamps'] # Absolute timestamps from XDF

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
                self.logger.warning(f"DataLoader - EDA stream '{self._eda_stream_name}' not found.")
                loaded_data['eda_signal'] = None
                loaded_data['eda_sfreq'] = None
                loaded_data['eda_times'] = None

            # --- Load fNIRS ---
            if self._fnirs_stream_name in stream_map:
                self.logger.info(f"DataLoader - Loading fNIRS stream: {self._fnirs_stream_name}")
                stream = stream_map[self._fnirs_stream_name]
                try:
                    data = np.array(stream['time_series']).T 
                    sfreq = float(stream['info']['nominal_srate'][0])
                    ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
                    ch_types = ['fnirs_od'] * len(ch_names)

                    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                    raw_fnirs_od = mne.io.RawArray(data, info, verbose=False)
                    fnirs_stream_start_time_xdf = stream['time_stamps'][0]
                    loaded_data['fnirs_stream_start_time_xdf'] = fnirs_stream_start_time_xdf


                    # Add annotations from marker stream if available (only if EEG was not loaded or had no markers)
                    if self._marker_stream_name in stream_map and (raw_eeg is None or not raw_eeg.annotations):
                         marker_stream = stream_map[self._marker_stream_name]
                         marker_times = marker_stream['time_stamps']
                         marker_values = [val[0] for val in marker_stream['time_series']]

                         annotations_list_fnirs = []
                         for marker_time, marker_value in zip(marker_times, marker_values):
                             onset_in_fnirs_time = marker_time - fnirs_stream_start_time_xdf
                             if onset_in_fnirs_time >= 0:
                                 annotations_list_fnirs.append([onset_in_fnirs_time, 0, marker_value])

                         if annotations_list_fnirs:
                             mne_annotations_fnirs = mne.Annotations(
                                 onset=[a[0] for a in annotations_list_fnirs],
                                 duration=[a[1] for a in annotations_list_fnirs],
                                 description=[a[2] for a in annotations_list_fnirs]
                             )
                             raw_fnirs_od.set_annotations(mne_annotations_fnirs)
                             self.logger.info(f"DataLoader - Added {len(mne_annotations_fnirs)} annotations from marker stream to fNIRS.")
                         else:
                             self.logger.warning("DataLoader - No valid annotations found for fNIRS or all markers occurred before fNIRS start.")


                    loaded_data['fnirs_od'] = raw_fnirs_od
                    self.logger.info(f"DataLoader - Loaded fNIRS data with {len(ch_names)} channels at {sfreq} Hz.")

                except Exception as e:
                    self.logger.error(f"DataLoader - Error loading or processing fNIRS stream: {e}", exc_info=True)
                    loaded_data['fnirs_od'] = None
            else:
                self.logger.warning(f"DataLoader - fNIRS stream '{self._fnirs_stream_name}' not found.")
                loaded_data['fnirs_od'] = None

            # --- Process Marker Stream for events_df ---
            xdf_marker_events_df = pd.DataFrame()
            if self._marker_stream_name in stream_map:
                 marker_stream = stream_map[self._marker_stream_name]
                 loaded_data['marker_times'] = marker_stream['time_stamps'] # Absolute XDF times
                 loaded_data['marker_values'] = [val[0] for val in marker_stream['time_series']]
                 # Markers usually have nominal_srate 0, but store if present.
                 loaded_data['marker_sfreq'] = float(marker_stream['info']['nominal_srate'][0]) if marker_stream['info']['nominal_srate'] else 0 
                 self.logger.info(f"DataLoader - Loaded Marker stream (raw XDF times).")

                # Construct events_df from XDF markers
                 xdf_marker_events_df = pd.DataFrame({
                     'onset_time_sec': loaded_data['marker_times'], # Use XDF timestamps directly
                     'condition_marker': loaded_data['marker_values'], # Raw marker values
                     'duration_sec': self._event_duration_default # Default duration, can be refined
                 })
                 self.logger.info(f"DataLoader - Created initial events_df from XDF markers ({len(xdf_marker_events_df)} events).")
            else:
                self.logger.warning(f"DataLoader - Marker stream '{self._marker_stream_name}' not found. Cannot create events_df from XDF markers.")

            # --- Process E-Prime Log for Metadata and Merge with XDF Events ---
            eprime_metadata_df = pd.DataFrame()
            if eprime_txt_file_path and os.path.exists(eprime_txt_file_path):
                try:
                    event_processor_instance = EventProcessor(
                        logger=self.logger,
                        baseline_marker_start_eprime=self._baseline_marker_start_eprime,
                        baseline_marker_end_eprime=self._baseline_marker_end_eprime,
                        event_duration_default=self._event_duration_default
                    )
                    eprime_metadata_df = event_processor_instance.process_event_log(eprime_txt_file_path)
                    self.logger.info(f"DataLoader - Extracted {len(eprime_metadata_df)} events from E-Prime log for metadata.")

                    if not xdf_marker_events_df.empty and not eprime_metadata_df.empty:
                        # Attempt to merge based on sequence if lengths are compatible
                        # This is a simplified merge; robust merging might need more sophisticated logic
                        # (e.g., matching specific marker patterns or trial counts per block)
                        if len(xdf_marker_events_df) >= len(eprime_metadata_df): # Allow XDF to have more markers (e.g. baseline)
                            self.logger.info("DataLoader - Attempting to merge XDF timing with E-Prime metadata.")
                            # Create a temporary DataFrame from XDF markers that are likely to match E-Prime trials
                            # This is heuristic: assumes E-Prime trials are a subset or match XDF markers
                            
                            # For now, let's assume a simple merge for trials that have movie_filename
                            # We need a way to link XDF markers to E-Prime trials.
                            # If E-Prime sends specific markers for each trial start that match XDF:
                            #   1. Filter xdf_marker_events_df for these trial start markers.
                            #   2. Align with eprime_metadata_df based on sequence.
                            # If not, this becomes very tricky.
                            
                            # Simplistic approach: Use XDF for timing, try to map E-Prime conditions
                            # This assumes that the `condition_marker` from XDF can be used to infer
                            # the E-Prime condition or that the sequence aligns.
                            # A more robust solution would involve specific sync markers.
                            
                            # For now, let's prioritize XDF timing and use E-Prime for condition names if possible.
                            # This part needs careful validation based on your specific marker strategy.
                            # If E-Prime sends LSL markers that are identical to what EventProcessor derives as 'condition',
                            # you could merge on that.
                            
                            # Placeholder: For now, we'll just use the XDF events and log a warning
                            # if E-Prime metadata couldn't be cleanly merged.
                            self.logger.warning("DataLoader - E-Prime metadata merging with XDF timed events is complex and needs specific logic based on your marker strategy. Using XDF marker events for now.")
                            loaded_data['events_df'] = xdf_marker_events_df.rename(columns={'condition_marker': 'condition'})
                        else:
                            self.logger.warning("DataLoader - Fewer XDF markers than E-Prime events. Using XDF markers only.")
                            loaded_data['events_df'] = xdf_marker_events_df.rename(columns={'condition_marker': 'condition'})
                    elif not xdf_marker_events_df.empty:
                        loaded_data['events_df'] = xdf_marker_events_df.rename(columns={'condition_marker': 'condition'})
                    else: # No XDF markers, E-Prime metadata exists but times are relative
                        self.logger.error("DataLoader - No XDF markers found. E-Prime event times are relative and cannot be used directly without synchronization. No events_df created.")
                        loaded_data['events_df'] = pd.DataFrame()
                except Exception as e_eprime:
                    self.logger.error(f"DataLoader - Error processing E-Prime log or merging events: {e_eprime}", exc_info=True)
                    if not xdf_marker_events_df.empty: # Fallback to XDF if E-Prime fails
                        loaded_data['events_df'] = xdf_marker_events_df.rename(columns={'condition_marker': 'condition'})
            elif not xdf_marker_events_df.empty:
                self.logger.info("DataLoader - No E-Prime log provided. Using XDF markers for events.")
                loaded_data['events_df'] = xdf_marker_events_df.rename(columns={'condition_marker': 'condition'})
            else:
                self.logger.warning("DataLoader - No E-Prime log and no XDF markers. No events_df created.")
                loaded_data['events_df'] = pd.DataFrame()

        except Exception as e:
            self.logger.error(f"DataLoader - Critical error loading or parsing XDF file {xdf_path}: {e}", exc_info=True)
            return {}

        # --- Identify Baseline Period from Annotations ---
        baseline_start_time_abs = None
        baseline_end_time_abs = None

        # Prioritize EEG for annotations, then fNIRS
        annotated_raw_obj = None
        if loaded_data.get('eeg') and loaded_data['eeg'].annotations:
            annotated_raw_obj = loaded_data['eeg']
            # Get the absolute start time of this MNE Raw object from XDF
            # This assumes EEG_STREAM_NAME was found and 'time_stamps' exists for it
            if self._eeg_stream_name in stream_map:
                annotated_raw_obj_start_time_xdf = stream_map[self._eeg_stream_name]['time_stamps'][0]
            else: # Should not happen if raw_eeg exists
                annotated_raw_obj_start_time_xdf = 0 
                self.logger.warning("Could not determine absolute start time for EEG stream from XDF.")

        elif loaded_data.get('fnirs_od') and loaded_data['fnirs_od'].annotations:
            annotated_raw_obj = loaded_data['fnirs_od']
            if self._fnirs_stream_name in stream_map:
                annotated_raw_obj_start_time_xdf = stream_map[self._fnirs_stream_name]['time_stamps'][0]
            else:
                annotated_raw_obj_start_time_xdf = 0
                self.logger.warning("Could not determine absolute start time for fNIRS stream from XDF.")
        else:
             self.logger.warning("DataLoader - No EEG or fNIRS data with annotations loaded, cannot determine baseline times from XDF markers.")

        if annotated_raw_obj:
            # MNE annotations onsets are relative to the start of the Raw object.
            # We need to convert them to absolute XDF time to be consistent.
            for ann in annotated_raw_obj.annotations:
                ann_onset_relative = ann['onset']
                ann_desc = ann['description']
                ann_onset_absolute = annotated_raw_obj_start_time_xdf + ann_onset_relative

                if ann_desc == self._baseline_marker_start_eprime: # Use configured marker name
                    if baseline_start_time_abs is None or ann_onset_absolute < baseline_start_time_abs: # Take the earliest
                        baseline_start_time_abs = ann_onset_absolute
                elif ann_desc == self._baseline_marker_end_eprime: # Use configured marker name
                    if baseline_end_time_abs is None or ann_onset_absolute > baseline_end_time_abs: # Take the latest
                        baseline_end_time_abs = ann_onset_absolute
            
            if baseline_start_time_abs is not None:
                self.logger.info(f"DataLoader - Identified baseline start from XDF marker at absolute time {baseline_start_time_abs:.2f} s.")
            if baseline_end_time_abs is not None:
                self.logger.info(f"DataLoader - Identified baseline end from XDF marker at absolute time {baseline_end_time_abs:.2f} s.")

            # Fallback for baseline end if only start marker is present
            if baseline_start_time_abs is not None and baseline_end_time_abs is None:
                # Find the earliest stimulus onset after baseline_start_time_abs
                earliest_stim_after_baseline_abs = float('inf')
                # Use EMOTIONAL_CONDITIONS directly if they represent your stimulus markers
                for ann in annotated_raw_obj.annotations:
                    if ann['description'] in self._emotional_conditions_list: # Check if it's a stimulus condition
                        stim_onset_abs = annotated_raw_obj_start_time_xdf + ann['onset']
                        if stim_onset_abs > baseline_start_time_abs:
                            earliest_stim_after_baseline_abs = min(earliest_stim_after_baseline_abs, stim_onset_abs)
                
                if earliest_stim_after_baseline_abs != float('inf'):
                    baseline_end_time_abs = earliest_stim_after_baseline_abs
                    self.logger.info(f"DataLoader - Baseline end set to first stimulus time (absolute): {baseline_end_time_abs:.2f} s.")
                else: # No stimulus found after baseline start, use fallback duration
                    baseline_end_time_abs = baseline_start_time_abs + self._baseline_duration_fallback_sec
                    self.logger.info(f"DataLoader - No stimulus after baseline start. Baseline end set by fallback duration (absolute): {baseline_end_time_abs:.2f} s.")
            
            # If no baseline markers at all, but stimulus markers exist, try to define baseline before first stimulus
            elif baseline_start_time_abs is None and baseline_end_time_abs is None:
                earliest_stim_abs = float('inf')
                for ann in annotated_raw_obj.annotations:
                    if ann['description'] in self._emotional_conditions_list:
                        stim_onset_abs = annotated_raw_obj_start_time_xdf + ann['onset']
                        earliest_stim_abs = min(earliest_stim_abs, stim_onset_abs)
                
                if earliest_stim_abs != float('inf'):
                    baseline_end_time_abs = earliest_stim_abs
                    baseline_start_time_abs = max(annotated_raw_obj_start_time_xdf, baseline_end_time_abs - self._baseline_duration_fallback_sec)
                    self.logger.info(f"DataLoader - No baseline markers. Baseline defined as fixed duration before first stimulus (absolute): {baseline_start_time_abs:.2f}s to {baseline_end_time_abs:.2f}s.")
                else:
                    self.logger.warning("DataLoader - Could not identify baseline period from XDF markers.")

        # Store absolute baseline times
        loaded_data['baseline_start_time_sec'] = baseline_start_time_abs
        loaded_data['baseline_end_time_sec'] = baseline_end_time_abs
        if baseline_start_time_abs is not None and baseline_end_time_abs is not None:
             self.logger.info(f"DataLoader - Final absolute baseline period: {baseline_start_time_abs:.2f}s to {baseline_end_time_abs:.2f}s")

        self.logger.info(f"DataLoader - Finished loading data for participant {participant_id}.")
        return loaded_data
    def load_survey_data(self, survey_file_path, participant_id):
        """
        Parses survey data from an E-Prime .txt file using QuestionnaireParser
        and returns the per-trial ratings DataFrame.
        """
        # This method now uses QuestionnaireParser, which is assumed to be in the same directory.
        # If QuestionnaireParser is moved to psyModuleToolbox.utils, the import at the top needs to change.
        # For now, assuming it's co-located or correctly imported.
        
        # Check if QuestionnaireParser is available (it might be a separate file)
        try:
            from .questionnaire_parser import QuestionnaireParser
        except ImportError:
            self.logger.error("DataLoader: QuestionnaireParser class not found. Cannot load survey data from E-Prime.")
            return None

        if survey_file_path is None or not os.path.exists(survey_file_path):
            self.logger.warning(f"Survey file not found: {survey_file_path}. Skipping survey data loading.")
            return None
        try:
            parser = QuestionnaireParser(self.logger) # Instantiate the parser
            parsed_questionnaire_data = parser.parse_eprime_file(survey_file_path) # Parse the file

            # Check if parsing was successful and returned the expected key
            if not isinstance(parsed_questionnaire_data, dict) or 'per_trial_ratings' not in parsed_questionnaire_data:
                self.logger.warning(f"QuestionnaireParser did not return 'per_trial_ratings' for {participant_id} from {survey_file_path}.")
                return None

            # Get the per-trial DataFrame
            per_trial_df = parsed_questionnaire_data['per_trial_ratings']

            if per_trial_df.empty:
                self.logger.warning(f"No per-trial survey data (SAM, etc.) extracted for participant {participant_id} from {survey_file_path}.")
                return None
            
            # Basic validation: Check for essential columns needed for merging
            required_survey_cols = ['participant_id', 'condition', 'trial_identifier_eprime', 'sam_arousal']
            missing_survey_cols = [col for col in required_survey_cols if col not in per_trial_df.columns]
            if missing_survey_cols:
                self.logger.error(f"Parsed survey data for {participant_id} is missing required columns for WP2 merge: {missing_survey_cols}. Skipping survey data loading.")
                # Optionally save the incomplete df for debugging: per_trial_df.to_csv(os.path.join(os.path.dirname(survey_file_path), f"{participant_id}_parsed_survey_incomplete.csv"))
                return None
            
            # Ensure participant_id column is consistent with the main pipeline's participant_id
            # The parser attempts to set it, but we can override/confirm here.
            # This assumes the `participant_id` argument to this function is the ground truth.
            per_trial_df['participant_id'] = participant_id 

            self.logger.info(f"Successfully parsed per-trial survey data for {participant_id} from {survey_file_path}. Found {len(per_trial_df)} trials with required columns.")
            # For WP2, we primarily need the per_trial_ratings.
            # Other data (BIS/BAS, PANAS) could be returned or handled separately if needed.
            return per_trial_df
        except Exception as e:
            self.logger.error(f"Error loading survey data from {survey_file_path}: {e}", exc_info=True)
            return None