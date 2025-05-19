# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\data_handling\questionnaire_parser.py
import pandas as pd
import re

class QuestionnaireParser:
    def __init__(self, logger):
        self.logger = logger
        self.participant_id = None
        self.parsed_data = [] # List of dicts, each dict is a trial's survey data
        self._current_movie_trial_info = {}

    def _extract_header_info(self, header_block):
        for line in header_block:
            if line.startswith("Subject:"):
                self.participant_id = line.split(":")[1].strip()
                break
        if not self.participant_id:
            self.logger.warning("Participant ID not found in header.")

    def _parse_log_frame(self, frame_block):
        frame_data = {}
        for line in frame_block:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                frame_data[key] = value

        procedure = frame_data.get("Procedure")

        if procedure == "movieTrialProc":
            # Store movie info, reset SAM for this trial
            self._current_movie_trial_info = {
                'participant_id': self.participant_id,
                'movie_filename': frame_data.get("movieFilename"),
                'condition': self._derive_condition(frame_data.get("movieFilename")),
                # Use a robust trial identifier from E-Prime log
                'trial_identifier_eprime': f"C{frame_data.get('movieList.Cycle', 'N/A')}_S{frame_data.get('movieList.Sample', 'N/A')}",
                'sam_valence': pd.NA, # Initialize
                'sam_arousal': pd.NA, # Initialize
                # Add other per-trial ratings if needed (e.g., familiarity)
                'familiarity': frame_data.get("familiarityScore.Choice1.Value")
            }
            # Add this trial to parsed_data immediately; SAM will update it later if found
            # Or, hold it and only add once SAM is collected. For simplicity here, let's assume SAM follows directly.
            self.logger.debug(f"Movie trial encountered: {self._current_movie_trial_info}")

        elif procedure == "samProc" and self._current_movie_trial_info: # SAM rating for the current movie
            sam_type_img = frame_data.get("samBackgroundImg")
            sam_value = pd.to_numeric(frame_data.get("SAM.Choice1.Value"), errors='coerce')

            if "samValence.png" in sam_type_img:
                self._current_movie_trial_info['sam_valence'] = sam_value
            elif "samArousal.png" in sam_type_img:
                self._current_movie_trial_info['sam_arousal'] = sam_value
                # Assuming Arousal SAM is the last rating for a movie trial,
                # now we can append the completed trial info.
                # A more robust way is to check if both valence and arousal are filled
                # or if the next LogFrame is a new movieTrialProc.
                if 'sam_valence' in self._current_movie_trial_info and not pd.isna(self._current_movie_trial_info['sam_valence']):
                    self.parsed_data.append(self._current_movie_trial_info.copy())
                    self.logger.debug(f"SAM Arousal recorded for trial: {self._current_movie_trial_info}")
                    self._current_movie_trial_info = {} # Reset for next movie trial
            # Add other questionnaire parsing logic here (BIS/BAS, PANAS, EA11, BE7)
            # These might be parsed into separate lists/dataframes or added to a general one
            # depending on whether they are trial-locked or session-level.

    def _derive_condition(self, movie_filename):
        if not movie_filename:
            return "Unknown"
        if "NEG" in movie_filename.upper():
            return "Negative"
        elif "NEU" in movie_filename.upper():
            return "Neutral"
        elif "POS" in movie_filename.upper():
            return "Positive"
        elif "TRAI" in movie_filename.upper(): # Training
            return "Training"
        return "Other"

    def parse_eprime_file(self, filepath):
        self.logger.info(f"Parsing E-Prime file: {filepath}")
        self.parsed_data = [] # Reset for new file
        self.participant_id = None
        self._current_movie_trial_info = {}

        try:
            with open(filepath, 'r', encoding='utf-16-le', errors='ignore') as f: # E-Prime often uses utf-16-le
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Could not read file {filepath} with utf-16-le or utf-8: {e}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error opening or reading file {filepath}: {e}")
            return pd.DataFrame()


        header_match = re.search(r"\*\*\* Header Start \*\*\*(.*?)\*\*\* Header End \*\*\*", content, re.DOTALL)
        if header_match:
            header_block_str = header_match.group(1)
            self._extract_header_info(header_block_str.strip().split('\n'))
        else:
            self.logger.warning("Could not find E-Prime header block.")
            # Try to get participant ID from filename if header fails
            basename = os.path.basename(filepath)
            match_pid = re.search(r'EV_P(\d+)', basename, re.IGNORECASE)
            if match_pid:
                self.participant_id = match_pid.group(1)
                self.logger.info(f"Extracted participant ID '{self.participant_id}' from filename.")


        log_frames = re.findall(r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*", content, re.DOTALL)
        
        for frame_str in log_frames:
            self._parse_log_frame(frame_str.strip().split('\n'))
        
        # Ensure any lingering _current_movie_trial_info (e.g., if last trial had valence but no arousal) is handled
        # This part needs careful thought based on exact E-Prime structure.
        # If SAM arousal always follows SAM valence for a movie, the current logic is okay.

        df = pd.DataFrame(self.parsed_data)
        # Filter out rows where essential movie trial info might be missing
        df = df.dropna(subset=['participant_id', 'movie_filename', 'trial_identifier_eprime', 'condition'])
        self.logger.info(f"Parsed {len(df)} movie trials with SAM ratings.")
        return df

    # ... (add methods to parse BIS/BAS, PANAS, etc., if they are not trial-locked
    #      or if they need to be aggregated differently)