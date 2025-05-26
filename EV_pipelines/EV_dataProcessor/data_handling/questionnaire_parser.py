# d:\repoShaggy\EmotiView\EV_pipelines\EV_dataProcessor\data_handling\questionnaire_parser.py
import pandas as pd
import re
import os # For basename and path joining
import logging # Import logging for the test block

class QuestionnaireParser:
    def __init__(self, logger):
        self.logger = logger
        self.participant_id = None
        self.per_trial_data_list = []
        self.bis_bas_data_list = []
        self.panas_data_list = []
        self._current_movie_trial_context = {} # Holds data for the current movie trial being processed

    def _extract_header_info(self, header_block):
        """Extracts participant ID from the E-Prime header."""
        self.participant_id = None # Reset for each file
        for line in header_block:
            if line.strip().startswith("Subject:"):
                try:
                    self.participant_id = line.split(":", 1)[1].strip()
                    # Clean up participant ID, e.g., remove leading zeros if they are not part of the ID scheme
                    # If IDs are like "005", this will make it "5". 
                    # If they are "P005", and you want "P005", this logic is fine.
                    # If they are "EV_P005" and you want "005", you'd need more specific parsing.
                    # For now, this handles simple numeric IDs or alphanumeric IDs as is.
                    # The orchestrator usually normalizes the ID based on folder names.
                    if self.participant_id.isdigit():
                        self.participant_id = str(int(self.participant_id)) 
                    # If it's like EV_P005, and you want 005, more specific regex might be needed.
                    # For now, this handles simple numeric IDs.
                except IndexError:
                    self.logger.warning("QuestionnaireParser: Could not parse Subject ID line.")
                break
        if self.participant_id is None: # Check explicitly for None
            self.logger.warning("QuestionnaireParser: Participant ID not found in E-Prime header.")

    def _finalize_current_movie_trial(self):
        """Appends the completed current movie trial context to the list."""
        # Only finalize if there's an active context with a movie filename
        if self._current_movie_trial_context and self._current_movie_trial_context.get('movie_filename'):
            self.logger.debug(f"QuestionnaireParser: Finalizing trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}")
            # Ensure all expected keys are present, fill with NA or empty lists if not
            expected_keys = ['participant_id', 'movie_filename', 'condition', 
                             'trial_identifier_eprime', 'familiarity_rating',
                             'sam_valence', 'sam_arousal', 'ea11_ratings', 'be7_ratings']
            for key in expected_keys:
                if key not in self._current_movie_trial_context:
                    if key.endswith('_ratings'): # For list-based ratings
                        self._current_movie_trial_context[key] = []
                    else:
                        self._current_movie_trial_context[key] = pd.NA
            
            # Append a copy of the context and reset for the next trial
            self.per_trial_data_list.append(self._current_movie_trial_context.copy())
            self.logger.debug(f"QuestionnaireParser: Finalized and appended trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}")
        self._current_movie_trial_context = {} # Reset for the next movie trial
        self.logger.debug("QuestionnaireParser: Resetting movie trial context.")

    def _parse_log_frame(self, frame_block):
        """Parses a single LogFrame block."""
        frame_data = {}
        for line in frame_block:
            # Split only on the first colon to handle values that might contain colons
            parts = line.split(":", 1) 
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                frame_data[key] = value
            # Handle lines without a colon if necessary (e.g., blank lines)
            elif parts[0].strip(): # If there's content but no colon, log it
                 self.logger.debug(f"QuestionnaireParser: Skipping line without colon in log frame: {line.strip()}")


        procedure = frame_data.get("Procedure")
        level = frame_data.get("Level", "N/A") # Get Level, default to N/A if not found
        
        # --- Level 2 Procedures ---
        if level == '2':
            # Check if this is a new movie trial procedure
            if procedure in ["movieTrialProc", "movieTrainingProc"]:
                self._finalize_current_movie_trial() # Finalize previous movie trial data if any

                movie_filename = frame_data.get("movieFilename")
                # Use both Cycle and Sample for a more robust identifier. Handle potential missing keys.
                movie_cycle = frame_data.get("movieList.Cycle", frame_data.get("movieTraining.Cycle", "N/A")) 
                movie_sample = frame_data.get("movieList.Sample", frame_data.get("movieTraining.Sample", "N/A"))
                
                self._current_movie_trial_context = {
                    'participant_id': self.participant_id, # Will be set from header or filename
                    'movie_filename': movie_filename,
                    'condition': self._derive_condition(movie_filename),
                    'trial_identifier_eprime': f"C{movie_cycle}_S{movie_sample}", # Composite ID
                    'familiarity_rating': pd.to_numeric(frame_data.get("familiarityScore.Choice1.Value"), errors='coerce'),
                    'sam_valence': pd.NA, # Initialize
                    'sam_arousal': pd.NA, # Initialize
                    'ea11_ratings': [], # List to store multiple EA11 items for this trial
                    'be7_ratings': []   # List to store multiple BE7 items for this trial
                }
                self.logger.debug(f"QuestionnaireParser: Started movie trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}")

            # Session-level questionnaires (BIS/BAS, PANAS) are also at Level 2 in your file
            elif procedure == "bisBasProc":
                self.bis_bas_data_list.append({
                    'participant_id': self.participant_id,
                    'item_text': frame_data.get("bis"),
                    'item_id_eprime': frame_data.get("bisBasList"), # Original item number from E-Prime list
                    'item_sample_order': pd.to_numeric(frame_data.get("bisBasList.Sample"), errors='coerce'), # Order presented
                    'value': pd.to_numeric(frame_data.get("bisBas.Choice1.Value"), errors='coerce'),
                    'rt': pd.to_numeric(frame_data.get("bisBas.RT"), errors='coerce')
                })
                self.logger.debug(f"QuestionnaireParser: Parsed BIS/BAS item: {frame_data.get('bisBasList.Sample')}")

            elif procedure == "panasProc":
                self.panas_data_list.append({
                    'participant_id': self.participant_id,
                    'item_text': frame_data.get("panas"),
                    'item_id_eprime': frame_data.get("panasList"), # Original item number
                    'item_sample_order': pd.to_numeric(frame_data.get("panasList.Sample"), errors='coerce'), # Order presented
                    'value': pd.to_numeric(frame_data.get("panas.Choice1.Value"), errors='coerce'),
                    'rt': pd.to_numeric(frame_data.get("panas.RT"), errors='coerce')
                })
                self.logger.debug(f"QuestionnaireParser: Parsed PANAS item: {frame_data.get('panasList.Sample')}")

        # --- Level 3 Procedures (Ratings associated with a movie trial) ---
        elif level == '3' and self._current_movie_trial_context: # Check if any movie context is active
            # Check if we are currently inside a movie trial context
            
            if procedure == "samProc":
                sam_type_img = frame_data.get("samBackgroundImg")
                sam_value = pd.to_numeric(frame_data.get("SAM.Choice1.Value"), errors='coerce')
                
                if sam_type_img: # Ensure samBackgroundImg exists
                    if "samValence.png" in sam_type_img:
                        self._current_movie_trial_context['sam_valence'] = sam_value
                        self.logger.debug(f"QuestionnaireParser: Added SAM Valence ({sam_value}) to current trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}.")
                    elif "samArousal.png" in sam_type_img:
                        self._current_movie_trial_context['sam_arousal'] = sam_value
                        self.logger.debug(f"QuestionnaireParser: Added SAM Arousal ({sam_value}) to current trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}.")
                    else:
                         self.logger.warning(f"QuestionnaireParser: Unknown SAM image type: {sam_type_img}")
                else:
                     self.logger.warning("QuestionnaireParser: samBackgroundImg not found in samProc frame.")


            elif procedure == "ea11Proc":
                self._current_movie_trial_context['ea11_ratings'].append({
                    'adjective': frame_data.get("adjective"),
                    'value': pd.to_numeric(frame_data.get("ea11.Choice1.Value"), errors='coerce'),
                    'rt': pd.to_numeric(frame_data.get("ea11.RT"), errors='coerce')
                })
                self.logger.debug(f"QuestionnaireParser: Added EA11 rating to current trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}.")

            elif procedure == "be7Proc":
                self._current_movie_trial_context['be7_ratings'].append({
                    'emotion': frame_data.get("emotion"),
                    'value': pd.to_numeric(frame_data.get("be7.Choice1.Value"), errors='coerce'),
                    'rt': pd.to_numeric(frame_data.get("be7.RT"), errors='coerce')
                })
                self.logger.debug(f"QuestionnaireParser: Added BE7 rating to current trial context for {self._current_movie_trial_context.get('trial_identifier_eprime')}.")
        
        # --- Other Levels or Procedures ---
        # You can add logic here to handle other LogFrame types if needed
        # elif procedure == "someOtherProc":
        #     pass # Handle other data

    def _derive_condition(self, movie_filename):
        """Derives condition name from movie filename."""
        if not movie_filename:
            return "Unknown"
        name_upper = movie_filename.upper()
        if "NEG" in name_upper:
            return "Negative"
        elif "NEU" in name_upper:
            return "Neutral"
        elif "POS" in name_upper:
            return "Positive"
        elif "TRAI" in name_upper: # Training
            return "Training"
        return "Other" # Or raise an error/warning for unexpected filenames

    def parse_eprime_file(self, filepath):
        """
        Parses the E-Prime .txt log file to extract questionnaire data.
        Returns a dictionary of DataFrames for different questionnaire types.
        """
        self.logger.info(f"QuestionnaireParser: Parsing E-Prime file: {filepath}")
        # Reset lists and context for each new file parse
        self.per_trial_data_list = []
        self.bis_bas_data_list = []
        self.panas_data_list = []
        self.participant_id = None 
        self._current_movie_trial_context = {} 

        try:
            # Attempt common E-Prime encodings
            try:
                with open(filepath, 'r', encoding='utf-16-le', errors='ignore') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e_read:
                self.logger.error(f"QuestionnaireParser: Could not read file {filepath} with utf-16-le or utf-8: {e_read}")
                return {} # Return empty dict of DataFrames

        except FileNotFoundError:
             self.logger.error(f"QuestionnaireParser: File not found: {filepath}")
             return {}
        except Exception as e_open:
            self.logger.error(f"QuestionnaireParser: Error opening or reading file {filepath}: {e_open}")
            return {}

        # Extract Header
        header_match = re.search(r"\*\*\* Header Start \*\*\*(.*?)\*\*\* Header End \*\*\*", content, re.DOTALL)
        if header_match:
            header_block_str = header_match.group(1)
            self._extract_header_info(header_block_str.strip().split('\n'))
        else:
            self.logger.warning("QuestionnaireParser: Could not find E-Prime header block.")
            # Attempt to get participant ID from filename as a fallback
            if self.participant_id is None: # Only try filename if header parsing failed
                basename = os.path.basename(filepath)
                # Regex to find P followed by digits (e.g., EV_P005, P5). More specific: EV_P(\d+)
                match_pid_fname = re.search(r'EV_P(\d+)', basename, re.IGNORECASE) # More specific
                if not match_pid_fname: # Fallback to simpler P(\d+)
                    match_pid_fname = re.search(r'[Pp](\d+)', basename, re.IGNORECASE)
                
                if match_pid_fname:
                    pid_from_fname = match_pid_fname.group(1)
                    self.participant_id = pid_from_fname.lstrip('0') if pid_from_fname.isdigit() else pid_from_fname
                    self.logger.info(f"QuestionnaireParser: Extracted participant ID '{self.participant_id}' from filename '{basename}'.")
                else:
                    self.logger.error("QuestionnaireParser: Participant ID could not be determined from header or filename.")
                    self.participant_id = "Unknown_PID" # Assign a placeholder if truly not found

        # Ensure participant_id is set in the context if it was derived from filename after header check
        if self._current_movie_trial_context and not self._current_movie_trial_context.get('participant_id') and self.participant_id:
            self._current_movie_trial_context['participant_id'] = self.participant_id

        # Extract LogFrames
        log_frames = re.findall(r"\*\*\* LogFrame Start \*\*\*(.*?)\*\*\* LogFrame End \*\*\*", content, re.DOTALL)
        
        for frame_str in log_frames:
            # Ensure participant_id is available to _parse_log_frame if it was just set from filename
            if not self._current_movie_trial_context.get('participant_id') and self.participant_id:
                 # This case is less likely if _parse_log_frame is called after header,
                 # but good for robustness if _parse_log_frame was called before ID is fully set.
                 # However, _parse_log_frame sets participant_id from self.participant_id when creating context.
                 pass
            self._parse_log_frame(frame_str.strip().split('\n'))
        
        # Finalize the very last movie trial data if any context remains
        self._finalize_current_movie_trial() 

        # Convert lists to DataFrames
        df_per_trial = pd.DataFrame(self.per_trial_data_list)
        df_bis_bas = pd.DataFrame(self.bis_bas_data_list)
        df_panas = pd.DataFrame(self.panas_data_list)

        # --- Post-processing / Cleaning ---
        # Ensure participant_id is consistently applied, especially if derived from filename
        # The DataLoader will ultimately set the participant_id on the returned df based on its input,
        # but it's good practice for the parser to try its best.
        if self.participant_id and self.participant_id != "Unknown_PID":
            if not df_per_trial.empty:
                if 'participant_id' not in df_per_trial.columns:
                    df_per_trial.insert(0, 'participant_id', self.participant_id)
                else: # Ensure it's consistent if already present
                    df_per_trial['participant_id'] = self.participant_id
            if not df_bis_bas.empty:
                if 'participant_id' not in df_bis_bas.columns:
                    df_bis_bas.insert(0, 'participant_id', self.participant_id)
                else:
                    df_bis_bas['participant_id'] = self.participant_id
            if not df_panas.empty:
                if 'participant_id' not in df_panas.columns:
                    df_panas.insert(0, 'participant_id', self.participant_id)
                else:
                    df_panas['participant_id'] = self.participant_id

        # Filter out any per_trial rows that might be incomplete (e.g., no movie_filename)
        if not df_per_trial.empty:
            df_per_trial = df_per_trial.dropna(subset=['movie_filename', 'trial_identifier_eprime'])
            # Optional: Filter out training trials if they are not needed for the main analysis output for WP2
            # df_per_trial = df_per_trial[df_per_trial['condition'] != 'Training'].reset_index(drop=True) 
            # Ensure SAM values are numeric
            if 'sam_valence' in df_per_trial.columns:
                df_per_trial['sam_valence'] = pd.to_numeric(df_per_trial['sam_valence'], errors='coerce')
            if 'sam_arousal' in df_per_trial.columns:
                df_per_trial['sam_arousal'] = pd.to_numeric(df_per_trial['sam_arousal'], errors='coerce')


        self.logger.info(f"QuestionnaireParser: Parsing complete for {filepath}. "
                         f"Extracted {len(df_per_trial)} movie trials, "
                         f"{len(df_bis_bas)} BIS/BAS items, {len(df_panas)} PANAS items.")
        
        return {
            'per_trial_ratings': df_per_trial,
            'bis_bas_scores': df_bis_bas,
            'panas_scores': df_panas
        }

# Example usage (for testing the parser directly)
if __name__ == '__main__':
    # Setup a basic logger for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger("TestQuestionnaireParser")

    parser = QuestionnaireParser(test_logger)
    
    # --- IMPORTANT ---
    # Replace this with the actual path to your EV_P005.txt file for testing
    test_file_path = r"D:\pilotRawData\EV_P005\EV_P005.txt" 
    # -----------------

    if os.path.exists(test_file_path):
        parsed_dfs = parser.parse_eprime_file(test_file_path)
        
        print("\n--- Parsing Results ---")
        
        per_trial_df = parsed_dfs.get('per_trial_ratings')
        if per_trial_df is not None and not per_trial_df.empty:
            print("\nPer-Trial Ratings (SAM, Familiarity, EA11, BE7):")
            # Display key columns for verification
            print(per_trial_df[['participant_id', 'movie_filename', 'condition', 'trial_identifier_eprime', 'familiarity_rating', 'sam_valence', 'sam_arousal']].head())
            print(f"\nTotal per-trial entries: {len(per_trial_df)}")
            # You can inspect EA11/BE7 lists for a specific row if needed:
            # print(per_trial_df.iloc[0]['ea11_ratings']) 
            # print(per_trial_df.iloc[0]['be7_ratings'])

        bis_bas_df = parsed_dfs.get('bis_bas_scores')
        if bis_bas_df is not None and not bis_bas_df.empty:
            print("\nBIS/BAS Scores:")
            print(bis_bas_df[['participant_id', 'item_sample_order', 'item_text', 'value']].head())
            print(f"\nTotal BIS/BAS items: {len(bis_bas_df)}")

        panas_df = parsed_dfs.get('panas_scores')
        if panas_df is not None and not panas_df.empty:
            print("\nPANAS Scores:")
            print(panas_df[['participant_id', 'item_sample_order', 'item_text', 'value']].head())
            print(f"\nTotal PANAS items: {len(panas_df)}")

        if not parsed_dfs:
             print("Parsing returned empty results.")

    else:
        test_logger.error(f"Test file not found: {test_file_path}")