import os
import numpy as np
from .. import config # Relative import

class QuestionnaireParser:
    def __init__(self, logger):
        self.logger = logger
        self.logger.info("QuestionnaireParser initialized.")

    def parse(self, participant_id, participant_raw_data_path):
        """
        Parses questionnaire data for a participant.
        Returns a dictionary with parsed data, including participant_id.
        Numeric fields are converted to float, or np.nan on error.
        """
        file_path = os.path.join(participant_raw_data_path, config.QUESTIONNAIRE_TXT_FILENAME)
        parsed_data = {'participant_id': participant_id} # Ensure participant_id is always present

        if not os.path.exists(file_path):
            self.logger.info(f"QuestionnaireParser - File not found: {file_path}")
            return parsed_data

        try:
            self.logger.info(f"QuestionnaireParser - Parsing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or ':' not in line:
                        if line: self.logger.warning(f"QuestionnaireParser - Malformed line {line_num}: '{line}'")
                        continue
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    # Attempt to convert known numeric fields
                    numeric_q_keys = ['SAM_arousal', 'SAM_valence', 'BIS_score', 'BAS_drive', 'PANAS_Positive', 'PANAS_Negative']
                    if key in numeric_q_keys:
                        try: parsed_data[key] = float(value)
                        except ValueError: parsed_data[key] = np.nan; self.logger.warning(f"QuestionnaireParser - Could not convert '{value}' to float for key '{key}'.")
                    else: parsed_data[key] = value
            self.logger.info(f"QuestionnaireParser - Successfully parsed: {file_path}")
        except Exception as e:
            self.logger.error(f"QuestionnaireParser - Error parsing {file_path}: {e}", exc_info=True)
        return parsed_data