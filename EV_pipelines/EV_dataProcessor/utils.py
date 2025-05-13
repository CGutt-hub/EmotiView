import os
import logging
import datetime
# Assuming config.py is in the same directory or EV_pipelines is in PYTHONPATH
from .. import config # Use relative import if utils.py is part of a package

# --- Main Pipeline Logger Setup ---
main_log_file_path = os.path.join(config.LOG_DIR, f"pilot_orchestrator_main_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

def setup_main_logger():
    """Sets up and returns the main pipeline logger."""
    logger = logging.getLogger("PilotOrchestratorMain")
    logger.setLevel(getattr(logging, config.MAIN_LOG_LEVEL.upper(), logging.INFO))
    
    # Prevent adding handlers multiple times if this function is called again
    if not logger.handlers:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        
        fh = logging.FileHandler(main_log_file_path)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
        sh = logging.StreamHandler()
        sh.setFormatter(file_formatter)
        logger.addHandler(sh)
    return logger

main_logger = setup_main_logger() # Initialize main logger once

# --- Participant-specific Logger ---
participant_loggers = {}

def get_participant_logger(participant_id):
    """Gets or creates a logger for a specific participant."""
    if participant_id in participant_loggers:
        return participant_loggers[participant_id]

    logger = logging.getLogger(f"PilotParticipant_{participant_id}")
    logger.setLevel(getattr(logging, config.PARTICIPANT_LOG_LEVEL.upper(), logging.DEBUG))
    logger.propagate = False 

    # Log to participant's raw data subfolder (as in original EV_pilotAnalysisPipeline)
    participant_log_dir = os.path.join(config.PARTICIPANT_DATA_BASE_DIR, participant_id)
    os.makedirs(participant_log_dir, exist_ok=True)
    log_file = os.path.join(participant_log_dir, f"EV_{participant_id}_pilot_processingLog.txt")

    fh = logging.FileHandler(log_file, mode='a') # Append mode
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    participant_loggers[participant_id] = logger
    main_logger.info(f"Participant log for {participant_id} set up at: {log_file}")
    return logger

def close_participant_logger(participant_id):
    """Closes handlers for a specific participant logger."""
    if participant_id in participant_loggers:
        logger = participant_loggers[participant_id]
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        del participant_loggers[participant_id]
        main_logger.info(f"Participant logger for {participant_id} closed.")