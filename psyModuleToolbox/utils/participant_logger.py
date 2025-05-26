import logging
import os

class ParticipantLogger:
    """
    Manages a dedicated logger for a single participant, writing to a specific file.
    """
    def __init__(self, participant_base_output_dir, participant_id, log_level_str="INFO"):
        """
        Initializes the participant-specific logger.

        Args:
            participant_base_output_dir (str): The base output directory for this participant.
            participant_id (str): The ID of the participant.
            log_level_str (str): The desired logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        self.participant_id = participant_id
        self.log_file_path = os.path.join(participant_base_output_dir, f"{participant_id}_processing.log")
        self._logger = logging.getLogger(f"ParticipantLogger_{participant_id}")
        self._handler = None # To store the file handler

        # Prevent adding multiple handlers if the logger is reused (e.g., in interactive sessions)
        if not self._logger.handlers:
            # Set logger level (should be lowest level you want to process)
            self._logger.setLevel(logging.DEBUG) # Set logger to DEBUG to capture all messages

            # Create file handler
            try:
                # Ensure the directory exists before creating the file
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
                self._handler = logging.FileHandler(self.log_file_path, mode='w')
                # Set handler level based on config
                log_level = getattr(logging, log_level_str.upper(), logging.INFO)
                self._handler.setLevel(log_level)

                # Create formatter
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
                self._handler.setFormatter(formatter)

                # Add handler to logger
                self._logger.addHandler(self._handler)
                self._logger.info(f"ParticipantLogger initialized for {participant_id}. Logging to {self.log_file_path} at level {log_level_str}.")
            except Exception as e:
                # Fallback to console logging if file handler fails
                # Note: This error will likely only appear in the main orchestrator log
                # because the participant logger's file handler failed to initialize.
                print(f"ERROR: Failed to create file handler for participant {participant_id}: {e}. Logging will only go to main orchestrator log.")
                self._handler = None # Ensure handler is None if creation failed


    def get_logger(self):
        """
        Returns the configured logger instance for the participant.
        """
        return self._logger

    def close_handlers(self):
        """
        Closes the file handler associated with this logger.
        Should be called when processing for the participant is complete.
        """
        if self._handler:
            try:
                # Check if the handler is still attached before removing
                if self._handler in self._logger.handlers:
                     self._logger.removeHandler(self._handler)
                self._handler.close()
                # Note: We don't log here because the handler is being closed.
                # The main orchestrator log should indicate participant processing is finished.
            except Exception as e:
                # Log to the main logger if closing fails, as participant logger might be unusable
                # We can't rely on the participant logger here, so print or use a known main logger
                main_logger = logging.getLogger("MainOrchestrator") # Assuming this logger exists
                if main_logger:
                    main_logger.error(f"Error closing log handler for participant {self.participant_id}: {e}", exc_info=True)
                else:
                    print(f"ERROR: Error closing log handler for participant {self.participant_id}: {e}")