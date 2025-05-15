import os
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Handles loading and validating configuration from environment variables.
    """
    def __init__(self):
        self.IMAP_SERVER = os.environ.get('MY_IMAP_SERVER')
        self.IMAP_USERNAME = os.environ.get('MY_EMAIL_USER')
        self.IMAP_PASSWORD = os.environ.get('MY_EMAIL_PASS')

        self.SMTP_SERVER = os.environ.get('MY_SMTP_SERVER')
        self.SMTP_PORT = int(os.environ.get('MY_SMTP_PORT', 587))
        # Often the same, but allow override if needed
        self.SMTP_USERNAME = os.environ.get('MY_SMTP_USER', self.IMAP_USERNAME)
        self.SMTP_PASSWORD = os.environ.get('MY_SMTP_PASS', self.IMAP_PASSWORD)

        self.REMINDER_RECIPIENT = os.environ.get('MY_REMINDER_RECIPIENT')

        self.CHECK_INTERVAL_MINUTES = int(os.environ.get('MY_CHECK_INTERVAL_MINUTES', 15))
        self.MAILBOX_TO_CHECK = os.environ.get('MY_MAILBOX_TO_CHECK', "inbox")
        self.IMAP_SEARCH_CRITERIA = os.environ.get('MY_IMAP_SEARCH_CRITERIA', '(UNSEEN)')

        # Logging level can also be configured
        log_level_str = os.environ.get('MY_LOG_LEVEL', 'INFO').upper()
        self.LOG_LEVEL = getattr(logging, log_level_str, logging.INFO)

    def validate(self):
        """Validates that essential configuration is present."""
        required = {
            'IMAP_SERVER': self.IMAP_SERVER,
            'IMAP_USERNAME': self.IMAP_USERNAME,
            'IMAP_PASSWORD': self.IMAP_PASSWORD,
            'SMTP_SERVER': self.SMTP_SERVER,
            'SMTP_USERNAME': self.SMTP_USERNAME,
            'SMTP_PASSWORD': self.SMTP_PASSWORD,
            'REMINDER_RECIPIENT': self.REMINDER_RECIPIENT,
        }

        missing = [key for key, value in required.items() if not value]

        if missing:
            logger.critical(f"Missing required environment variables: {', '.join(missing)}. Exiting.")
            return False

        if 'example.com' in self.IMAP_SERVER or 'your_email' in self.IMAP_USERNAME:
             logger.warning("Using default example configuration values. Please set environment variables.")
             # In a production scenario, you might want to return False here

        return True

    def setup_logging(self):
        """Configures basic logging."""
        logging.basicConfig(level=self.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.info(f"Logging level set to {logging.getLevelName(self.LOG_LEVEL)}")