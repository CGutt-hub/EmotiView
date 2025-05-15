import imaplib
import logging
import email

logger = logging.getLogger(__name__)

class EmailClient:
    """
    Handles IMAP connection, fetching, and marking emails.
    """
    def __init__(self, config):
        self.config = config
        self.mail = None

    def connect_and_login(self):
        """Connects to the IMAP server and logs in."""
        if self.mail and self.mail.state != 'LOGOUT':
             logger.debug("Already connected or connecting.")
             return True # Assume connection is good if state is not LOGOUT

        logger.info(f"Connecting to IMAP server {self.config.IMAP_SERVER}...")
        try:
            self.mail = imaplib.IMAP4_SSL(self.config.IMAP_SERVER)
            self.mail.login(self.config.IMAP_USERNAME, self.config.IMAP_PASSWORD)
            logger.info(f"Logged in as {self.config.IMAP_USERNAME}.")
            return True
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP Connection or Login Error: {e}", exc_info=True)
            self.mail = None # Ensure mail is None on failure
            return False
        except OSError as e:
            logger.error(f"Network error during IMAP connection: {e}")
            self.mail = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during IMAP connection/login: {e}", exc_info=True)
            self.mail = None
            return False

    def select_mailbox(self):
        """Selects the configured mailbox."""
        if not self.mail or self.mail.state == 'LOGOUT':
            logger.error("Not connected to IMAP server.")
            return False, None

        try:
            status, messages_count = self.mail.select(self.config.MAILBOX_TO_CHECK, readonly=False)
            if status != 'OK':
                logger.error(f"Failed to select mailbox '{self.config.MAILBOX_TO_CHECK}'. Response: {status}")
                return False, None
            logger.info(f"Selected mailbox '{self.config.MAILBOX_TO_CHECK}'.")
            return True, int(messages_count[0])
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP Error selecting mailbox: {e}")
            return False, None
        except Exception as e:
            logger.error(f"An unexpected error occurred selecting mailbox: {e}", exc_info=True)
            return False, None

    def search_emails(self):
        """Searches for emails based on configured criteria."""
        if not self.mail or self.mail.state != 'SELECTED':
            logger.error("Mailbox not selected.")
            return []

        logger.debug(f"Searching using criteria: {self.config.IMAP_SEARCH_CRITERIA}")
        try:
            status, message_ids_bytes = self.mail.search(None, self.config.IMAP_SEARCH_CRITERIA)
            if status != "OK":
                logger.error(f"Failed to search emails. Response: {status}")
                return []
            email_ids = message_ids_bytes[0].split()
            logger.info(f"Found {len(email_ids)} email(s) matching criteria.")
            return email_ids
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP Error searching emails: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred searching emails: {e}", exc_info=True)
            return []

    def fetch_email(self, email_id_bytes):
        """Fetches the full content of a single email."""
        if not self.mail or self.mail.state != 'SELECTED':
            logger.error("Mailbox not selected.")
            return None

        email_id_str = email_id_bytes.decode()
        logger.debug(f"Fetching email ID: {email_id_str}")
        try:
            status, msg_data = self.mail.fetch(email_id_bytes, '(RFC822)')
            if status != "OK":
                logger.error(f"Failed to fetch email ID {email_id_str}. Status: {status}")
                return None

            # msg_data is a list containing tuples, the email content is usually in the second element of the first tuple
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # Parse the raw email bytes into an EmailMessage object
                    return email.message_from_bytes(response_part[1])
            return None # Should not happen if status is OK, but handle defensively

        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP Error fetching email ID {email_id_str}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching email ID {email_id_str}: {e}", exc_info=True)
            return None

    def mark_as_seen(self, email_id_bytes):
        """Marks an email as seen."""
        if not self.mail or self.mail.state != 'SELECTED':
            logger.error("Mailbox not selected.")
            return False

        email_id_str = email_id_bytes.decode()
        try:
             logger.debug(f"Marking email ID {email_id_str} as seen.")
             res, _ = self.mail.store(email_id_bytes, '+FLAGS', '\\Seen')
             if res != 'OK':
                 logger.warning(f"Could not mark email ID {email_id_str} as seen. Response: {res}")
                 return False
             return True
        except imaplib.IMAP4.error as e:
             logger.warning(f"IMAP Error marking email ID {email_id_str} as seen: {e}")
             return False
        except Exception as e:
             logger.warning(f"An unexpected error occurred marking email ID {email_id_str} as seen: {e}", exc_info=True)
             return False

    def close_and_logout(self):
        """Closes the mailbox and logs out."""
        if self.mail:
            try:
                if self.mail.state == 'SELECTED':
                    self.mail.close()
                    logger.debug("Mailbox closed.")
                if self.mail.state != 'LOGOUT':
                    self.mail.logout()
                    logger.info("Logged out from IMAP server.")
            except imaplib.IMAP4.error as e:
                logger.warning(f"Error during IMAP close/logout: {e}")
            except Exception as e:
                logger.warning(f"An unexpected error occurred during IMAP close/logout: {e}", exc_info=True)
            finally:
                self.mail = None

    def is_ready(self):
        """Checks if the client is connected and a mailbox is selected."""
        return self.mail and self.mail.state == 'SELECTED'

    def __enter__(self):
        """Enters the runtime context related to this object."""
        logger.debug("EmailClient entering context...")
        if not self.connect_and_login():
            # connect_and_login logs its own errors
            raise ConnectionError("Failed to connect or login to IMAP for EmailClient context.")

        success, _ = self.select_mailbox()
        if not success:
            # select_mailbox logs its own errors
            self.close_and_logout() # Ensure cleanup if mailbox selection fails after successful login
            raise imaplib.IMAP4.error("Failed to select mailbox for EmailClient context.") # Or a custom exception

        logger.debug("EmailClient context entered successfully.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits the runtime context related to this object."""
        logger.debug(f"EmailClient exiting context. Exception: {exc_type}")
        self.close_and_logout()
        # Return False to propagate exceptions, True to suppress (if handled)
        return False # Default behavior: do not suppress exceptions