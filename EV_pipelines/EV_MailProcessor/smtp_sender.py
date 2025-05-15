import smtplib
import logging
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

class SMTPSender:
    """
    Handles sending emails via SMTP.
    """
    def __init__(self, config):
        self.config = config

    def send_email(self, recipient, subject, body):
        """Sends an email using configured SMTP settings."""
        message = MIMEText(body, 'plain', 'utf-8') # Ensure UTF-8 encoding
        message['Subject'] = subject
        message['From'] = self.config.SMTP_USERNAME
        message['To'] = recipient

        try:
            logger.info(f"Attempting to send email to {recipient} with subject: {subject}")
            # Connect using a context manager (ensures connection is closed)
            with smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT, timeout=30) as server:
                server.ehlo() # Identify client to ESMTP server
                server.starttls() # Upgrade connection to secure TLS
                server.ehlo() # Re-identify after TLS
                server.login(self.config.SMTP_USERNAME, self.config.SMTP_PASSWORD)
                server.sendmail(self.config.SMTP_USERNAME, recipient, message.as_string())
                logger.info(f"Successfully sent email to {recipient}")
                return True # Indicate success
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP Authentication failed. Check username/password/app password and SMTP settings.")
        except smtplib.SMTPServerDisconnected:
            logger.error("SMTP server disconnected unexpectedly. Check server/port.")
        except smtplib.SMTPConnectError:
            logger.error(f"Could not connect to SMTP server {self.config.SMTP_SERVER}:{self.config.SMTP_PORT}.")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP Error during sending: {e}")
        except OSError as e:
            logger.error(f"Network error during SMTP connection: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during email sending: {e}", exc_info=True)
        return False # Indicate failure