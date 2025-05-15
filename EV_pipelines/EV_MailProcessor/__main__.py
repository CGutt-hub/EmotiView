import time
import logging
from threading import Thread, Event
import sys
import imaplib # For specific exception handling if EmailClient raises it

# Import the classes from our package
from .config import Config
from .email_client import EmailClient
from .smtp_sender import SMTPSender
from .email_processor import EmailProcessor
from .reminder_scheduler import ReminderScheduler

logger = logging.getLogger(__name__)

class MailReminderService:
    """
    Orchestrates the email reminder service.
    Connects components, schedules the email check, and runs the main loop.
    """
    def __init__(self, config, email_client, email_processor, reminder_scheduler):
        self.config = config
        self.email_client = email_client
        self.email_processor = email_processor
        self.reminder_scheduler = reminder_scheduler
        self._stop_event = Event() # Use threading.Event for graceful shutdown

    def check_emails_and_schedule(self):
        """
        Connects to IMAP, fetches emails based on criteria, finds dates,
        and schedules reminders via the ReminderScheduler.
        """
        logger.info("Starting periodic email check...")
        try:
            with self.email_client as client: # Use EmailClient as a context manager
                # The __enter__ method of EmailClient handles connect, login, and select_mailbox.
                # If any of those fail, an exception is raised and caught below.

                email_ids = client.search_emails()

                if not email_ids:
                    logger.info("No new emails found matching the criteria.")
                else:
                    logger.info(f"Found {len(email_ids)} email(s) matching criteria. Processing...")

                    for email_id_bytes in email_ids:
                        email_id_str = email_id_bytes.decode()
                        msg = client.fetch_email(email_id_bytes)

                        if msg:
                            try:
                                subject = self.email_processor.decode_subject(msg)
                                body = self.email_processor.get_email_body(msg)
                                sender = msg.get("From", "Unknown Sender")

                                logger.info(f"Processing Email ID {email_id_str}: From='{sender}', Subject='{subject}'")

                                # Combine subject and body for date searching
                                text_to_search = f"{subject}\n{body}"
                                found_dates = self.email_processor.find_dates_in_text(text_to_search)

                                if not found_dates:
                                    logger.info(f"No future dates found in email ID {email_id_str}.")
                                else:
                                    logger.info(f"Found {len(found_dates)} potential date(s) in email ID {email_id_str}: "
                                                 f"{[d.strftime('%Y-%m-%d %H:%M %Z') for d in found_dates]}")

                                    # Schedule a reminder for each unique future date found
                                    body_snippet = (body[:250] + '...') if len(body) > 250 else body # Snippet for context
                                    for reminder_date in found_dates:
                                        self.reminder_scheduler.schedule_reminder(
                                            reminder_date, subject, body_snippet, email_id_str
                                        )

                                # Mark the email as seen after processing (optional)
                                client.mark_as_seen(email_id_bytes)

                            except Exception as e: # Catch errors during processing of a single email
                                logger.error(f"Error processing content of email ID {email_id_str}: {e}", exc_info=True)
                        else:
                            logger.error(f"Could not fetch email ID {email_id_str}. Skipping processing.")

                    logger.info("Finished processing batch of emails.")
        except (ConnectionError, imaplib.IMAP4.error) as e: # Catch errors from EmailClient.__enter__
            logger.error(f"IMAP client setup failed for email check: {e}")
        except Exception as e: # Catch any other unexpected errors during the check
            logger.error(f"Unexpected error during email check: {e}", exc_info=True)
        finally:
            # EmailClient.__exit__ ensures close_and_logout is called automatically.
            logger.info("Periodic email check cycle finished.")

    def run_scheduler_loop(self):
        """Runs the schedule loop indefinitely in a thread."""
        logger.info("Starting scheduler loop thread...")
        while not self._stop_event.is_set():
            try:
                self.reminder_scheduler.run_pending()
                # Sleep for a short duration before checking the schedule again
                # This prevents high CPU usage in an empty schedule
                time.sleep(1) # Check schedule more frequently (e.g., every second)
                               # The actual job check inside runs only when due.
                               # This makes the one-minute check more responsive.
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                # Avoid crashing the loop, wait a bit before retrying
                time.sleep(5) # Wait less on error to potentially recover faster
        logger.info("Scheduler loop thread stopped.")

    def run(self):
        """Starts the email reminder service."""
        logger.info("Starting Email Reminder Service...")

        # Perform an initial check immediately upon starting
        logger.info("Performing initial email check...")
        self.check_emails_and_schedule()

        # Schedule the email check function to run periodically
        logger.info(f"Scheduling periodic email check every {self.config.CHECK_INTERVAL_MINUTES} minutes.")
        schedule.every(self.config.CHECK_INTERVAL_MINUTES).minutes.do(self.check_emails_and_schedule)

        # Start the scheduler loop in a separate thread
        # Using daemon=True means the thread won't prevent the program from exiting
        scheduler_thread = Thread(target=self.run_scheduler_loop, daemon=True, name="SchedulerThread")
        scheduler_thread.start()

        # Keep the main thread alive to allow the daemon scheduler thread to run
        # Handle Ctrl+C (KeyboardInterrupt) for graceful shutdown
        try:
            while scheduler_thread.is_alive():
                # Keep the main thread alive by periodically checking the scheduler thread.
                # This allows KeyboardInterrupt to be caught by the main thread.
                # A short timeout ensures the join doesn't block indefinitely if the
                # thread were to exit unexpectedly for other reasons.
                scheduler_thread.join(timeout=1.0) # Check every second
        except KeyboardInterrupt:
            logger.info("Ctrl+C received. Initiating shutdown...")
        except Exception as e:
            logger.error(f"Main loop encountered an error: {e}", exc_info=True)
        finally:
            logger.info("Shutdown sequence initiated...")
            self._stop_event.set() # Signal the scheduler thread to stop
            scheduler_thread.join(timeout=10) # Wait for the thread to finish (with a generous timeout)
            if scheduler_thread.is_alive():
                 logger.warning("Scheduler thread did not stop gracefully within the timeout.")
            self.reminder_scheduler.clear_jobs() # Clear scheduled jobs on shutdown
            logger.info("Scheduled jobs cleared.")
            logger.info("Email Reminder Service stopped.")


if __name__ == "__main__":
    # Setup logging early
    # Basic config here, Config class will refine level later
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load and validate configuration
    config = Config()
    config.setup_logging() # Apply configured log level

    if not config.validate():
        sys.exit(1) # Exit if configuration is invalid

    # Instantiate components
    smtp_sender = SMTPSender(config)
    email_client = EmailClient(config)
    email_processor = EmailProcessor()
    reminder_scheduler = ReminderScheduler(smtp_sender, config)

    # Instantiate and run the service
    service = MailReminderService(config, email_client, email_processor, reminder_scheduler)
    service.run()