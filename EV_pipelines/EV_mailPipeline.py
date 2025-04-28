# reminder_service.py

import imaplib
import email
from email.header import decode_header
import smtplib
from email.mime.text import MIMEText
import schedule
import time
import dateparser # Finds dates in human-readable text
import datetime
import os
import logging
from threading import Thread
import re # For basic cleanup

# --- Configuration ---
# !! IMPORTANT: Use environment variables or a secure config management system.
# !! DO NOT hardcode credentials directly in the script.
# !! For Gmail/Outlook with 2FA, generate and use an App Password.

# Read from Environment Variables
IMAP_SERVER = os.environ.get('MY_IMAP_SERVER', 'imap.example.com') # e.g., 'imap.gmail.com'
IMAP_USERNAME = os.environ.get('MY_EMAIL_USER', 'your_email@example.com')
IMAP_PASSWORD = os.environ.get('MY_EMAIL_PASS', 'your_password_or_app_password')

SMTP_SERVER = os.environ.get('MY_SMTP_SERVER', 'smtp.example.com') # e.g., 'smtp.gmail.com'
SMTP_PORT = int(os.environ.get('MY_SMTP_PORT', 587)) # Common port for TLS
SMTP_USERNAME = IMAP_USERNAME # Often the same as IMAP username
SMTP_PASSWORD = IMAP_PASSWORD # Often the same as IMAP password

# Who should receive the reminder emails?
REMINDER_RECIPIENT = os.environ.get('MY_REMINDER_RECIPIENT', 'reminder_destination@example.com')
# How often (in minutes) to check the inbox for new emails
CHECK_INTERVAL_MINUTES = 15
# Which mailbox to check
MAILBOX_TO_CHECK = "inbox"
# IMAP search criteria (e.g., '(UNSEEN)', '(SUBJECT "meeting")', '(FROM "boss@example.com")')
# Check only unseen emails to avoid reprocessing
IMAP_SEARCH_CRITERIA = '(UNSEEN)'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global store for scheduled jobs (to avoid duplicates) ---
# In a real app, use a persistent store (DB, file) for resilience
scheduled_reminders = {} # Key: unique job ID, Value: schedule job object

# --- Helper Functions ---

def clean_text(text):
    """Basic text cleaning."""
    if not text:
        return ""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def decode_subject(msg):
    """Decodes email subject potentially encoded."""
    subject = "No Subject"
    if "Subject" in msg:
        try:
            subject_header = decode_header(msg["Subject"])
            decoded_parts = []
            for part, encoding in subject_header:
                if isinstance(part, bytes):
                    # If encoding is None, guess common encodings or use a default
                    charset = encoding if encoding else 'utf-8'
                    try:
                        decoded_parts.append(part.decode(charset, errors='replace'))
                    except LookupError: # Unknown encoding
                        decoded_parts.append(part.decode('iso-8859-1', errors='replace')) # Fallback
                elif isinstance(part, str):
                    decoded_parts.append(part)
            subject = "".join(decoded_parts)
        except Exception as e:
            logging.warning(f"Could not decode subject: {msg.get('Subject')}, Error: {e}")
            subject = msg.get('Subject') # Use raw subject as fallback
    return clean_text(subject)


def get_email_body(msg):
    """Extracts the plain text body from an email message."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Look for plain text parts that are not attachments
            if "attachment" not in content_disposition and content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    # Determine charset, default to utf-8 if not specified
                    charset = part.get_content_charset() or 'utf-8'
                    body = payload.decode(charset, errors='replace')
                    # Take the first plain text part found
                    break
                except Exception as e:
                    logging.warning(f"Could not decode part with charset {part.get_content_charset()}: {e}")
                    continue # Try next part
    else:
        # Not multipart, try to get the payload directly if it's plain text
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            try:
                payload = msg.get_payload(decode=True)
                charset = msg.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='replace')
            except Exception as e:
                logging.warning(f"Could not decode single part body: {e}")

    return clean_text(body)


def find_dates_in_text(text):
    """Uses dateparser to find dates in text. Returns a list of future datetime objects."""
    if not text:
        return []

    # Configure dateparser: prefer future dates, require timezone awareness
    # Adjust 'languages' if you expect dates in specific languages other than English
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RETURN_AS_TIMEZONE_AWARE': True,
        'STRICT_PARSING': False # Be more lenient finding dates
        # 'RELATIVE_BASE': datetime.datetime.now(datetime.timezone.utc) # Set base for relative dates like "tomorrow"
    }
    try:
        # Use search_dates which is good for finding multiple dates in blocks of text
        found_dates_info = dateparser.search.search_dates(text, settings=settings)
    except Exception as e:
        logging.error(f"Dateparser failed to search text: {e}")
        return []


    if not found_dates_info:
        return []

    # Extract datetime objects and filter
    now_aware = datetime.datetime.now(datetime.timezone.utc) # Use timezone-aware comparison
    future_dates = []
    processed_texts = set() # Avoid scheduling based on the exact same text snippet if repeated

    for date_text, dt_obj in found_dates_info:
        if dt_obj.tzinfo is None:
             # This shouldn't happen with RETURN_AS_TIMEZONE_AWARE=True, but handle defensively
             logging.warning(f"Skipping naive datetime from dateparser: {dt_obj} (from text: '{date_text}')")
             continue

        # Compare timezone-aware dates. Only schedule if date is clearly in the future (e.g., > 5 mins from now)
        if dt_obj > now_aware + datetime.timedelta(minutes=5):
            # Check if we already processed this exact date string to avoid near duplicates
            if date_text not in processed_texts:
                future_dates.append(dt_obj)
                processed_texts.add(date_text)
                logging.info(f"Found potential future date: {dt_obj} (from text: '{date_text}')")
        else:
            logging.debug(f"Ignoring past or very near date: {dt_obj} (from text: '{date_text}')")

    # Return unique dates, sorted chronologically
    return sorted(list(set(future_dates)))


def send_reminder_email(recipient, subject, body):
    """Sends an email using configured SMTP settings."""
    message = MIMEText(body, 'plain', 'utf-8') # Ensure UTF-8 encoding
    message['Subject'] = subject
    message['From'] = SMTP_USERNAME
    message['To'] = recipient

    try:
        logging.info(f"Attempting to send reminder to {recipient} with subject: {subject}")
        # Connect using a context manager (ensures connection is closed)
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo() # Identify client to ESMTP server
            server.starttls() # Upgrade connection to secure TLS
            server.ehlo() # Re-identify after TLS
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, recipient, message.as_string())
            logging.info(f"Successfully sent reminder to {recipient}")
            return True # Indicate success
    except smtplib.SMTPAuthenticationError:
        logging.error("SMTP Authentication failed. Check username/password/app password and SMTP settings.")
    except smtplib.SMTPServerDisconnected:
        logging.error("SMTP server disconnected unexpectedly. Check server/port.")
    except smtplib.SMTPConnectError:
        logging.error(f"Could not connect to SMTP server {SMTP_SERVER}:{SMTP_PORT}.")
    except smtplib.SMTPException as e:
        logging.error(f"SMTP Error during sending: {e}")
    except OSError as e:
        logging.error(f"Network error during SMTP connection: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during email sending: {e}", exc_info=True)
    return False # Indicate failure

# --- Scheduling Logic ---

def schedule_reminder(reminder_date, original_subject, original_body_snippet, email_id_for_log):
    """
    Schedules a reminder check using the 'schedule' library.
    Note: This relies on the script running continuously.
    """
    # Ensure reminder_date is timezone-aware (should be from find_dates_in_text)
    if not reminder_date or reminder_date.tzinfo is None:
        logging.error(f"Cannot schedule reminder for invalid/naive datetime: {reminder_date}")
        return None

    # Format reminder content
    reminder_subject = f"Reminder: {original_subject}"
    # Convert date to a readable string (consider local time for readability if needed)
    date_str = reminder_date.strftime('%Y-%m-%d %H:%M %Z') # Include timezone
    reminder_body = (
        f"This is an automated reminder based on an email received.\n\n"
        f"Original Email Subject: '{original_subject}'\n"
        f"Date Mentioned: {date_str}\n\n"
        f"--- Email Snippet ---\n"
        f"{original_body_snippet}\n"
        f"---------------------\n"
        f"(Email ID: {email_id_for_log})" # For tracing back
    )

    # Create a unique ID for the job to prevent duplicates if email is processed again
    # Using email ID and date should be reasonably unique
    job_id = f"reminder_{email_id_for_log}_{reminder_date.isoformat()}"

    if job_id in scheduled_reminders:
        logging.info(f"Reminder for '{original_subject}' on {date_str} (Job ID: {job_id}) already scheduled.")
        return None # Already scheduled

    try:
        # --- The `schedule` library limitation workaround ---
        # `schedule` is best for recurring tasks (e.g., every hour).
        # To run a task *once* at a specific future time, we schedule a frequent *check*.
        # This check sees if the reminder time has passed.
        # A better library like APScheduler handles one-off future dates directly.

        def reminder_job_check():
            """Function executed periodically by schedule to check if it's time."""
            now_aware = datetime.datetime.now(datetime.timezone.utc)
            logging.debug(f"Checking job {job_id}. Now: {now_aware}, Target: {reminder_date}")

            if now_aware >= reminder_date:
                logging.info(f"Time reached for reminder (Job ID: {job_id}). Sending email...")
                success = send_reminder_email(REMINDER_RECIPIENT, reminder_subject, reminder_body)

                if success:
                    logging.info(f"Reminder sent successfully for Job ID: {job_id}. Unscheduling check.")
                    # Remove from our tracking dict
                    if job_id in scheduled_reminders:
                        del scheduled_reminders[job_id]
                    # Tell `schedule` to stop running this specific check job
                    return schedule.CancelJob
                else:
                    # Optional: Implement retry logic here? Or just log and keep checking?
                    logging.error(f"Failed to send reminder for Job ID: {job_id}. Will retry on next check.")
                    # Keep the job scheduled for the next check interval
                    return None # Keep job running
            else:
                # Not time yet, do nothing this cycle
                return None # Keep job running

        # Schedule the check function to run frequently (e.g., every minute)
        # Adjust frequency based on desired precision vs system load
        scheduled_job = schedule.every(1).minute.do(reminder_job_check)
        scheduled_reminders[job_id] = scheduled_job # Store the job reference
        logging.info(f"Scheduled check for reminder: {date_str} (Job ID: {job_id})")
        return scheduled_job

    except Exception as e:
        logging.error(f"Error scheduling reminder check for {date_str} (Job ID: {job_id}): {e}", exc_info=True)
        # Clean up if job ID was added prematurely
        if job_id in scheduled_reminders:
            del scheduled_reminders[job_id]
        return None


# --- Main Email Processing Logic ---

def check_emails_and_schedule():
    """Connects to IMAP, fetches emails based on criteria, finds dates, and schedules reminders."""
    logging.info(f"Connecting to IMAP server {IMAP_SERVER}...")
    mail = None # Initialize to ensure logout happens in finally block
    try:
        # Connect to the server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        # Login
        mail.login(IMAP_USERNAME, IMAP_PASSWORD)
        logging.info(f"Logged in as {IMAP_USERNAME}.")
        # Select mailbox
        status, messages_count = mail.select(MAILBOX_TO_CHECK, readonly=False) # Readonly=False to allow marking as seen
        if status != 'OK':
            logging.error(f"Failed to select mailbox '{MAILBOX_TO_CHECK}'. Response: {status}")
            return

        logging.info(f"Selected mailbox '{MAILBOX_TO_CHECK}'. Searching using criteria: {IMAP_SEARCH_CRITERIA}")

        # Search for emails matching the criteria
        status, message_ids_bytes = mail.search(None, IMAP_SEARCH_CRITERIA)
        if status != "OK":
            logging.error(f"Failed to search emails with criteria '{IMAP_SEARCH_CRITERIA}'. Response: {status}")
            return

        email_ids = message_ids_bytes[0].split()
        if not email_ids:
            logging.info("No emails found matching the criteria.")
            return

        logging.info(f"Found {len(email_ids)} email(s) matching criteria. Processing...")

        for email_id_bytes in email_ids:
            email_id_str = email_id_bytes.decode() # For logging and job IDs
            logging.debug(f"Fetching email ID: {email_id_str}")
            # Fetch the email data (RFC822 means full email content)
            status, msg_data = mail.fetch(email_id_bytes, '(RFC822)')

            if status != "OK":
                logging.error(f"Failed to fetch email ID {email_id_str}. Status: {status}")
                continue

            # msg_data is a list containing tuples, the email content is usually in the second element of the first tuple
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    try:
                        # Parse the raw email bytes into an EmailMessage object
                        msg = email.message_from_bytes(response_part[1])

                        subject = decode_subject(msg)
                        body = get_email_body(msg)
                        sender = msg.get("From", "Unknown Sender")

                        logging.info(f"Processing Email ID {email_id_str}: From='{sender}', Subject='{subject}'")

                        # Combine subject and body for date searching
                        text_to_search = f"{subject}\n{body}"
                        found_dates = find_dates_in_text(text_to_search)

                        if not found_dates:
                            logging.info(f"No future dates found in email ID {email_id_str}.")
                        else:
                            logging.info(f"Found {len(found_dates)} potential date(s) in email ID {email_id_str}: "
                                         f"{[d.strftime('%Y-%m-%d %H:%M %Z') for d in found_dates]}")

                            # Schedule a reminder for each unique future date found
                            body_snippet = (body[:250] + '...') if len(body) > 250 else body # Snippet for context
                            for reminder_date in found_dates:
                                schedule_reminder(reminder_date, subject, body_snippet, email_id_str)

                        # Mark the email as seen after processing (optional)
                        # Be careful with this if multiple processes might access the same mailbox
                        # Or if you want to re-run on already seen emails for testing.
                        try:
                             logging.debug(f"Marking email ID {email_id_str} as seen.")
                             res, _ = mail.store(email_id_bytes, '+FLAGS', '\\Seen')
                             if res != 'OK':
                                 logging.warning(f"Could not mark email ID {email_id_str} as seen.")
                        except Exception as e:
                             logging.warning(f"Error marking email ID {email_id_str} as seen: {e}")

                    except Exception as e:
                        logging.error(f"Error processing content of email ID {email_id_str}: {e}", exc_info=True)
                    # Break after processing the first valid response part for this email ID
                    break

        logging.info("Finished processing batch of emails.")

    except imaplib.IMAP4.error as e:
        logging.error(f"IMAP Error: {e}", exc_info=True)
    except smtplib.SMTPAuthenticationError:
        # Specific handling if login fails during the check (less likely here, more for sending)
        logging.error("IMAP Authentication failed. Check credentials.")
    except OSError as e:
        logging.error(f"Network error during IMAP connection: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during email checking: {e}", exc_info=True)
    finally:
        # Ensure logout happens even if errors occur
        if mail and mail.state == 'SELECTED':
            try:
                mail.close() # Close the mailbox
            except imaplib.IMAP4.error as e:
                logging.warning(f"Error closing mailbox: {e}")
        if mail and mail.state != 'LOGOUT':
             try:
                 mail.logout()
                 logging.info("Logged out from IMAP server.")
             except imaplib.IMAP4.error as e:
                 logging.warning(f"Error during IMAP logout: {e}")


# --- Scheduler Runner ---
def run_scheduler():
    """Runs the schedule loop indefinitely."""
    logging.info("Starting scheduler loop...")
    while True:
        try:
            schedule.run_pending()
            # Sleep for a short duration before checking the schedule again
            # This prevents high CPU usage in an empty schedule
            time.sleep(30) # Check schedule every 30 seconds
        except Exception as e:
            logging.error(f"Error in scheduler loop: {e}", exc_info=True)
            # Avoid crashing the loop, wait a bit before retrying
            time.sleep(60)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Email Reminder Service...")

    # --- Validate Configuration ---
    if not all([IMAP_SERVER, IMAP_USERNAME, IMAP_PASSWORD, SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, REMINDER_RECIPIENT]):
        logging.error("CRITICAL: Missing one or more required environment variables (MY_IMAP_*, MY_SMTP_*, MY_EMAIL_*, MY_REMINDER_RECIPIENT). Exiting.")
        exit(1)
    if 'example.com' in IMAP_SERVER or 'your_email' in IMAP_USERNAME:
         logging.warning("Using default example configuration values. Please set environment variables.")
         # Consider exiting here in a production scenario

    # Perform an initial check immediately upon starting
    logging.info("Performing initial email check...")
    check_emails_and_schedule()

    # Schedule the email check function to run periodically
    logging.info(f"Scheduling periodic email check every {CHECK_INTERVAL_MINUTES} minutes.")
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(check_emails_and_schedule)

    # Start the scheduler loop in a separate thread
    # This allows the main thread to potentially handle other tasks or signals if needed
    # Using daemon=True means the thread won't prevent the program from exiting
    scheduler_thread = Thread(target=run_scheduler, daemon=True, name="SchedulerThread")
    scheduler_thread.start()

    # Keep the main thread alive to allow the daemon scheduler thread to run
    # Handle Ctrl+C (KeyboardInterrupt) for graceful shutdown
    try:
        while scheduler_thread.is_alive():
            # Keep the main thread running, checking the scheduler thread's status
            scheduler_thread.join(timeout=1.0) # Wait for thread with timeout
    except KeyboardInterrupt:
        logging.info("Ctrl+C received. Shutting down scheduler...")
        # Clear scheduled jobs (optional, depends if you want them to persist)
        schedule.clear()
        logging.info("Scheduled jobs cleared. Exiting.")
    except Exception as e:
        logging.error(f"Main loop encountered an error: {e}", exc_info=True)
    finally:
        logging.info("Email Reminder Service stopped.")
