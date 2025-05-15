import schedule
import time
import datetime
import logging

logger = logging.getLogger(__name__)

class ReminderScheduler:
    """
    Manages scheduling and executing reminder jobs using the 'schedule' library.
    Relies on an SMTPSender instance to send the actual emails.
    """
    def __init__(self, smtp_sender, config):
        self.smtp_sender = smtp_sender
        self.config = config
        # Global store for scheduled jobs (to avoid duplicates)
        # In a real app, use a persistent store (DB, file) for resilience
        self.scheduled_reminders = {} # Key: unique job ID, Value: schedule job object
        self._schedule = schedule # Use the schedule library instance

    def schedule_reminder(self, reminder_date, original_subject, original_body_snippet, email_id_for_log):
        """
        Schedules a reminder check using the 'schedule' library.
        Note: This relies on the script running continuously.
        """
        # Ensure reminder_date is timezone-aware (should be from find_dates_in_text)
        if not reminder_date or reminder_date.tzinfo is None:
            logger.error(f"Cannot schedule reminder for invalid/naive datetime: {reminder_date}")
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

        if job_id in self.scheduled_reminders:
            logger.info(f"Reminder for '{original_subject}' on {date_str} (Job ID: {job_id}) already scheduled.")
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
                logger.debug(f"Checking job {job_id}. Now: {now_aware}, Target: {reminder_date}")

                if now_aware >= reminder_date:
                    logger.info(f"Time reached for reminder (Job ID: {job_id}). Sending email...")
                    success = self.smtp_sender.send_email(self.config.REMINDER_RECIPIENT, reminder_subject, reminder_body)

                    if success:
                        logger.info(f"Reminder sent successfully for Job ID: {job_id}. Unscheduling check.")
                        # Remove from our tracking dict
                        if job_id in self.scheduled_reminders:
                            del self.scheduled_reminders[job_id]
                        # Tell `schedule` to stop running this specific check job
                        return self._schedule.CancelJob
                    else:
                        # Optional: Implement retry logic here? Or just log and keep checking?
                        logger.error(f"Failed to send reminder for Job ID: {job_id}. Will retry on next check.")
                        # Keep the job scheduled for the next check interval
                        return None # Keep job running
                else:
                    # Not time yet, do nothing this cycle
                    return None # Keep job running

            # Schedule the check function to run frequently (e.g., every minute)
            # Adjust frequency based on desired precision vs system load
            scheduled_job = self._schedule.every(1).minute.do(reminder_job_check)
            self.scheduled_reminders[job_id] = scheduled_job # Store the job reference
            logger.info(f"Scheduled check for reminder: {date_str} (Job ID: {job_id})")
            return scheduled_job

        except Exception as e:
            logger.error(f"Error scheduling reminder check for {date_str} (Job ID: {job_id}): {e}", exc_info=True)
            # Clean up if job ID was added prematurely
            if job_id in self.scheduled_reminders:
                del self.scheduled_reminders[job_id]
            return None

    def run_pending(self):
        """Runs all jobs that are scheduled to run."""
        self._schedule.run_pending()

    def clear_jobs(self):
        """Clears all scheduled jobs."""
        self._schedule.clear()
        self.scheduled_reminders.clear()
        logger.info("All scheduled jobs cleared.")