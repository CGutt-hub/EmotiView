import email
from email.header import decode_header
import dateparser
import datetime
import logging
import re

logger = logging.getLogger(__name__)

class EmailProcessor:
    """
    Handles parsing email content and finding dates within the text.
    """
    def __init__(self):
        # No external dependencies needed for parsing logic itself
        pass

    def clean_text(self, text):
        """Basic text cleaning."""
        if not text:
            return ""
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def decode_subject(self, msg):
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
                logger.warning(f"Could not decode subject: {msg.get('Subject')}, Error: {e}")
                subject = msg.get('Subject') # Use raw subject as fallback
        return self.clean_text(subject)


    def get_email_body(self, msg):
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
                        logger.warning(f"Could not decode part with charset {part.get_content_charset()}: {e}")
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
                    logger.warning(f"Could not decode single part body: {e}")

        return self.clean_text(body)


    def find_dates_in_text(self, text):
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
            logger.error(f"Dateparser failed to search text: {e}")
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
                 logger.warning(f"Skipping naive datetime from dateparser: {dt_obj} (from text: '{date_text}')")
                 continue

            # Compare timezone-aware dates. Only schedule if date is clearly in the future (e.g., > 5 mins from now)
            if dt_obj > now_aware + datetime.timedelta(minutes=5):
                # Check if we already processed this exact date string to avoid near duplicates
                if date_text not in processed_texts:
                    future_dates.append(dt_obj)
                    processed_texts.add(date_text)
                    logger.info(f"Found potential future date: {dt_obj} (from text: '{date_text}')")
            else:
                logger.debug(f"Ignoring past or very near date: {dt_obj} (from text: '{date_text}')")

        # Return unique dates, sorted chronologically
        return sorted(list(set(future_dates)))