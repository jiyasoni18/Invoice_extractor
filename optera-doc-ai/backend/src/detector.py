import logging

logger = logging.getLogger(__name__)

def is_document(extracted_text: str) -> tuple[bool, str]:
    """
    Determines if the image is a valid document based on the OCR text.
    Returns (is_valid, reason)
    """
    text = extracted_text.strip()
    
    if not text:
        return False, "No text found in image"
        
    # Count alphanumeric characters
    alnum_count = sum(c.isalnum() for c in text)
    
    # If the text has very few characters, it's likely a non-document (e.g. battery, tyre)
    if alnum_count < 20:
        return False, f"Insufficient text (only {alnum_count} chars). Likely non-document."
        
    return True, "Valid document"
