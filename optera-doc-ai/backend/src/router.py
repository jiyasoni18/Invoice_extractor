import logging
from src.llm_client import extract_json_with_llm
from src.schemas import DocumentTypeClassification

logger = logging.getLogger(__name__)

def classify_document(ocr_text: str) -> tuple[str, dict]:
    """
    Classifies the document type using a small LLM based on the OCR text.
    Returns (document_type, stats)
    """
    system_prompt = """
    You are a document classification assistant. Your job is to classify the document into EXACTLY ONE of the following types based on its OCR text:
    - invoice (contains vendor, items, total, gst, etc.)
    - mechanic_log (handwritten notes about vehicles, oil changes, mechanic work)
    - meter_reading (dashboard or machine reading showing litres, urea, price)
    - non_document (ONLY use this if the text is completely random garbage, an empty wall, or a picture of a random object with no business text)
    
    CRITICAL: The OCR text is often heavily mangled, garbled, or mashed together. Do NOT classify as 'non_document' just because the text looks messy. If you see ANY numbers, amounts, dates, or business words, classify it as an invoice or mechanic_log.
    
    Respond strictly in JSON format matching this schema:
    {
        "document_type": "string"
    }
    """
    
    user_prompt = f"Here is the OCR text from the document:\n\n{ocr_text}"
    
    try:
        parsed_json, stats = extract_json_with_llm(system_prompt, user_prompt)
        # Validate with pydantic
        classification = DocumentTypeClassification(**parsed_json)
        return classification.document_type, stats
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        # Default to invoice if classification fails (much safer than rejecting valid documents)
        return "invoice", {}

