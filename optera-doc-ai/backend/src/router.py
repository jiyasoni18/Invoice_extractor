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
    - non_document (if the text is random or doesn't match the above)
    
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
        # Default to unknown/non_document if classification fails
        return "non_document", {}

