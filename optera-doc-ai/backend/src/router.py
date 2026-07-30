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
    You are a document classification assistant. Your job is to classify the document into EXACTLY ONE of the following types based on its OCR text.

    CRITICAL RULES FOR CLASSIFICATION:
    1. If the OCR text is extremely clean and contains standard printed billing words (GST, Tax, Total, Invoice, Bill To), it is an 'invoice'.
    2. If the OCR text is highly garbled, scrambled, or messy AND contains ANY automotive terms (e.g., oil, clutch, brake, coolant, engine, tyre) even if misspelled (like "eluhcy oil", "Enojnacoil"), you MUST classify it as 'mechanic_log'.
    3. If the text is messy/garbled but looks like a bill of sale (amounts, totals, shop names) without automotive work, classify as 'handwritten_invoice'.
    4. If it just shows numbers like litres or price (dashboard/machine), classify as 'meter_reading'.
    5. ONLY use 'non_document' if the text is completely random garbage with absolutely zero business or automotive meaning.

    Types:
    - invoice
    - handwritten_invoice
    - mechanic_log
    - meter_reading
    - non_document
    
    Respond strictly in JSON format matching this schema:
    {
        "document_type": "string"
    }
    """
    
    # Truncate the text to a maximum of 1000 characters. 
    # This drastically caps the LLM token cost for the router if a user uploads a massive text-heavy non-document.
    truncated_text = ocr_text[:1000]
    
    user_prompt = f"Here is the OCR text from the document:\n\n{truncated_text}"
    
    try:
        parsed_json, stats = extract_json_with_llm(system_prompt, user_prompt)
        # Validate with pydantic
        classification = DocumentTypeClassification(**parsed_json)
        return classification.document_type, stats
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        # Default to invoice if classification fails (much safer than rejecting valid documents)
        return "invoice", {}

