import logging
import re
from typing import Dict, Any, Tuple
from src.llm_client import extract_json_with_llm
from src.schemas import InvoiceDocument, HandwrittenLogDocument, MeterReadingDocument

logger = logging.getLogger(__name__)

def parse_invoice(ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system_prompt = """
    You are an expert AI trained to extract structured data from challenging, hybrid (printed + handwritten) invoice OCR text.
    Extract the supplier_name, invoice_no, date, customer_name, vehicle_no, items (name, quantity, unit_price, tax_amount, amount), subtotal, tax_amount, and total.
    
    CRITICAL INSTRUCTIONS for field mapping:
    - SUPPLIER_NAME: This is usually the largest printed text at the very top of the page (e.g., a store or company name).
    - CUSTOMER_NAME: This is usually next to labels like "NAME:", "M/s", or "CUSTOMER:". Do NOT confuse the Customer with the Supplier.
    - INVOICE_NO: This is usually a number near "INVOICE NO:" or printed alone in a top corner. Do NOT confuse the invoice number with the Total Amount!
    - VEHICLE_NO: Look for "Vehicle Reg", "Vehicle No", etc.
    - ITEMS: Look for grid-like data. IMPORTANT: Item descriptions often span MULTIPLE lines. If you see orphan words on the lines immediately below an item, COMBINE them into a single item name! Extract the unit price, item tax/GST, and final item amount.
    - SUBTOTAL: The total before taxes are applied.
    - TAX_AMOUNT: The total GST, CGST, SGST, or tax applied to the whole invoice.
    - TOTAL: Look near the bottom for "Total", "Balance", or the final summed amount (subtotal + tax).
    
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "invoice",
      "supplier_name": "string",
      "invoice_no": "string",
      "date": "string",
      "customer_name": "string",
      "vehicle_no": "string",
      "items": [{"name": "string", "quantity": number, "unit_price": number, "tax_amount": number, "amount": number}],
      "subtotal": number,
      "tax_amount": number,
      "total": number
    }
    If a field is missing, set it to null.
    """
    user_prompt = f"Extract data from this text:\n\n{ocr_text}"
    
    parsed_json, stats = extract_json_with_llm(system_prompt, user_prompt)
    validated = InvoiceDocument(**parsed_json).model_dump()
    return validated, stats

def parse_handwritten_log(ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system_prompt = """
    You are an AI trained to extract structured data from handwritten mechanic logs.
    Extract the date and all work entries (vehicle identifier and work description).
    
    CRITICAL INSTRUCTIONS for field mapping:
    - VEHICLE: This is usually the first alphanumeric code on a row (e.g., TAM23, MAM38, TCM47). If the OCR severely misspelled it (e.g., "Amos", "ripmss"), infer the likely vehicle code based on the context. 
    - WORK: The description of the maintenance. IMPORTANT: The input text you receive is formatted as a JSON array of rows (`[{"row": 1, "text": "..."}, ...]`). Work descriptions often span MULTIPLE rows! If you see a row that doesn't start with a clear vehicle code (e.g., "change", "to stcering pump pipe"), it is a CONTINUATION of the work description from the previous row. COMBINE these orphan rows into the previous vehicle's work description.
    
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "mechanic_log",
      "date": "string",
      "entries": [{"vehicle": "string", "work": "string"}]
    }
    If a field is missing, set it to null.
    """
    user_prompt = f"Extract data from this text:\n\n{ocr_text}"
    
    parsed_json, stats = extract_json_with_llm(system_prompt, user_prompt)
    validated = HandwrittenLogDocument(**parsed_json).model_dump()
    return validated, stats

def parse_handwritten_log_vision(image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.llm_client import call_gemini_vision
    import json
    
    system_prompt = """
    You are an AI trained to extract structured data from highly cursive handwritten mechanic logs.
    Extract the date and all work entries (vehicle identifier and work description).
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "mechanic_log",
      "date": "string",
      "entries": [{"vehicle": "string", "work": "string"}]
    }
    If a field is missing, set it to null.
    """
    
    text_content, stats = call_gemini_vision(image_path, system_prompt)
    parsed_json = json.loads(text_content)
    validated = HandwrittenLogDocument(**parsed_json).model_dump()
    return validated, stats

def parse_meter_reading(ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Attempt simple regex first to save LLM cost
    # This is a naive regex approach. If it fails to find numbers, we use the LLM.
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", ocr_text)
    
    if len(numbers) >= 2:
        # We got some numbers, let's just use the LLM to map them properly since meter layout varies
        pass
        
    system_prompt = """
    You are an AI trained to extract structured data from meter readings (like DEF machines).
    Extract the total amount, volume in litres, price per litre, and urea percentage/amount.
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "meter_reading",
      "amount": number,
      "litres": number,
      "price_per_litre": number,
      "urea": number
    }
    If a field is missing, set it to null.
    """
    user_prompt = f"Extract data from this text:\n\n{ocr_text}"
    
    parsed_json, stats = extract_json_with_llm(system_prompt, user_prompt)
    validated = MeterReadingDocument(**parsed_json).model_dump()
    return validated, stats
