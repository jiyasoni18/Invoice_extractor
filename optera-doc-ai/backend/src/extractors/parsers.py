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
    - GENERAL: The OCR text is severely corrupted and may contain random symbols or mashed characters. You MUST aggressively infer the real words. For example, "SHIIV SHAKI! Tजायगन MगoRS" should be corrected to "SHIV SHAKTI MOTORS". Never return null if there is a highly probable match.
    - SUPPLIER_NAME: This is usually the largest printed text at the very top of the page (e.g., a store or company name). Aggressively fix typos.
    - CUSTOMER_NAME: This is usually next to labels like "NAME:", "M/s", or "CUSTOMER:". Do NOT confuse the Customer with the Supplier.
    - INVOICE_NO: This is usually a number near "INVOICE NO:" or printed alone in a top corner. Do NOT confuse the invoice number with the Total Amount!
    - VEHICLE_NO: Look for "Vehicle Reg", "Vehicle No", etc.
    - ITEMS: Look for grid-like data. The OCR might have mashed an entire table into a single long string of text! You MUST aggressively reconstruct the individual items from the numbers (e.g. quantity, rate, amount) hidden in the text. Split the items logically.
    - TAXABLE_VALUE: The base amount before taxes are applied (often called "Taxable Value" or "Subtotal").
    - TAX BREAKDOWN: Extract the exact CGST, SGST, and IGST amounts from the bottom of the invoice or the tax summary table.
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
      "taxable_value": number,
      "cgst": number,
      "sgst": number,
      "igst": number,
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
    - VEHICLE: The OCR for cursive text is often severely corrupted. The valid vehicle code format in this fleet is ALWAYS three letters followed by two digits (e.g., TAM23, MAM38, TCM47). 
      IMPORTANT: The vehicle code is SOMETIMES preceded by a serial number or bullet point like "(03)", "1)", "(7)", "44)", "09)", etc. 
      If there is a serial number, extract the alphanumeric code immediately following it. If there is NO serial number (e.g. the very first row), extract the alphanumeric code itself as the VEHICLE.
      If the OCR severely misspelled the vehicle code, you MUST aggressively correct it to the nearest valid fleet format (3 letters + 2 digits) based on phonetic or visual similarities.
      Do NOT skip the first vehicle just because it lacks a serial number!
      
    - WORK: Reconstruct the work description logically using automotive terms (e.g. "wheel brake set", "coolant hose", "steering pipe"). If you see a row that doesn't start with a clear vehicle code (or a serial number + vehicle code), it is a CONTINUATION of the work description from the previous row. COMBINE these orphan rows into the previous vehicle's work description. Do NOT group all rows into a single vehicle! Each time you see a new vehicle code (with or without a serial number), start a new entry in the entries array.
    
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
