import logging
import re
from typing import Dict, Any, Tuple
from src.llm_client import extract_json_with_llm
from src.schemas import InvoiceDocument, HandwrittenLogDocument, MeterReadingDocument

logger = logging.getLogger(__name__)

def parse_invoice(ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system_prompt = """
    You are an expert AI trained to extract structured data from challenging, hybrid (printed + handwritten) invoice OCR text.
    Extract the supplier_name, supplier_gstin, invoice_no, date, customer_name, customer_gstin, payment_mode, vehicle_no, items (name, hsn_code, quantity, unit_price, tax_amount, discount, amount), subtotal, tax_amount, and total.
    
    CRITICAL INSTRUCTIONS for field mapping:
    - GENERAL: The OCR text is severely corrupted and may contain random symbols or mashed characters. You MUST aggressively infer the real words. For example, "SHIIV SHAKI! Tजायगन MगoRS" should be corrected to "SHIV SHAKTI MOTORS". Never return null if there is a highly probable match.
    - SUPPLIER_NAME: This is usually the largest printed text at the very top of the page (e.g., a store or company name). Aggressively fix typos.
    - CUSTOMER_NAME: This is usually next to labels like "NAME:", "M/s", or "CUSTOMER:". Do NOT confuse the Customer with the Supplier.
    - GSTIN: Extract the 15-character alphanumeric GSTIN for the supplier and the customer (if present).
    - PAYMENT_MODE: Extract if the sale is CREDIT, CASH, UPI, etc.
    - INVOICE_NO: This is usually a number near "INVOICE NO:" or printed alone in a top corner. Do NOT confuse the invoice number with the Total Amount!
    - VEHICLE_NO: Look for "Vehicle Reg", "Vehicle No", etc.
    - ITEMS: Look for grid-like data. Extract the product name, HSN/SAC code, quantity, rate, discount, and amount. Split the items logically.
    - TAXABLE_VALUE: The base amount before taxes are applied (often called "Taxable Value" or "Subtotal").
    - TAX BREAKDOWN: Extract the exact CGST, SGST, and IGST amounts from the bottom of the invoice or the tax summary table.
    - TOTAL: Look near the bottom for "Total", "Balance", or the final summed amount (subtotal + tax).
    
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "invoice",
      "supplier_name": "string",
      "supplier_gstin": "string",
      "invoice_no": "string",
      "date": "string",
      "customer_name": "string",
      "customer_gstin": "string",
      "payment_mode": "string",
      "vehicle_no": "string",
      "items": [{"name": "string", "hsn_code": "string", "quantity": number, "unit_price": number, "tax_amount": number, "discount": number, "amount": number}],
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

def parse_handwritten_invoice_vision(image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from src.llm_client import call_vision_model
    import json
    
    system_prompt = """
    You are an expert AI trained to extract structured data from challenging, highly cursive handwritten invoices.
    Extract the supplier_name, supplier_gstin, invoice_no, date, customer_name, customer_gstin, payment_mode, vehicle_no, items (name, hsn_code, quantity, unit_price, tax_amount, discount, amount), subtotal, tax_amount, and total.
    
    CRITICAL INSTRUCTIONS:
    - This invoice is heavily handwritten. Look closely at the handwriting for the items, rates, and amounts.
    - Ensure perfect accuracy in spelling and numbers for handwritten items.
    
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "invoice",
      "supplier_name": "string",
      "supplier_gstin": "string",
      "invoice_no": "string",
      "date": "string",
      "customer_name": "string",
      "customer_gstin": "string",
      "payment_mode": "string",
      "vehicle_no": "string",
      "items": [{"name": "string", "hsn_code": "string", "quantity": number, "unit_price": number, "tax_amount": number, "discount": number, "amount": number}],
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
    
    text_content, stats = call_vision_model(image_path, system_prompt)
    parsed_json = json.loads(text_content)
    validated = InvoiceDocument(**parsed_json).model_dump()
    return validated, stats

def parse_handwritten_log(ocr_text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system_prompt = """
    You are an AI trained to extract structured data from handwritten mechanic logs.
    Extract the date and all work entries (vehicle identifier and work description).
    
    CRITICAL INSTRUCTIONS for field mapping:
    - VEHICLE: The OCR for cursive text is often severely corrupted. The vehicle code can be alphanumeric (e.g., TAM23, MAM38) or purely numeric (e.g., 42, 28, 11).
      IMPORTANT: The vehicle code is SOMETIMES preceded by a serial number or bullet point like "(03)", "1)", "(7)", "44)", "09)", etc. 
      If there is a serial number, extract the code immediately following it. If there is NO serial number (e.g. the very first row), extract the code itself as the VEHICLE.
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
    from src.llm_client import call_vision_model
    import json
    
    system_prompt = """
    You are an expert AI trained to extract structured data from diverse and highly cursive handwritten mechanic logs. 
    Mechanic logs vary wildly in format. You must adapt to the document in front of you.

    UNIVERSAL EXTRACTION RULES:
    1. LAYOUT FLEXIBILITY: 
       - Column Format: If the log has columns (like SR, BUS NO, MECH, WORK DONE), extract the vehicle from the BUS NO column and the work from the WORK DONE column (including the mechanic's name if relevant).
       - Free-form Format: If the log lacks columns and is just a list, each entry will start with the vehicle identifier followed by the work description.
    
    2. VEHICLE IDENTIFIER FLEXIBILITY:
       - Formats vary by fleet. It might be a purely numeric ID (e.g., "42", "28"), a 3-letter prefix with digits (e.g., "TCM 47", "MAM 23", "TAM 06"), or alphanumeric.
       - NEVER hallucinate or assume a prefix (like adding "MAM" everywhere). Extract EXACTLY the characters written on the page for that specific row.
       
    3. WORK DESCRIPTION & MULTI-LINGUAL:
       - Extract the work description exactly as written. Do not hallucinate or repeat text.
       - If a description spans multiple lines without a new vehicle identifier, combine it with the previous vehicle's entry.
       - Do not skip rows written in non-English scripts (like Hindi/Devanagari). Extract them just as you would English text.
       
    4. COMPLETENESS:
       - Extract ALL entries present on the page. Do not skip any rows.
    
    Return strictly as a JSON object matching this schema:
    {
      "document_type": "mechanic_log",
      "date": "string",
      "entries": [{"vehicle": "string", "work": "string"}]
    }
    If a field is missing, set it to null.
    """
    
    text_content, stats = call_vision_model(image_path, system_prompt)
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

def parse_vision_fallback(image_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    A universal safety net that uses the Vision Model to dynamically determine 
    if a garbled document is an Invoice or a Mechanic Log, and extracts it accordingly.
    """
    from src.llm_client import call_vision_model
    import json
    
    system_prompt = """
    You are an expert AI fallback parser. A previous text-only extraction failed because the document is likely highly cursive, handwritten, or rotated.
    
    Look at the image and determine its true type:
    1. If it looks like a bill of sale, receipt, or invoice (contains items, rates, totals, shop names):
       Return STRICTLY this JSON schema:
       {
         "document_type": "invoice",
         "supplier_name": "string", "supplier_gstin": "string", "invoice_no": "string", "date": "string",
         "customer_name": "string", "customer_gstin": "string", "payment_mode": "string", "vehicle_no": "string",
         "items": [{"name": "string", "hsn_code": "string", "quantity": number, "unit_price": number, "tax_amount": number, "discount": number, "amount": number}],
         "subtotal": number, "taxable_value": number, "cgst": number, "sgst": number, "igst": number, "tax_amount": number, "total": number
       }
       
    2. If it looks like a mechanic's daily log or repair list (lists of vehicle numbers and work descriptions without standard billing totals):
       Return STRICTLY this JSON schema:
       {
         "document_type": "mechanic_log",
         "date": "string",
         "entries": [{"vehicle": "string", "work": "string"}]
       }
       
    Do not hallucinate data. If a field is missing, set it to null. Return ONLY the raw JSON.
    """
    
    text_content, stats = call_vision_model(image_path, system_prompt)
    parsed_json = json.loads(text_content)
    
    # Validate based on what the Vision Model decided
    if parsed_json.get("document_type") == "mechanic_log":
        validated = HandwrittenLogDocument(**parsed_json).model_dump()
    else:
        validated = InvoiceDocument(**parsed_json).model_dump()
        
    return validated, stats
