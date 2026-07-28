import logging
import re
import unicodedata
from typing import List, Optional
from bs4 import BeautifulSoup
from .schemas import InvoiceItem

logger = logging.getLogger(__name__)

# Try to import the OCR engine we use globally to share the PPStructure instance
try:
    from .ocr_engine import get_ocr_engine
except ImportError:
    # Fallback in case of weird import cycles
    def get_ocr_engine():
        return None

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.lower().strip()

NUMERIC_FIELDS = {"quantity", "unit_price", "tax_amount", "amount"}
MONEY_RE = re.compile(r"(\d{1,3}(?:[.,\s]\d{2,3})*[.,]\d{1,3}|\d+[.,]\d{1,3}|\d+)")

def _parse_number(raw: str) -> Optional[float]:
    if raw is None:
        return None
    raw = raw.strip().replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("%", "").strip()
    if not raw:
        return None
        
    # Handle Indian numbering system (e.g. 1,00,000.00)
    # If the string has a comma and a period, assume comma is thousands separator
    if "," in raw and "." in raw:
        if raw.rindex(",") > raw.rindex("."):
            # Comma is the decimal separator (European style, less common in India but possible)
            raw = raw.replace(".", "").replace(",", ".")
        else:
            # Comma is thousands separator
            raw = raw.replace(",", "")
    elif "," in raw and "." not in raw:
        # Check if it looks like a decimal comma (e.g., 12,34) or thousands (e.g., 12,000)
        # In India, usually 12,000 means thousands. We'll assume thousands unless it has exactly 2 digits after comma.
        if re.search(r",\d{2}$", raw):
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")

    match = MONEY_RE.search(raw)
    if not match:
        return None
    try:
        # We've already replaced thousands separators with nothing, so any remaining commas can be safely ignored
        val = match.group(1).replace(" ", "").replace(",", "")
        return float(val)
    except ValueError:
        return None

INVOICE_ITEM_HEADERS = {
    "name": ["description", "product", "item", "particulars", "name", "hsn", "sac"],
    "quantity": ["qty", "quantity", "nos", "unit", "units", "vol"],
    "unit_price": ["rate", "price", "unit price", "mrp", "price/unit"],
    "tax_amount": ["cgst", "sgst", "igst", "tax", "gst", "vat", "tax amt"],
    "amount": ["amount", "total", "gross", "value", "net", "total value", "final price"]
}

def _map_header_to_field(header_text: str) -> Optional[str]:
    if not header_text:
        return None
    header_norm = _normalize(header_text)
    for field_name, synonyms in INVOICE_ITEM_HEADERS.items():
        for syn in synonyms:
            if _normalize(syn) in header_norm or header_norm in _normalize(syn):
                return field_name
    return None

def _html_table_to_rows(html: str) -> List[List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    return rows

def rows_to_invoice_items(rows: List[List[str]]) -> List[InvoiceItem]:
    if len(rows) < 2:
        return []

    header_row = rows[0]
    column_fields = [_map_header_to_field(h) for h in header_row]

    if all(f is None for f in column_fields):
        return []

    SUMMARY_ROW_KEYWORDS = ("subtotal", "total", "cgst", "sgst", "igst", "tax", "balance", "amount chargeable")

    line_items = []
    for data_row in rows[1:]:
        row_text_norm = _normalize(" ".join(c for c in data_row if c))
        # Skip summary rows that might have been detected as part of the table
        if any(kw in row_text_norm for kw in SUMMARY_ROW_KEYWORDS) and len(row_text_norm) < 30:
            # Only skip if it's a short summary row (to avoid skipping an item named "total engine oil")
            if any(row_text_norm.startswith(kw) for kw in SUMMARY_ROW_KEYWORDS):
                continue
                
        kwargs = {}
        # In Indian invoices, there might be multiple tax columns (CGST, SGST). 
        # We'll sum them up into a single tax_amount.
        tax_sum = 0.0
        
        for col_idx, field_name in enumerate(column_fields):
            if field_name is None or col_idx >= len(data_row):
                continue
            raw_value = data_row[col_idx]
            if raw_value is None or not raw_value.strip():
                continue
                
            if field_name in NUMERIC_FIELDS:
                parsed_val = _parse_number(raw_value)
                if parsed_val is not None:
                    if field_name == "tax_amount":
                        tax_sum += parsed_val
                    else:
                        kwargs[field_name] = parsed_val
            else:
                # Append to existing name if multiple columns map to "name" (e.g. Item Name + HSN)
                if field_name == "name" and "name" in kwargs:
                    kwargs["name"] = f"{kwargs['name']} - {raw_value.strip()}"
                else:
                    kwargs[field_name] = raw_value.strip()

        if tax_sum > 0:
            kwargs["tax_amount"] = tax_sum

        # We must have at least a name or an amount to consider it a valid line item
        if kwargs.get("name") or kwargs.get("amount"):
            line_items.append(InvoiceItem(**kwargs))

    return line_items

def extract_invoice_items(image_path: str) -> List[InvoiceItem]:
    """
    Extracts structured InvoiceItems from an image using PPStructure and BeautifulSoup.
    """
    engine = get_ocr_engine()
    if not engine or not hasattr(engine, 'table_engine'):
        logger.error("OCR table engine not initialized.")
        return []
        
    try:
        result_structure = engine.table_engine(image_path)
    except Exception as e:
        logger.exception(f"PPStructure table inference failed for {image_path}: {e}")
        return []
        
    # Find the largest HTML table by string length
    html_candidates = []
    for region in result_structure:
        if region.get('type') == 'table' and 'res' in region and 'html' in region['res']:
            html = region['res']['html']
            html_candidates.append((len(html), html))
            
    if not html_candidates:
        logger.info(f"[{image_path}] No HTML table found by PPStructure.")
        return []
        
    largest_html = max(html_candidates, key=lambda c: c[0])[1]
    rows = _html_table_to_rows(largest_html)
    
    items = rows_to_invoice_items(rows)
    logger.info(f"[{image_path}] Extracted {len(items)} line items from table.")
    return items

def extract_rows(image_path: str) -> list:
    """
    Legacy fallback: returns the original image in a list.
    (Kept for backwards compatibility if needed somewhere).
    """
    import cv2
    img = cv2.imread(image_path)
    return [img] if img is not None else []
