from typing import List, Optional, Union
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Common / Output Schemas
# ---------------------------------------------------------

class RejectedDocument(BaseModel):
    document_type: str = Field(default="non_document", description="Type of the document")
    status: str = Field(default="rejected", description="Status of processing")
    reason: str = Field(description="Reason for rejection, e.g., 'vehicle battery', 'tyre', 'insufficient text'")

# ---------------------------------------------------------
# Invoice Schema
# ---------------------------------------------------------

class InvoiceItem(BaseModel):
    name: Optional[str] = Field(None, description="Name of the product or service")
    quantity: Optional[float] = Field(None, description="Quantity of the item")
    unit_price: Optional[float] = Field(None, description="Price per single unit")
    tax_amount: Optional[float] = Field(None, description="Tax or GST amount for this item")
    amount: Optional[float] = Field(None, description="Total amount for this item including tax")

class InvoiceDocument(BaseModel):
    document_type: str = Field(default="invoice", description="Type of the document")
    supplier_name: Optional[str] = Field(None, description="Name of the supplier, vendor, or company issuing the invoice")
    invoice_no: Optional[str] = Field(None, description="Invoice number")
    date: Optional[str] = Field(None, description="Date of the invoice")
    customer_name: Optional[str] = Field(None, description="Name of the customer or purchasing company")
    vehicle_no: Optional[str] = Field(None, description="Vehicle Registration Number, if present")
    items: List[InvoiceItem] = Field(default_factory=list, description="List of line items in the invoice")
    subtotal: Optional[float] = Field(None, description="Base amount before taxes")
    tax_amount: Optional[float] = Field(None, description="Total tax or GST amount")
    total: Optional[float] = Field(None, description="Final total amount of the invoice")

# ---------------------------------------------------------
# Handwritten / Mechanic Log Schema
# ---------------------------------------------------------

class LogEntry(BaseModel):
    vehicle: Optional[str] = Field(None, description="Vehicle identifier or number plate")
    work: Optional[str] = Field(None, description="Description of the work done")

class HandwrittenLogDocument(BaseModel):
    document_type: str = Field(default="mechanic_log", description="Type of the document")
    date: Optional[str] = Field(None, description="Date of the log entry")
    entries: List[LogEntry] = Field(default_factory=list, description="List of work entries")

# ---------------------------------------------------------
# Meter Reading Schema
# ---------------------------------------------------------

class MeterReadingDocument(BaseModel):
    document_type: str = Field(default="meter_reading", description="Type of the document")
    amount: Optional[float] = Field(None, description="Total amount in currency")
    litres: Optional[float] = Field(None, description="Volume in litres")
    price_per_litre: Optional[float] = Field(None, description="Price per litre")
    urea: Optional[float] = Field(None, description="Urea percentage or amount if applicable")

# ---------------------------------------------------------
# Document Classification Schema (Used by Router)
# ---------------------------------------------------------

class DocumentTypeClassification(BaseModel):
    document_type: str = Field(description="Must be one of: 'invoice', 'mechanic_log', 'meter_reading', 'non_document'")
