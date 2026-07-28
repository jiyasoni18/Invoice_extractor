from backend.src.ocr_engine import get_ocr_engine
from backend.src.preprocess import preprocess_image
import os

img_path = "backend/input/optera_doc_29.jpg"
prep_path = "backend/output/test_prep.jpg"

print("Preprocessing...")
preprocess_image(img_path, prep_path)

print("Running OCR...")
ocr = get_ocr_engine()
text = ocr.extract_text(prep_path)
print("EXTRACTED TEXT:\n")
print(text)
