import json
from src.ocr_engine import get_ocr_engine
from src.preprocess import preprocess_image

prep = preprocess_image("input/optera_doc_10.jpg", "debug_prep.jpg")
blocks = get_ocr_engine().extract_raw(prep)
for line in blocks[0]:
    box = line[0]
    text = line[1][0]
    y_coords = [p[1] for p in box]
    center_y = sum(y_coords) / 4.0
    height = max(y_coords) - min(y_coords)
    print(f"Text: '{text}', Center Y: {center_y:.1f}, Height: {height:.1f}")
