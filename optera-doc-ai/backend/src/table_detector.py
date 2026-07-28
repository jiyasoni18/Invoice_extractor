import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_rows(image_path: str) -> list:
    """
    Detects horizontal lines in an image and crops the image row by row.
    Returns a list of numpy arrays (cropped row images).
    """
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to isolate lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)
    
    # Morphological operations to extract horizontal lines
    # A kernel size of width/30 means the line must span at least 1/30th of the page width
    h, w = img.shape[:2]
    horizontal_kernel_size = w // 20 
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find contours of the lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the Y-coordinate of each line
    y_coords = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > w // 10:  # Ignore very short artifact lines
            y_coords.append(y + ch // 2)
            
    # Remove duplicates/close lines
    y_coords.sort()
    merged_y = []
    for y in y_coords:
        if not merged_y or abs(y - merged_y[-1]) > h // 50:
            merged_y.append(y)
            
    if len(merged_y) < 2:
        logger.info(f"[{image_path}] No table rows detected. Using full image.")
        return [img]
        
    # Crop between lines
    rows = []
    # Crop from top of image to first line
    if merged_y[0] > h // 20:
        rows.append(img[0:merged_y[0], :])
        
    for i in range(len(merged_y) - 1):
        y1 = merged_y[i]
        y2 = merged_y[i+1]
        if y2 - y1 > h // 100:  # ignore extremely thin rows
            rows.append(img[y1:y2, :])
            
    # Crop from last line to bottom of image
    if merged_y[-1] < h - (h // 20):
        rows.append(img[merged_y[-1]:h, :])
        
    logger.info(f"[{image_path}] Detected {len(rows)} table rows.")
    return rows
