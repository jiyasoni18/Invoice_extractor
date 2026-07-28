import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path: str, output_path: str) -> str:
    """
    Preprocess the image for better OCR accuracy.
    - Resizes if too large
    - Converts to grayscale
    - Applies simple denoising
    Returns the path to the preprocessed image.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return image_path
            
        # Resize if width or height is greater than 1600
        # PaddleOCR's DBNet is optimized for smaller resolutions. Forcefully upscaling causes detection failures.
        max_dim = 1600
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # --- Deskewing Logic (Rotation Correction) ---
        # Threshold the image to find text regions
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) > 0:
            # Calculate the minimum bounding box that contains all text
            angle = cv2.minAreaRect(coords)[-1]
            # OpenCV 4.5+ returns angles in [0, 90)
            if angle > 45:
                angle = angle - 90
                
            # Only deskew if the tilt is small (e.g., between 0.5 and 20 degrees)
            # Deskewing is for fixing slight page tilts, not full 90-degree flips
            if 0.5 < abs(angle) < 20:
                logger.info(f"Deskewing image by {angle:.2f} degrees")
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                # Re-compute grayscale after rotation
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        # --- Step 1: Better Image Enhancement (Adaptive Thresholding) ---
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        enhanced = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            15
        )
            
        # Save processed binary image
        cv2.imwrite(output_path, enhanced)
        return output_path
    except Exception as e:
        logger.error(f"Error in preprocessing {image_path}: {e}")
        return image_path
