import logging
import json
import re
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class OCREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing PaddleOCR Engine with Angle Classifier enabled...")
            cls._instance = super(OCREngine, cls).__new__(cls)
            # Re-enabling use_angle_cls=True to automatically rotate 90/180/270 degree images
            cls._instance.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        return cls._instance

    def extract_text(self, image_path: str) -> str:
        """
        Runs OCR on the given image path and returns the full extracted text as a single string.
        """
        result = self.ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
            
        # Detect if the image is sideways (landscape text)
        # If height > width for most text boxes, the image is rotated 90 or 270 degrees.
        vertical_count = 0
        horizontal_count = 0
        for line in result[0]:
            box = line[0]
            w = max([p[0] for p in box]) - min([p[0] for p in box])
            h = max([p[1] for p in box]) - min([p[1] for p in box])
            
            # Stricter bounds to prevent false positives on single handwritten letters
            if h > w * 2.5:
                vertical_count += 1
            elif w > h * 1.5:
                horizontal_count += 1
                
        # Only rotate if overwhelmingly vertical
        if vertical_count > horizontal_count * 3:
            logger.info("Image appears to be sideways! Rotating 90 degrees and re-running OCR...")
            import cv2
            img = cv2.imread(image_path)
            # Rotate 90 degrees clockwise
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(image_path, img)
            # Re-run OCR on the newly rotated image
            result = self.ocr.ocr(image_path, cls=True)
            if not result or not result[0]:
                return ""
            
        # result[0] is a list of [box, (text, confidence)]
        # box is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # We need to sort and group them into lines to preserve table structures.
        
        # Calculate bounding box centers and heights
        blocks = []
        for line in result[0]:
            box = line[0]
            text = line[1][0]
            
            # Get center y and center x
            y_coords = [point[1] for point in box]
            x_coords = [point[0] for point in box]
            center_y = sum(y_coords) / 4.0
            center_x = sum(x_coords) / 4.0
            height = max(y_coords) - min(y_coords)
            
            blocks.append({
                "text": text,
                "center_y": center_y,
                "center_x": center_x,
                "height": height
            })
            
        # Sort blocks by Y coordinate first
        blocks.sort(key=lambda b: b["center_y"])
        
        lines = []
        current_line = []
        
        for block in blocks:
            if not current_line:
                current_line.append(block)
            else:
                # If the center_y of this block is within half the height of the previous block, 
                # they are on the same line.
                prev_block = current_line[-1]
                if abs(block["center_y"] - prev_block["center_y"]) < (prev_block["height"] * 0.5):
                    current_line.append(block)
                else:
                    lines.append(current_line)
                    current_line = [block]
                    
        if current_line:
            lines.append(current_line)
            
        # Step 4: Fuzzy correct the first word of each line and Step 5: Format as JSON array
        json_rows = []
        for i, line in enumerate(lines):
            line.sort(key=lambda b: b["center_x"])
            
            # Fuzzy correct the first word (likely the vehicle code)
            if line:
                first_word = line[0]["text"]
                # Match 2-4 letters followed by 2-3 characters that might be numbers or misread letters
                match = re.match(r'^([A-Za-z]{2,4})([0-9A-Za-z]{2,3})$', first_word)
                if match:
                    prefix, suffix = match.groups()
                    prefix = prefix.upper()
                    # Replace common OCR letter-to-number mistakes
                    suffix = suffix.upper().replace('S', '5').replace('O', '0').replace('L', '1').replace('I', '1')
                    line[0]["text"] = prefix + suffix
                    
            # Join with spaces (JSON structure replaces the need for tabs/newlines)
            row_text = " ".join([b["text"] for b in line])
            json_rows.append({"row": i + 1, "text": row_text})
            
        return json.dumps(json_rows, indent=2)

    def extract_raw(self, image_path: str) -> list:
        """
        Runs OCR and returns the raw result (useful if bounding boxes or confidences are needed).
        """
        return self.ocr.ocr(image_path, cls=True)

    def extract_text_from_rows(self, row_images: list) -> str:
        """
        Runs OCR on a list of cropped row images sequentially.
        This perfectly preserves the table layout since we OCR row by row.
        """
        row_texts = []
        for row_img in row_images:
            result = self.ocr.ocr(row_img, cls=True)
            if not result or not result[0]:
                continue
            
            # Extract text from this specific row and sort purely left-to-right
            blocks = []
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                # x-coordinate
                center_x = sum([p[0] for p in box]) / 4.0
                blocks.append({"text": text, "x": center_x})
                
            blocks.sort(key=lambda b: b["x"])
            row_text = "\t".join([b["text"] for b in blocks])
            if row_text.strip():
                row_texts.append(row_text)
                
        return "\n".join(row_texts)

# Global instance getter
def get_ocr_engine():
    return OCREngine()
