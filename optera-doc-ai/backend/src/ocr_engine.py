import logging
import json
import re
from paddleocr import PaddleOCR
import statistics

logger = logging.getLogger(__name__)

class OCREngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Initializing PaddleOCR Engine and PPStructure...")
            cls._instance = super(OCREngine, cls).__new__(cls)
            cls._instance.ocr = PaddleOCR(use_angle_cls=True, lang='en')
            
            from paddleocr import PPStructure
            cls._instance.table_engine = PPStructure(layout=True, table=True, ocr=True, recovery=False, lang='en')
            
        return cls._instance

    def extract_text(self, image_path: str) -> str:
        """
        Runs PPStructure on the given image path and returns the full extracted text.
        Tables are natively returned as HTML `<table>` blocks to perfectly preserve column structure.
        """
        result_basic = self.ocr.ocr(image_path)
        if not result_basic or not result_basic[0]:
            return ""
            
        # Now run PPStructure to extract layout and tables natively
        result_structure = self.table_engine(image_path)
        tables_html = []
        for region in result_structure:
            if region['type'] == 'table':
                tables_html.append(region['res']['html'])
                
        # Now run our manual bounding-box clustering for the ENTIRE document.
        # This perfectly handles borderless receipts and messy columns that PPStructure misses.
        blocks = []
        for line in result_basic[0]:
            box = line[0]
            text = line[1][0]
            y_coords = [point[1] for point in box]
            x_coords = [point[0] for point in box]
            min_y, max_y = min(y_coords), max(y_coords)
            min_x, max_x = min(x_coords), max(x_coords)
            center_y = sum(y_coords) / 4.0
            height = max_y - min_y
            
            blocks.append({
                "text": text,
                "center_y": center_y,
                "min_x": min_x,
                "max_x": max_x,
                "height": height
            })
            
        blocks.sort(key=lambda b: b["center_y"])
        
        median_height = statistics.median([b["height"] for b in blocks]) if blocks else 20.0
        
        lines = []
        current_line = []
        for block in blocks:
            if not current_line:
                current_line.append(block)
            else:
                prev_block = current_line[-1]
                # If it's within half a line's height, it's the same line
                if abs(block["center_y"] - prev_block["center_y"]) < (median_height * 0.5):
                    current_line.append(block)
                else:
                    lines.append(current_line)
                    current_line = [block]
                    
        if current_line:
            lines.append(current_line)
            
        text_rows = []
        # Estimate average character width to convert pixel distance to spaces
        # Assuming ~10 pixels per character as a rough estimate
        char_width_px = 12.0
        
        for line in lines:
            line.sort(key=lambda b: b["min_x"])
            row_str = ""
            last_x = 0
            for b in line:
                if last_x > 0:
                    # Calculate gap in pixels between the end of the last word and the start of this one
                    gap_px = b["min_x"] - last_x
                    spaces_to_add = max(1, int(gap_px / char_width_px))
                    # Cap massive spaces to avoid breaking LLM context with huge empty gaps
                    spaces_to_add = min(spaces_to_add, 15)
                    row_str += " " * spaces_to_add
                row_str += b["text"]
                last_x = b["max_x"]
                
            if row_str.strip():
                text_rows.append(row_str)
                
        full_text = "\n".join(text_rows)
        
        # Combine the perfectly clustered raw text with the PPStructure HTML tables
        if tables_html:
            full_text += "\n\n[DETECTED HTML TABLES]\n" + "\n".join([f"<table>\n{t}\n</table>" for t in tables_html])
            
        return full_text

    def extract_raw(self, image_path: str) -> list:
        """
        Runs OCR and returns the raw result (useful if bounding boxes or confidences are needed).
        """
        return self.ocr.ocr(image_path)

    def extract_text_from_rows(self, row_images: list) -> str:
        """
        Runs OCR on a list of cropped row images sequentially.
        This perfectly preserves the table layout since we OCR row by row.
        """
        row_texts = []
        for row_img in row_images:
            result = self.ocr.ocr(row_img)
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
