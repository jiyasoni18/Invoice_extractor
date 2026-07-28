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
            center_y = sum(y_coords) / 4.0
            center_x = sum(x_coords) / 4.0
            height = max(y_coords) - min(y_coords)
            
            blocks.append({
                "text": text,
                "center_y": center_y,
                "center_x": center_x,
                "height": height
            })
            
        blocks.sort(key=lambda b: b["center_y"])
        
        # Calculate median height to make row clustering completely immune to outliers (like tall watermarks)
        median_height = statistics.median([b["height"] for b in blocks]) if blocks else 20.0
        
        lines = []
        current_line = []
        for block in blocks:
            if not current_line:
                current_line.append(block)
            else:
                prev_block = current_line[-1]
                if abs(block["center_y"] - prev_block["center_y"]) < (median_height * 0.5):
                    current_line.append(block)
                else:
                    lines.append(current_line)
                    current_line = [block]
                    
        if current_line:
            lines.append(current_line)
            
        text_rows = []
        for line in lines:
            line.sort(key=lambda b: b["center_x"])
            row_text = "\t".join([b["text"] for b in line])
            if row_text.strip():
                text_rows.append(row_text)
                
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
