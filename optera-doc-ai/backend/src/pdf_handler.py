import os
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def render_pdf_to_images(pdf_path: str, output_dir: str, zoom_factor: float = 2.0) -> list:
    """
    Renders each page of a PDF document to an image file.
    
    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory to save the rendered images
        zoom_factor: Scale factor for rendering. 2.0 is a good balance of speed and OCR accuracy.
        
    Returns:
        A list of paths to the rendered image files.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(pdf_path)
    img_paths = []
    
    try:
        doc = fitz.open(pdf_path)
        logger.info(f"Loaded PDF {pdf_path} with {len(doc)} pages.")
        
        # Matrix to scale the rendering resolution
        mat = fitz.Matrix(zoom_factor, zoom_factor)
        
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            img_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_page_{i}.jpg")
            pix.save(img_path)
            img_paths.append(img_path)
            
        doc.close()
        return img_paths
        
    except Exception as e:
        logger.error(f"Failed to render PDF {pdf_path}: {str(e)}")
        raise e
