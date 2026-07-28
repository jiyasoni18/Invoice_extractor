import logging
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

_trocr_processors = None
_trocr_model = None

def _get_trocr():
    global _trocr_processors, _trocr_model
    if _trocr_processors is None:
        logger.info("Loading TrOCR handwriting model (microsoft/trocr-base-handwritten)...")
        from transformers import ViTImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel
        
        # Bypassing TrOCRProcessor and AutoTokenizer entirely because of a bug in transformers 5.14
        _trocr_processors = {
            "image": ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten"),
            "text": RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
        }
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model.eval()
        logger.info("TrOCR model loaded successfully.")
    return _trocr_processors, _trocr_model

def trocr_read(crop_rgb: np.ndarray) -> str:
    """
    Run TrOCR on a single cropped image region (numpy RGB array).
    Returns the recognised text string.
    """
    import torch
    processors, model = _get_trocr()
    image_processor = processors["image"]
    tokenizer       = processors["text"]
    
    pil_img = Image.fromarray(crop_rgb).convert("RGB")
    pixel_values = image_processor(images=pil_img, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=20)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()
