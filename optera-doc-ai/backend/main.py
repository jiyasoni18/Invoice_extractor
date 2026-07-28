import os
import argparse
import logging
import json
import time

from src.preprocess import preprocess_image
from src.ocr_engine import get_ocr_engine
from src.detector import is_document
from src.router import classify_document
from src.extractors.parsers import parse_invoice, parse_handwritten_log, parse_handwritten_log_vision, parse_meter_reading
from src.llm_client import call_gemini_vision
from src.schemas import RejectedDocument
from src.cost_logger import CostLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def process_image(image_path: str, output_dir: str, cost_logger: CostLogger, update_state=None, mode: str = "optimized"):
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
    
    start_time = time.time()
    stages = []
    
    def _add_stage(name, status, details):
        stages.append({"name": name, "status": status, "details": details})
        if update_state:
            update_state({"stages": stages.copy(), "done": False, "final_json": None})

    try:
        if mode == "baseline":
            _add_stage("Baseline VLM", "success", "Bypassing OCR and sending directly to Vision Language Model.")
            system_prompt = """
            You are an AI trained to extract structured data from challenging invoice OCR text.
            Extract the vendor, invoice_no, date, customer, items (name, quantity, amount), and total.
            Return strictly as a JSON object matching this schema:
            {
              "document_type": "invoice",
              "vendor": "string",
              "invoice_no": "string",
              "date": "string",
              "customer": "string",
              "items": [{"name": "string", "quantity": number, "amount": number}],
              "total": number
            }
            """
            text_content, extract_stats = call_gemini_vision(image_path, system_prompt)
            result_json = json.loads(text_content)
            
            result_json["_debug_model_used"] = extract_stats.get("model_used", "gemini-1.5-flash-vision")
            result_json["_debug_mode"] = "baseline"
            
            with open(output_path, 'w') as f:
                json.dump(result_json, f, indent=2)
                
            cost_logger.log(base_name, extract_stats, "invoice_baseline")
            _add_stage("Structured JSON Extraction", "success", "Successfully mapped into invoice schema using VLM.")
            if update_state: update_state({"stages": stages, "done": True, "final_json": result_json})
            return {"stages": stages, "final_json": result_json}
            
        # --- OPTIMIZED PATH (OCR + LLM) ---
        # 1. Preprocess
        prep_image_path = preprocess_image(image_path, os.path.join(output_dir, f"prep_{base_name}"))
        _add_stage("Preprocessing", "success", "Image resized and grayscaled.")
        
        # 2. OCR (Single pass for everything)
        ocr_engine = get_ocr_engine()
        ocr_text = ocr_engine.extract_text(prep_image_path)
        
        _add_stage("OCR Text Extraction", "success", f"Extracted {len(ocr_text)} characters using PaddleOCR.")
        
        # Save raw OCR text to a dedicated folder for user inspection
        ocr_texts_dir = os.path.join(output_dir, "ocr_texts")
        os.makedirs(ocr_texts_dir, exist_ok=True)
        ocr_txt_path = os.path.join(ocr_texts_dir, f"{os.path.splitext(base_name)[0]}.txt")
        with open(ocr_txt_path, 'w') as txt_file:
            txt_file.write(ocr_text)
        
        # 3. Density check (Free filter)
        is_valid, reason = is_document(ocr_text)
        if not is_valid:
            logger.info(f"[{base_name}] Rejected: {reason}")
            _add_stage("Density Filter", "error", reason)
            result = RejectedDocument(reason=reason).model_dump()
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            cost_logger.log(base_name, {"time_taken": time.time() - start_time}, "rejected_density")
            if update_state: update_state({"stages": stages, "done": True, "final_json": result})
            return {"stages": stages, "final_json": result}
        else:
            _add_stage("Density Filter", "success", "Passed text density check.")
            
        # 4. Classify (Small LLM)
        doc_type, class_stats = classify_document(ocr_text)
        logger.info(f"[{base_name}] Classified as: {doc_type}")
        _add_stage("Document Classification", "success", f"Classified as '{doc_type}'.")
        
        if doc_type == "non_document":
            result = RejectedDocument(reason="Classified as non-document by router").model_dump()
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            class_stats["time_taken"] = time.time() - start_time
            cost_logger.log(base_name, class_stats, "rejected_router")
            if update_state: update_state({"stages": stages, "done": True, "final_json": result})
            return {"stages": stages, "final_json": result}
            
        # 5. Extract
        result_json = {}
        extract_stats = {}
        if doc_type == "invoice":
            result_json, extract_stats = parse_invoice(ocr_text)
        elif doc_type == "mechanic_log":
            # The user explicitly requested to stick with OCR + Text LLM for handwritten logs.
            # We pass the OCR text to OpenRouter.
            result_json, extract_stats = parse_handwritten_log(ocr_text)
        elif doc_type == "meter_reading":
            result_json, extract_stats = parse_meter_reading(ocr_text)
        else:
            result_json, extract_stats = parse_invoice(ocr_text) # fallback
            
        # Inject debugging metadata for the user
        result_json["_debug_raw_ocr"] = ocr_text
        result_json["_debug_model_used"] = extract_stats.get("model_used", "unknown")
            
        # Write output
        with open(output_path, 'w') as f:
            json.dump(result_json, f, indent=2)
            
        # Combine stats
        total_input = class_stats.get("input_tokens", 0) + extract_stats.get("input_tokens", 0)
        total_output = class_stats.get("output_tokens", 0) + extract_stats.get("output_tokens", 0)
        total_cost = class_stats.get("estimated_cost", 0.0) + extract_stats.get("estimated_cost", 0.0)
        
        combined_stats = {
            "model_used": extract_stats.get("model_used"),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "estimated_cost": total_cost,
            "time_taken": time.time() - start_time
        }
        cost_logger.log(base_name, combined_stats, doc_type)
        logger.info(f"[{base_name}] Successfully processed as {doc_type}")
        
        _add_stage("Structured JSON Extraction", "success", f"Successfully mapped into {doc_type} schema.")
        if update_state: update_state({"stages": stages, "done": True, "final_json": result_json})
        return {"stages": stages, "final_json": result_json}
        
    except Exception as e:
        logger.error(f"[{base_name}] Pipeline failed: {e}")
        _add_stage("Error", "error", str(e))
        result = RejectedDocument(reason=f"Pipeline failed: {str(e)}").model_dump()
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        cost_logger.log(base_name, {"time_taken": time.time() - start_time}, "error")
        if update_state: update_state({"stages": stages, "done": True, "final_json": result})
        return {"stages": stages, "final_json": result}

def main():
    parser = argparse.ArgumentParser(description="Optera Document AI Pipeline")
    parser.add_argument("--mode", type=str, default="optimized", help="Run mode (optimized or baseline)")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = "output"
    
    os.makedirs(output_dir, exist_ok=True)
    cost_logger = CostLogger()
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            logger.info(f"Processing {image_path}...")
            process_image(image_path, output_dir, cost_logger)
            
if __name__ == "__main__":
    main()
