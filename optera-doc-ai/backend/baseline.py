import os
import argparse
import logging
import json
import time

from src.llm_client import call_gemini_vision
from src.cost_logger import CostLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def process_baseline(image_path: str, output_dir: str, cost_logger: CostLogger):
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
    
    start_time = time.time()
    
    system_prompt = """
    You are an AI trained to extract structured data from documents.
    If the document is an invoice, extract the vendor, invoice_no, date, customer, items (name, quantity, amount), and total.
    If the document is a mechanic log, extract the date and entries (vehicle, work).
    Return strictly as a JSON object matching the appropriate schema.
    """
    
    try:
        text_content, extract_stats = call_gemini_vision(image_path, system_prompt)
        
        try:
            result_json = json.loads(text_content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON for {base_name}: {text_content}")
            result_json = {"error": "Invalid JSON response from LLM", "raw_text": text_content}
        
        result_json["_debug_model_used"] = extract_stats.get("model_used", "gemini-1.5-flash-vision")
        result_json["_debug_mode"] = "baseline"
        
        with open(output_path, 'w') as f:
            json.dump(result_json, f, indent=2)
            
        extract_stats["time_taken"] = time.time() - start_time
        cost_logger.log(base_name, extract_stats, "invoice_baseline")
        logger.info(f"[{base_name}] Successfully processed via baseline Vision LLM.")
        
    except Exception as e:
        logger.error(f"[{base_name}] Baseline Pipeline failed: {e}")
        cost_logger.log(base_name, {"time_taken": time.time() - start_time}, "error")


def main():
    parser = argparse.ArgumentParser(description="Baseline VLM Document Pipeline (Directly to LLM without OCR)")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    args = parser.parse_args()
    
    input_dir = args.input
    output_dir = "output_baseline"
    
    os.makedirs(output_dir, exist_ok=True)
    cost_logger = CostLogger(log_file="logs/cost_log_baseline.csv")
    
    logger.info("=========================================")
    logger.info("RUNNING BASELINE VLM-ONLY EXTRACTOR")
    logger.info("=========================================")
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        logger.info(f"Processing {image_path}...")
        process_baseline(image_path, output_dir, cost_logger)
        
    cost_logger.print_summary("baseline")
            
if __name__ == "__main__":
    main()
