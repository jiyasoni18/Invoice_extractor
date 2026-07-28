import csv
import os
from typing import Dict, Any

class CostLogger:
    def __init__(self, log_file: str = "logs/cost_log.csv"):
        self.log_file = log_file
        self.headers = ["image_name", "model_used", "input_tokens", "output_tokens", "estimated_cost", "time_taken", "routing_decision"]
        self._ensure_log_file()
        
        # In-memory stats for summary
        self.total_docs = 0
        self.total_calls = 0
        self.total_cost = 0.0
        self.models_used = set()

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, image_name: str, stats: Dict[str, Any], routing_decision: str):
        self.total_docs += 1
        model_name = stats.get("model_used")
        if model_name and model_name != "none":
            self.models_used.add(model_name)
        
        # Count API calls. A combined router + extractor run counts as 2 calls unless it was rejected early
        if "rejected" in routing_decision or routing_decision == "error":
            self.total_calls += 1
        elif routing_decision == "invoice_baseline":
            self.total_calls += 1
        else:
            self.total_calls += 2 # Router + Extractor
            
        self.total_cost += float(stats.get("estimated_cost", 0.0))
        
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                image_name,
                stats.get("model_used", "none"),
                stats.get("input_tokens", 0),
                stats.get("output_tokens", 0),
                stats.get("estimated_cost", 0.0),
                stats.get("time_taken", 0.0),
                routing_decision
            ])

    def print_summary(self, mode: str):
        cost_per_doc = self.total_cost / self.total_docs if self.total_docs > 0 else 0.0
        models_str = ", ".join(list(self.models_used)) if self.models_used else "none"
        
        summary_text = (
            f"\n=== Cost summary ===\n"
            f"{mode:<11} models=[{models_str}]\n"
            f"            docs={self.total_docs:>2}  calls={self.total_calls:>3}  total=${self.total_cost:.6f}  cost/doc=${cost_per_doc:.6f}\n\n"
            f"Full per-call log: {self.log_file}\n"
        )
        
        # Print to terminal
        print(summary_text)
        
        # Save to a human-readable text file
        summary_file = os.path.join(os.path.dirname(self.log_file), f"cost_summary_{mode}.txt")
        with open(summary_file, "w") as f:
            f.write(summary_text.strip())
