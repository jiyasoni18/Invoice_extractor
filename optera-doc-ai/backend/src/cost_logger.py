import csv
import os
from typing import Dict, Any

class CostLogger:
    def __init__(self, log_file: str = "logs/cost_log.csv"):
        self.log_file = log_file
        self.headers = ["image_name", "model_used", "input_tokens", "output_tokens", "estimated_cost", "time_taken", "routing_decision"]
        self._ensure_log_file()

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, image_name: str, stats: Dict[str, Any], routing_decision: str):
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
