import json
import os
from datetime import datetime
import pandas as pd
import math


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ResultLogger:
    def __init__(self, dataset_name: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

        self.base_dir = f"results/experiments/{dataset_name}_{timestamp}"
        self.dataset_name = dataset_name

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(f"{self.base_dir}/plots", exist_ok=True)

        self.results = []

    def log(self, result: dict):
        self.results.append(result)

    def save(self):
        json_path = f"{self.base_dir}/results.json"
        with open(json_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(json_safe(self.results), f, indent=4, ensure_ascii=False, allow_nan=False)
        os.replace(json_path + ".tmp", json_path)

        rows = []

        for r in self.results:
            row = {
                "dataset": r["dataset"],
                "model": r["model"],
                **r["model_metrics"]
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = f"{self.base_dir}/summary.csv"
        df.to_csv(csv_path, index=False)

        print(f"📁 Results saved to: {self.base_dir}")
