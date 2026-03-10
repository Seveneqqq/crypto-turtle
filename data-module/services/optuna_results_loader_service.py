import os
import json
import pandas as pd


class OptunaResultsLoaderService:

    def __init__(self):
        self.log_dirs = [
            "shared/files/processing_optuna_log",
            "shared/files/processing_optuna_lstm_log"
        ]

    def load_all_results(self) -> pd.DataFrame:

        rows = []

        for log_dir in self.log_dirs:

            if not os.path.exists(log_dir):
                continue

            for file in os.listdir(log_dir):

                if not (file.endswith(".json") or file.endswith(".log")):
                    continue

                path = os.path.join(log_dir, file)

                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Skipping file {file}: {e}")
                    continue

                trials = data.get("top_trials") or data.get("top_k_feature_combinations")

                if not trials:
                    continue

                for t in trials:
                    rows.append(t)

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # -------------------------------------------------
        # ZAGWARANTUJ ISTNIENIE KLUCZOWYCH KOLUMN
        # -------------------------------------------------

        if "accuracy" not in df.columns:
            df["accuracy"] = None

        if "val_loss" not in df.columns:
            df["val_loss"] = None

        if "features" not in df.columns:
            df["features"] = None

        if "feature_combo" not in df.columns:
            df["feature_combo"] = None

        # -------------------------------------------------
        # NORMALIZACJA SCORE
        # -------------------------------------------------

        df["score"] = None

        df.loc[df["accuracy"].notna(), "score"] = df["accuracy"]
        df.loc[df["val_loss"].notna(), "score"] = -df["val_loss"]

        # -------------------------------------------------
        # CZYŚCZENIE WARTOŚCI
        # -------------------------------------------------

        df = df.replace([float("inf"), float("-inf")], None)
        df = df.where(pd.notnull(df), None)

        return df