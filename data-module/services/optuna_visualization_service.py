import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import pandas as pd


class OptunaVisualizationService:

    def __init__(self, output_dir="shared/files/optuna_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_feature_importance(self, df: pd.DataFrame):

        if df is None or df.empty:
            print("Feature importance skipped (no data)")
            return

        plt.figure(figsize=(12, 6))

        df.head(20).plot.bar(x="feature", y="count")

        plt.title("Feature Importance (Top 20)")
        plt.ylabel("Occurrences in best trials")
        plt.xlabel("Feature")
        plt.xticks(rotation=45)

        plt.tight_layout()

        path = os.path.join(self.output_dir, "feature_importance.png")
        plt.savefig(path)

        plt.close()

        print(f"Saved {path}")

    def plot_preprocessing(self, series: pd.Series):

        if series is None or series.empty:
            print("Preprocessing plot skipped (no data)")
            return

        plt.figure(figsize=(10, 6))

        series.plot.bar()

        plt.title("Preprocessing Impact (Mean Accuracy)")
        plt.ylabel("Accuracy")
        plt.xlabel("Preprocessing method")

        plt.tight_layout()

        path = os.path.join(self.output_dir, "preprocessing_impact.png")
        plt.savefig(path)

        plt.close()

        print(f"Saved {path}")

    def plot_horizon(self, series: pd.Series):

        if series is None or series.empty:
            print("Horizon plot skipped (no data)")
            return

        plt.figure(figsize=(10, 6))

        series.plot(marker="o")

        plt.title("Horizon vs Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Prediction Horizon")

        plt.grid(True)

        plt.tight_layout()

        path = os.path.join(self.output_dir, "horizon_impact.png")
        plt.savefig(path)

        plt.close()

        print(f"Saved {path}")

    def plot_seq_len(self, df: pd.DataFrame):

        if df is None or df.empty:
            print("Seq_len plot skipped (no data)")
            return

        df = df[df["seq_len"].notna()]

        if df.empty:
            print("Seq_len plot skipped (no seq_len values)")
            return

        seq_len_impact = df.groupby("seq_len")["val_loss"].mean()

        plt.figure(figsize=(10, 6))

        seq_len_impact.plot(marker="o")

        plt.title("Sequence Length vs Validation Loss")
        plt.ylabel("Validation Loss")
        plt.xlabel("Sequence Length")

        plt.grid(True)

        plt.tight_layout()

        path = os.path.join(self.output_dir, "seq_len_vs_loss.png")
        plt.savefig(path)

        plt.close()

        print(f"Saved {path}")

    def plot_loss_distribution(self, df: pd.DataFrame):

        if df is None or df.empty:
            print("Loss distribution skipped (no data)")
            return

        df = df[df["val_loss"].notna()]

        if df.empty:
            print("Loss distribution skipped (no val_loss)")
            return

        plt.figure(figsize=(10, 6))

        df["val_loss"].hist(bins=30)

        plt.title("Validation Loss Distribution")
        plt.xlabel("Loss")
        plt.ylabel("Frequency")

        plt.tight_layout()

        path = os.path.join(self.output_dir, "val_loss_distribution.png")
        plt.savefig(path)

        plt.close()

        print(f"Saved {path}")