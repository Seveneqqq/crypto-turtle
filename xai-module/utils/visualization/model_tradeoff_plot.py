import numpy as np
import matplotlib.pyplot as plt


class ModelTradeoffPlot:

    @staticmethod
    def plot(df, output_path):

        if df is None or df.empty:
            print("⚠️ Tradeoff plot skipped: empty dataframe.")
            return

        df = df.copy()

        required_cols = [
            "model",
            "accuracy",
            "interpretability_score"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            print(
                "⚠️ Tradeoff plot skipped: missing columns:",
                missing_cols
            )
            return

        df["accuracy"] = (
            df["accuracy"]
            .replace([np.inf, -np.inf], np.nan)
        )

        df["interpretability_score"] = (
            df["interpretability_score"]
            .replace([np.inf, -np.inf], np.nan)
        )

        df = df.dropna(
            subset=["accuracy", "interpretability_score"]
        )

        if df.empty:
            print(
                "⚠️ Tradeoff plot skipped: "
                "no finite accuracy/interpretability values."
            )
            return

        plt.figure(figsize=(8, 6))

        plt.scatter(
            df["accuracy"],
            df["interpretability_score"]
        )

        for _, row in df.iterrows():
            plt.text(
                row["accuracy"],
                row["interpretability_score"],
                str(row["model"])
            )

        x_min = df["accuracy"].min()
        x_max = df["accuracy"].max()

        y_min = df["interpretability_score"].min()
        y_max = df["interpretability_score"].max()

        if x_min == x_max:
            x_min -= 0.01
            x_max += 0.01

        if y_min == y_max:
            y_min -= 0.01
            y_max += 0.01

        plt.xlim(x_min - 0.02, x_max + 0.02)
        plt.ylim(y_min - 0.02, y_max + 0.02)

        plt.xlabel("Accuracy / model score")
        plt.ylabel("Interpretability score")
        plt.title("Accuracy vs Interpretability")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()