import matplotlib.pyplot as plt

class ModelTradeoffPlot:

    @staticmethod
    def plot(df, save_path):

        plt.figure(figsize=(8,6))

        plt.scatter(
            df["interpretability"],
            df["accuracy"],
            s=120,
            color="#1f77b4",
            edgecolors="black"
        )

        for _, row in df.iterrows():
            plt.annotate(
                row["model"],
                (row["interpretability"], row["accuracy"]),
                textcoords="offset points",
                xytext=(5,5),
                fontsize=11
            )

        plt.xlabel("Interpretability Score", fontsize=12)
        plt.ylabel("Model Accuracy", fontsize=12)
        plt.title("Accuracy vs Interpretability Trade-off", fontsize=14)

        plt.grid(True, linestyle="--", alpha=0.6)

        plt.xlim(
            df["interpretability"].min() - 0.05,
            df["interpretability"].max() + 0.05
        )

        plt.ylim(
            df["accuracy"].min() - 0.02,
            df["accuracy"].max() + 0.02
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()