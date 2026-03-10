import os
import matplotlib.pyplot as plt
import seaborn as sns


class OptunaResearchVisualizationService:

    def __init__(self, output_dir="shared/files/optuna_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def feature_heatmap(self, feature_table):

        if feature_table.empty:
            return

        pivot = feature_table.pivot_table(
            index="feature",
            values="score",
            aggfunc="mean"
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(pivot, annot=True, cmap="viridis")

        plt.title("Feature Score Heatmap")

        path = os.path.join(self.output_dir, "feature_heatmap.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def dataset_ranking(self, series):

        if series.empty:
            return

        plt.figure(figsize=(10, 6))

        series.plot.bar()

        plt.title("Dataset Ranking")

        path = os.path.join(self.output_dir, "dataset_ranking.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def top_feature_combos(self, df):

        if df.empty:
            return

        plt.figure(figsize=(12, 6))

        df.set_index("features")["score"].plot.bar()

        plt.title("Top Feature Combinations")

        path = os.path.join(self.output_dir, "top_feature_combos.png")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()