import os
import matplotlib.pyplot as plt


class OptunaModelComparisonVisualizationService:

    def __init__(self, output_dir="shared/files/optuna_reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_sklearn_features(self, df):

        if df.empty:
            return

        plt.figure(figsize=(12, 6))

        df.head(20).set_index("feature")["count"].plot.bar()

        plt.title("Top Features - Sklearn Models")

        plt.tight_layout()

        path = os.path.join(self.output_dir, "sklearn_feature_ranking.png")

        plt.savefig(path)

        plt.close()

    def plot_lstm_features(self, df):

        if df.empty:
            return

        plt.figure(figsize=(12, 6))

        df.head(20).set_index("feature")["count"].plot.bar()

        plt.title("Top Features - LSTM Models")

        plt.tight_layout()

        path = os.path.join(self.output_dir, "lstm_feature_ranking.png")

        plt.savefig(path)

        plt.close()