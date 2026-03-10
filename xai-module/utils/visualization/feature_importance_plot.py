import matplotlib.pyplot as plt
import pandas as pd


class FeatureImportancePlot:

    @staticmethod
    def plot(importance_dict, save_path, top_n=15):

        items = list(importance_dict.items())[:top_n]

        features = [x[0] for x in items]
        values = [x[1] for x in items]

        plt.figure(figsize=(8,6))

        plt.barh(features[::-1], values[::-1])

        plt.xlabel("Importance")
        plt.title("Top Feature Importance")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()