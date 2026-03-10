import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class XaiHeatmap:
    @staticmethod
    def plot(comparisons, save_path):
        methods = set()

        for key in comparisons.keys():
            m1, m2 = key.split("_vs_")
            methods.add(m1)
            methods.add(m2)

        methods = sorted(list(methods))
        matrix = pd.DataFrame(
            np.eye(len(methods)),
            index=methods,
            columns=methods
        )

        for key, value in comparisons.items():
            m1, m2 = key.split("_vs_")
            score = value["spearman"]

            matrix.loc[m1, m2] = score
            matrix.loc[m2, m1] = score

        plt.figure(figsize=(7, 6))

        sns.heatmap(
            matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        plt.title("Agreement Between XAI Methods")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()