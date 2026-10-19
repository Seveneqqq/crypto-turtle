import pandas as pd
from scipy.stats import spearmanr


class XaiComparator:

    @staticmethod
    def rank_features(importance_dict):

        sorted_items = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [k for k, v in sorted_items]

    @staticmethod
    def spearman(imp1, imp2):

        from utils.evaluation.paired_evaluation import correlation
        return correlation(imp1, imp2)

    @staticmethod
    def top_k_overlap(imp1, imp2, k=10):

        k = min(k, len(imp1), len(imp2))
        if k < 1:
            return float("nan")
        top1 = set(
            sorted(imp1, key=imp1.get, reverse=True)[:k]
        )

        top2 = set(
            sorted(imp2, key=imp2.get, reverse=True)[:k]
        )

        return len(top1 & top2) / k