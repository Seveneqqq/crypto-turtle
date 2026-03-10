import numpy as np
from utils.evaluation.xai_comparator import XaiComparator

class XaiStability:

    @staticmethod
    def add_noise(X, noise_level=0.01):

        X_noisy = X.copy()

        numeric_cols = X_noisy.select_dtypes(include=["float", "int"]).columns

        for col in numeric_cols:

            noise = np.random.normal(0, noise_level, len(X_noisy))

            X_noisy[col] = X_noisy[col] + noise

        return X_noisy


    @staticmethod
    def stability_score(importance1, importance2):

        return XaiComparator.spearman(importance1, importance2)

