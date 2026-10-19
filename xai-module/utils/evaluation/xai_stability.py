from utils.evaluation.paired_evaluation import Perturbation, correlation

class XaiStability:
    @staticmethod
    def add_noise(X, X_train=None, noise_level=.01, random_state=42,
                  discrete_probability=.01, feature_types=None):
        if X_train is None:
            raise ValueError("Training data are required to calibrate perturbations")
        return Perturbation(X_train, noise_level, discrete_probability, feature_types).apply(X, random_state)[0]

    @staticmethod
    def stability_score(importance1, importance2):
        return correlation(importance1, importance2)
