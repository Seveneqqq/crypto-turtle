import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


class XaiFidelity:
    @staticmethod
    def _safe_feature_mapping(raw_feature_names, transformed_feature_names):
        """
        Mapowanie:
        raw feature -> lista feature po preprocessingu
        np. 'sex' -> ['cat__sex_F', 'cat__sex_M'] albo 'num__age'
        """
        mapping = {}

        for raw_col in raw_feature_names:
            matched = []
            for tf in transformed_feature_names:
                tf_clean = tf.lower()
                raw_clean = raw_col.lower()

                if raw_clean in tf_clean:
                    matched.append(tf)

            mapping[raw_col] = matched

        return mapping

    @staticmethod
    def feature_removal_test(
        pipeline,
        X_test,
        y_test,
        importance,
        task="classification",
        k=5
    ):
        """
        Test fidelity:
        1. wybiera top-k cech z explanation
        2. zeruje odpowiadające im cechy surowe
        3. sprawdza spadek jakości predykcji

        Zwraca:
        - baseline_score
        - perturbed_score
        - fidelity_drop
        """
        top_features = list(importance.keys())[:k]

        X_modified = X_test.copy()

        # pracujemy na surowych kolumnach X_test
        raw_columns = list(X_modified.columns)

        for feature in top_features:
            matched_raw_cols = [
                col for col in raw_columns
                if col.lower() in feature.lower() or feature.lower() in col.lower()
            ]

            for col in matched_raw_cols:
                if pd.api.types.is_numeric_dtype(X_modified[col]):
                    X_modified[col] = 0
                else:
                    if not X_modified[col].mode().empty:
                        X_modified[col] = X_modified[col].mode()[0]
                    else:
                        X_modified[col] = "unknown"

        y_pred_base = pipeline.predict(X_test)
        y_pred_perturbed = pipeline.predict(X_modified)

        if task == "classification":
            baseline_score = f1_score(y_test, y_pred_base, average="macro")
            perturbed_score = f1_score(y_test, y_pred_perturbed, average="macro")
            fidelity_drop = baseline_score - perturbed_score
        else:
            baseline_score = np.sqrt(mean_squared_error(y_test, y_pred_base))
            perturbed_score = np.sqrt(mean_squared_error(y_test, y_pred_perturbed))
            fidelity_drop = perturbed_score - baseline_score

        return {
            "baseline_score": float(baseline_score),
            "perturbed_score": float(perturbed_score),
            "fidelity_drop": float(fidelity_drop),
            "top_k_used": top_features
        }