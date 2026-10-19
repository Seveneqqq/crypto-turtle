import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


from utils.evaluation.feature_mapping import transformed_to_raw


class XaiFidelity:
    @staticmethod
    def feature_removal_test(
        pipeline,
        X_test,
        y_test,
        importance,
        task="classification",
        k=5,
        feature_space="transformed"
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
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not X_test.columns.is_unique:
            raise ValueError("Masking requires unique input column names")
        if feature_space == "raw":
            mapping = {col: col for col in X_test.columns}
        elif feature_space == "transformed":
            mapping = transformed_to_raw(pipeline.named_steps["preprocessing"])
        else:
            raise ValueError("feature_space must be raw or transformed")
        top_features = list(importance.keys())[:k]
        unknown = [feature for feature in top_features if feature not in mapping]
        if unknown:
            raise ValueError(f"Unmapped masking features: {unknown}")
        selected_mapping = {feature: mapping[feature] for feature in top_features}
        selected_columns = list(dict.fromkeys(selected_mapping.values()))
        X_modified = X_test.copy()
        modified_columns = []
        changed_rows = {}
        for col in selected_columns:
            if pd.api.types.is_numeric_dtype(X_modified[col]):
                X_modified[col] = 0
            else:
                mode = X_modified[col].mode()
                X_modified[col] = mode.iloc[0] if not mode.empty else "unknown"
            before, after = X_test[col], X_modified[col]
            equal = before.eq(after).fillna(False) | (before.isna() & after.isna())
            changed_rows[col] = int((~equal).sum())
            if changed_rows[col]:
                modified_columns.append(col)

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
            "top_k_used": top_features,
            "feature_space": feature_space,
            "feature_to_column": selected_mapping,
            "selected_columns": selected_columns,
            "modified_columns": modified_columns,
            "changed_rows_per_column": changed_rows
        }