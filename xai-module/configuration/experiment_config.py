FINANCE_PATH = (
    "utils/data/raw/finance/"
    "BTCUSDT-model_1m_6m_2025-10-11_97f5b9c9-a08e-4a75-a0b6-efca2b5f3192.csv"
)

MEDICAL_PATH = (
    "utils/data/raw/medical/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)

CRIME_PATH = "utils/data/raw/law/crime.csv"


EXPERIMENTS = {
    # =========================
    # MEDICAL
    # =========================
    "medical_classification_random": {
        "dataset_name": "medical_classification_random",
        "data_path": MEDICAL_PATH,
        "target": "Diabetes_012",
        "task": "classification",
        "split_type": "random",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    "medical_classification_time": {
        "dataset_name": "medical_classification_time",
        "data_path": MEDICAL_PATH,
        "target": "Diabetes_012",
        "task": "classification",
        "split_type": "time",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    # =========================
    # CRIME
    # =========================
    "crime_classification_random": {
        "dataset_name": "crime_classification_random",
        "data_path": CRIME_PATH,
        "target": "OFFENSE_CODE_GROUP",
        "task": "classification",
        "split_type": "random",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    "crime_classification_time": {
        "dataset_name": "crime_classification_time",
        "data_path": CRIME_PATH,
        "target": "OFFENSE_CODE_GROUP",
        "task": "classification",
        "split_type": "time",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    # =========================
    # FINANCE — CLASSIFICATION
    # =========================
    "finance_classification_direction_h1_time": {
        "dataset_name": "finance_classification_direction_h1_time",
        "data_path": FINANCE_PATH,
        "target": "target_direction",
        "task": "classification",
        "split_type": "time",
        "finance_horizon": 1,
        "finance_target_type": "direction",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    "finance_classification_direction_h5_time": {
        "dataset_name": "finance_classification_direction_h5_time",
        "data_path": FINANCE_PATH,
        "target": "target_direction",
        "task": "classification",
        "split_type": "time",
        "finance_horizon": 5,
        "finance_target_type": "direction",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["rf", "logreg", "xgb", "dt"]
    },

    # =========================
    # FINANCE — REGRESSION
    # =========================
    "finance_regression_return_h1_time": {
        "dataset_name": "finance_regression_return_h1_time",
        "data_path": FINANCE_PATH,
        "target": "target",
        "task": "regression",
        "split_type": "time",
        "finance_horizon": 1,
        "finance_target_type": "return",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["linreg", "rf", "xgb"]
    },

    "finance_regression_return_h5_time": {
        "dataset_name": "finance_regression_return_h5_time",
        "data_path": FINANCE_PATH,
        "target": "target",
        "task": "regression",
        "split_type": "time",
        "finance_horizon": 5,
        "finance_target_type": "return",
        "xai_methods": ["shap", "permutation", "lime"],
        "models": ["linreg", "rf", "xgb"]
    }
}