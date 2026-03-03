EXPERIMENTS = {
    "medical": {
        "data_path": "utils/data/raw/medical/diabetes_012_health_indicators_BRFSS2015.csv",
        "target": "Diabetes_012",
        "task": "classification",
        "split_type": "random"
    },
    "crime": {
        "data_path": "utils/data/raw/law/crime.csv",
        "target": "OFFENSE_CODE_GROUP",
        "task": "classification",
        "split_type": "time"
    },
    "finance": {
        "data_path": "utils/data/raw/finance/BTCUSDT-model_1m_6m_2025-10-11_97f5b9c9-a08e-4a75-a0b6-efca2b5f3192.csv",
        "target": "target",
        "task": "regression",
        "split_type": "time"
    }
}