from sklearn.ensemble import RandomForestClassifier


def get_classifier():
    return RandomForestClassifier(
        # class_weight="balanced",
        n_estimators=573,
        max_depth=10,
        max_features=None,
        min_samples_leaf=5,
        min_samples_split=2,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
