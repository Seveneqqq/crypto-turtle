from sklearn.ensemble import RandomForestClassifier

def get_classifier():
    return RandomForestClassifier(
        n_estimators=891,
        max_depth=5,
        max_features=None,
        min_samples_leaf=4,
        min_samples_split=7,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
