from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd


def build_scaler(X: pd.DataFrame, scaling_type="standard"):
    """
    Automatyczne wykrywanie kolumn numerycznych do skalowania.
    Nie skaluje kolumn binarnych (0/1).
    """

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # wykryj kolumny binarne (0/1)
    binary_cols = [
        col for col in numeric_cols
        if set(X[col].dropna().unique()).issubset({0, 1})
    ]

    # kolumny do skalowania = numeryczne bez binarnych
    scale_cols = [col for col in numeric_cols if col not in binary_cols]

    if scaling_type == "standard":
        scaler = StandardScaler()
    elif scaling_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Scaling must be 'standard' or 'minmax'")

    column_transformer = ColumnTransformer(
        transformers=[
            ("scaler", scaler, scale_cols)
        ],
        remainder="passthrough"
    )

    return column_transformer