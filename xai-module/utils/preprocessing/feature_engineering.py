import pandas as pd


def generic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    datetime_cols = []

    for col in df.columns:
        col_lower = col.lower()

        # tylko kolumny typu object (string) potencjalnie zawierające datę
        if df[col].dtype == "object" and (
            "date" in col_lower or "time" in col_lower
        ):
            converted = pd.to_datetime(df[col], errors="coerce")

            # jeżeli konwersja faktycznie coś dała (nie same NaT)
            if converted.notna().sum() > 0:
                df[col] = converted

                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_hour"] = df[col].dt.hour

                datetime_cols.append(col)

    # usuń tylko prawdziwe datetime64
    df = df.drop(columns=datetime_cols)

    return df