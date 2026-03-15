import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    df = df.dropna(axis=1, how="all")

    df = df.dropna(axis=0, how="all")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df