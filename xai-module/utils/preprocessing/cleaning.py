import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # usuń duplikaty
    df = df.drop_duplicates()

    # usuń kolumny całkowicie puste
    df = df.dropna(axis=1, how="all")

    # usuń wiersze całkowicie puste
    df = df.dropna(axis=0, how="all")

    # prosta imputacja numerycznych
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df