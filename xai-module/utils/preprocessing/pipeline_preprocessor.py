import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


class GenericPreprocessor:

    def __init__(
        self,
        target: str,
        split_type: str = "random",
        test_size: float = 0.2,
        random_state: int = 42,
        scaling: bool = True,
        finance_horizon: int = 1,           
        finance_target_type: str = "return" 
    ):
        self.target = target
        self.split_type = split_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaling = scaling
        self.label_encoder = None
        self.finance_horizon = finance_horizon
        self.finance_target_type = finance_target_type

    # =============================
    # MAIN
    # =============================
    def prepare(self, df: pd.DataFrame):

        df = self._clean(df)

        df = self._feature_engineering(df)

        if self.target not in df.columns:
            raise KeyError(
                f"Target column '{self.target}' not found in dataframe columns: {list(df.columns)[:30]}..."
            )

        X = df.drop(columns=[self.target])
        y = df[self.target]

        X_train, X_test, y_train, y_test = self._split(X, y)

        if y_train.dtype == "object":
            self.label_encoder = LabelEncoder()
            y_train_enc = self.label_encoder.fit_transform(y_train)

            known = set(self.label_encoder.classes_)
            mask = y_test.isin(known)

            if (~mask).any():
                dropped = y_test[~mask].unique()
                print(f"⚠️ Dropping {len(y_test[~mask])} test rows with unseen classes: {dropped}")

            X_test = X_test.loc[mask]
            y_test = y_test.loc[mask]

            y_test_enc = self.label_encoder.transform(y_test)

            y_train = pd.Series(y_train_enc, index=X_train.index)
            y_test = pd.Series(y_test_enc, index=X_test.index)

        transformer = self._build_transformer(X_train)

        return X_train, X_test, y_train, y_test, transformer

    # =============================
    # CLEANING
    # =============================
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.drop_duplicates()
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        if self.target == "Diabetes_012" and self.target in df.columns:
            df[self.target] = df[self.target].apply(lambda x: 1 if x > 0 else 0)

        return df

    # =============================
    # FEATURE ENGINEERING
    # =============================
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if self.target == "OFFENSE_CODE_GROUP":
            cols_to_drop = [
                "OFFENSE_CODE",
                "OFFENSE_DESCRIPTION",
                "INCIDENT_NUMBER",
                "UCR_PART"
            ]

            df = df.drop(
                columns=[col for col in cols_to_drop if col in df.columns],
                errors="ignore"
            )

        if self.target in ["target", "target_direction"]:
        
            if "close" not in df.columns:
                raise KeyError(
                    "Finance mode: column 'close' not found. "
                    f"Available columns: {list(df.columns)}"
                )
        
            h = int(self.finance_horizon)
        
            if self.target == "target_direction" or self.finance_target_type == "direction":
                df["target_direction"] = (
                    df["close"].shift(-h) > df["close"]
                ).astype(int)
        
                df = df.dropna(subset=["target_direction"])
        
            elif self.finance_target_type == "future_close":
                df["target"] = df["close"].shift(-h)
                df = df.dropna(subset=["target"])
        
            else:
                df["target"] = np.log(df["close"].shift(-h) / df["close"])
                df = df.dropna(subset=["target"])

        datetime_cols = []
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_hour"] = df[col].dt.hour
                    datetime_cols.append(col)

        if len(datetime_cols) > 0:
            df = df.drop(columns=datetime_cols)

        return df

    # =============================
    # SPLIT
    # =============================
    def _split(self, X, y):

        if self.split_type == "random":
            return train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state
            )

        elif self.split_type == "time":
            split_index = int(len(X) * (1 - self.test_size))
            return (
                X.iloc[:split_index],
                X.iloc[split_index:],
                y.iloc[:split_index],
                y.iloc[split_index:]
            )

        else:
            raise ValueError("split_type must be 'random' or 'time'")

    # =============================
    # TRANSFORMER
    # =============================
    def _build_transformer(self, X: pd.DataFrame):

        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

        high_cardinality_cols = [
            col for col in categorical_cols
            if X[col].nunique() > 50
        ]

        categorical_cols = [
            col for col in categorical_cols
            if col not in high_cardinality_cols
        ]

        print("Dropped high-cardinality cols:", high_cardinality_cols)

        binary_cols = [
            col for col in numeric_cols
            if set(X[col].dropna().unique()).issubset({0, 1})
        ]

        continuous_cols = [
            col for col in numeric_cols
            if col not in binary_cols
        ]

        transformers = []

        if self.scaling and len(continuous_cols) > 0:
            transformers.append(("num", StandardScaler(), continuous_cols))

        if len(categorical_cols) > 0:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols
            ))

        if len(binary_cols) > 0:
            transformers.append(("bin", "passthrough", binary_cols))

        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return transformer