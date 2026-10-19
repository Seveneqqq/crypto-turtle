import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
        finance_target_type: str = "return",
        time_column=None 
    ):
        self.target = target
        self.split_type = split_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaling = scaling
        self.label_encoder = None
        self.finance_horizon = finance_horizon
        self.finance_target_type = finance_target_type
        self.time_column = time_column

    # =============================
    # MAIN
    # =============================
    def prepare(self, df: pd.DataFrame):

        self.chronology_ = "random_split" if self.split_type == "random" else "ordered_rows_without_time_axis"
        time_column = self.time_column
        if self.split_type == "time" and self.target == "OFFENSE_CODE_GROUP" and time_column is None:
            time_column = "OCCURRED_ON_DATE"
        if self.split_type == "time" and time_column:
            times = pd.to_datetime(df[time_column], errors="raise")
            if times.isna().any():
                raise ValueError("Missing timestamps in chronological split")
            df = df.assign(**{time_column:times}).sort_values(time_column, kind="stable")
            self.chronology_ = f"ascending:{time_column}"
        if self.target in ["target", "target_direction"]:
            self.chronology_ = "ascending:timestamp_or_datetime_index"
        df = self._clean(df)

        df = self._feature_engineering(df)

        if self.target not in df.columns:
            raise KeyError(
                f"Target column '{self.target}' not found in dataframe columns: {list(df.columns)[:30]}..."
            )

        df = df.dropna(subset=[self.target])
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
                print(f" Dropping {len(y_test[~mask])} test rows with unseen classes: {dropped}")

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

        df = df.copy()
        if self.target not in ["target", "target_direction"]:
            df = df.drop_duplicates().dropna(axis=0, how="all")
        if self.target == "Diabetes_012" and self.target in df.columns:
            df = df.dropna(subset=[self.target])
            df[self.target] = (df[self.target] > 0).astype(int)
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
            if h < 1 or h != self.finance_horizon:
                raise ValueError("finance_horizon must be a positive integer")
            if self.split_type != "time":
                raise ValueError("Financial targets require a chronological split")
            if "timestamp" in df.columns:
                times = pd.to_datetime(df["timestamp"], errors="raise")
                if times.isna().any() or times.duplicated().any():
                    raise ValueError("Finance timestamps must be present and unique")
                df = df.assign(timestamp=times).sort_values("timestamp", kind="stable")
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Finance data require timestamp or a DatetimeIndex")
            else:
                if df.index.hasnans or df.index.has_duplicates:
                    raise ValueError("Finance time index must be present and unique")
                df = df.sort_index()
            current = pd.to_numeric(df["close"], errors="raise")
            future = current.shift(-h)
            valid = np.isfinite(current) & np.isfinite(future)
            if self.target == "target_direction" or self.finance_target_type == "direction":
                labels = (future > current).astype(float).where(valid)
            elif self.finance_target_type == "future_close":
                labels = future.where(valid)
            elif self.finance_target_type == "return":
                valid &= (current > 0) & (future > 0)
                labels = np.log(future.where(valid) / current.where(valid))
            else:
                raise ValueError("Unknown finance_target_type")
            df[self.target] = labels
            df = df.loc[valid].copy()

        datetime_cols = []
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                if col == "close_time" and pd.api.types.is_numeric_dtype(df[col]):
                    converted = pd.to_datetime(df[col], unit="ms", errors="coerce")
                else:
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
            gap = int(self.finance_horizon) if self.target in ["target", "target_direction"] else 0
            train_end = split_index - gap
            if train_end < 1 or split_index >= len(X):
                raise ValueError("Not enough observations for train/test split and horizon gap")
            self.split_metadata_ = {"gap_rows": gap, "train_rows": train_end,
                                    "test_rows": len(X) - split_index}
            return (
                X.iloc[:train_end],
                X.iloc[split_index:],
                y.iloc[:train_end],
                y.iloc[split_index:]
            )

        else:
            raise ValueError("split_type must be 'random' or 'time'")

    # =============================
    # TRANSFORMER
    # =============================
    def _build_transformer(self, X: pd.DataFrame):

        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
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

        if continuous_cols:
            steps = [("imputation", SimpleImputer(strategy="median", keep_empty_features=True))]
            if self.scaling:
                steps.append(("scaling", StandardScaler()))
            transformers.append(("num", Pipeline(steps), continuous_cols))

        if len(categorical_cols) > 0:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_cols
            ))

        if len(binary_cols) > 0:
            transformers.append(("bin", SimpleImputer(strategy="median", keep_empty_features=True), binary_cols))

        transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        return transformer