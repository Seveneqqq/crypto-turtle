import pandas as pd
import numpy as np

def create_target(df):
    # Logarytmowane zwroty
    df["log_return"] = np.log(df["close"].shift(-1)) - np.log(df["close"])

    # Tworzymy target klasyfikacyjny na podstawie log_return
    df["target_cls"] = pd.cut(
        df["log_return"],
        bins=[-float("inf"), -0.00005, 0.00005, float("inf")],
        labels=["sell", "hold", "buy"]
    )

    return df
