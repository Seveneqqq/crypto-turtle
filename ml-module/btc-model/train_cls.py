# train_cls.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib

# === 1. Wczytanie danych ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "shared", "files", "btc_test_data", "btc_train.csv")
df = pd.read_csv(CSV_PATH)

# === 2. Feature engineering ===
df["return"] = df["close"].pct_change()
df["ma_5"] = df["close"].rolling(window=3).mean()
df["ma_15"] = df["close"].rolling(window=5).mean()
df["target_cls"] = pd.cut(df["close"].shift(-1) - df["close"],
                          bins=[-np.inf, -5, 5, np.inf],
                          labels=["sell", "hold", "buy"])
df.dropna(inplace=True)

features = ["close", "return", "ma_5", "ma_15"]
X = df[features]
y = df["target_cls"]

# === 3. Trenowanie klasyfikatora ===
cls_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cls_model.fit(X, y)

# === 4. Zapis modelu ===
date_str = datetime.now().strftime("%d%m")

MODEL_PATH = os.path.join(BASE_DIR, "shared", "models", "ml", "btc_model_cls.pkl")
joblib.dump(cls_model, MODEL_PATH)
print("✅ Model klasyfikatora zapisany jako btc_model_cls.pkl")
