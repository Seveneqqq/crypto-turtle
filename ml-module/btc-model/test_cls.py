# test_cls.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# === 1. Wczytanie danych i modelu ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "shared", "files", "btc_test_data", "btc_test.csv")
df = pd.read_csv(CSV_PATH)

CLS_PATH = os.path.join(BASE_DIR, "shared", "models", "ml", "btc_model_cls.pkl")
cls_model = joblib.load(CLS_PATH)

# === 2. Feature engineering ===
df["return"] = df["close"].pct_change()
df["ma_5"] = df["close"].rolling(window=3).mean()
df["ma_15"] = df["close"].rolling(window=5).mean()
df["target_cls"] = pd.cut(df["close"].shift(-1) - df["close"],
                          bins=[-np.inf, -5, 5, np.inf],
                          labels=["sell", "hold", "buy"])
df.dropna(inplace=True)

features = ["close", "return", "ma_5", "ma_15"]
X_test = df[features]
y_test = df["target_cls"]

# === 3. Testowanie ===
y_pred = cls_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=["sell", "hold", "buy"])

print(f"📊 Dokładność klasyfikatora: {acc:.4f}")
print("Macierz pomyłek (kup / trzymaj / sprzedaj):")
print(cm)
