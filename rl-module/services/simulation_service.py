import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from services.rl.config import CONFIG
from services.rl.wallet import Wallet
from services.rl.decision import DecisionEngine



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Ścieżka do danych
DATA_DIR = os.path.join(BASE_DIR, "shared", "simulation","data_to_test")
# Ścieżka do modelu
MODEL_DIR = os.path.join(BASE_DIR, "shared", "models","ml")
# Ścieżki do logów
LOGS_DIR = os.path.join(BASE_DIR, "shared", "simulation", "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "shared", "simulation", "plots")

# Ładowanie Decision Engine - na razie obsługa jednego
model_path = os.path.join(MODEL_DIR, CONFIG["models"]["BTC"])
engine = DecisionEngine(model_path=model_path, threshold=CONFIG["threshold_buy"])

# Ładowanie Plików z danymi - pełna obsługa w całym kodzie
def load_data():
    data = {}
    for symbol, filename in CONFIG["data_files"].items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Brak pliku: {filepath}")
        df = pd.read_csv(filepath)
        # Zakładamy, że CSV ma kolumny 'timestamp' i 'close'
        data[symbol] = df[["timestamp", "close"]].copy()
    return data

def extract_features(df):
    """
    df: DataFrame z kolumną 'close', wiersze od najstarszego do obecnego
    Zwraca DataFrame z jedną linią zawierającą cechy do modelu.
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(window=3).mean()
    df["ma_15"] = df["close"].rolling(window=5).mean()
    df = df.dropna()  # usuwamy wiersze gdzie rolling() daje NaN
    if df.empty:
        return None
    features = ["close", "return", "ma_5", "ma_15"]
    return df[features].iloc[[-1]]  # ostatni wiersz jako DataFrame


def run_simulation():
    data = load_data()
    symbols = list(data.keys())
    wallet = Wallet(CONFIG["balance_start"], symbols)

    num_ticks = min(len(df) for df in data.values())
    portfolio_values = []
    timestamps = []
    logs = []

    for t in range(num_ticks):
        prices = {symbol: data[symbol].iloc[t]["close"] for symbol in symbols}
        timestamps.append(data[symbols[0]].iloc[t]["timestamp"])

        for symbol in symbols:
            history = data[symbol].iloc[:t + 1].copy()
            if len(history) < 15:  # potrzeba danych do średnich kroczących
                continue

                # wyciągamy cechy z aktualnego ticka
            features = extract_features(history)

                # decyzja od decision engine
            amount,action = engine.decide(features)

                # wykonanie transakcji
            action_end,amount_end =wallet.execute(action, symbol, prices[symbol], amount)
            #  "close": prices[symbol],
            # "action": action_end,
            if action_end !="hold":
                logs.append({
                    symbol: action_end,
                    "timestamp": history.iloc[-1]["timestamp"],
                    "usd_balance": f"{wallet.balance:.2f}",
                    "portfel": f"{wallet.total_value(prices):.2f}"
                })

        portfolio_values.append(wallet.total_value(prices))

    # Generowanie wykresu do PNG i zakodowanie w base64
    plt.figure(figsize=(12,6))
    plt.plot(timestamps, portfolio_values, label="Wartość portfela (USD)")
    plt.xlabel("Timestamp")
    plt.ylabel("Wartość portfela (USD)")
    plt.title("Symulacja wartości portfela")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    LOGS_PATH = os.path.join(LOGS_DIR, f"{timestamp_str}_simulation_logs.txt")
    PLOT_PATH = os.path.join(PLOTS_DIR, f"{timestamp_str}_simulation_plot.png")

    plt.savefig(PLOT_PATH, format="png")
    plt.close()

    # --- zapis logów ---
    with open(LOGS_PATH, "w", encoding="utf-8") as f:
        for line in logs:
            f.write(json.dumps(line) + "\n")


