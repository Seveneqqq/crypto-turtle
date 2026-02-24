import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

#from .standalone_model import ModelPredictor
from .model_package import ModelPredictor

class ModelSignal:
    def __init__(self, model_name, data, base_dir, realtime = False):
        # base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "models")

        self.data = data
        self.close_prices = []
        self.open_times   = []
        self.volatility = []

        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.predicted_prices = []



        self.model_name = model_name
        self.model_path : str = ""
        self.model_type = ""
        print(f"[Core_Logic] Obliczanie predykcji.")



        match self.check_model_type():
            case "sklearn":
                self.model = joblib.load(self.model_path)

                if realtime:
                    self.features = self.model.feature_names_in_
                    self.predictions = []
                    self.class_map = ["buy", "hold", "sell"]
                else:
                    self.predictions, self.class_map = self.sklearn()

            case "lstm":
                self.model = ModelPredictor(self.model_path)
                self.predictions, self.class_map = self.lstm_price_predictor()

            case _:
                print("[ModelSignal][ERROR] Brak obsługi tego typu modelu.")




    def process_models_data(self, num_ticks):
        print(f"[Core_Logic] Ekstrakcja danych.")
        self.data = self.data.iloc[-num_ticks:].reset_index(drop=True)
        self.open_times = self.data["open_time"].to_numpy()
        self.close_prices = self.data["close"].to_numpy()

        #TODO: range dzielenia per symbol
        self.close = self.data["close"].to_numpy()
        self.high = self.data["high"].to_numpy()
        self.low = self.data["low"].to_numpy()
        self.open = self.data["open"].to_numpy()



        previous_close = self.data['close'].shift(1)
        true_range_pct = np.maximum(
            (self.data['high'] - self.data['low']) / self.data['close'],
            np.abs(self.data['high'] - previous_close) / previous_close
        )

        # Krok 2: Średnia z ostatnich 20 minut
        self.data['volatility'] = true_range_pct.rolling(window=20).mean()

        # Krok 3: Wypełnienie NaN (bo rolling robi dziury na początku)
        self.data['volatility'].fillna(0.0, inplace=True)
        self.volatility = self.data["volatility"].to_numpy()

    def generate_prediction(self, data):
        match self.model_type:
            case "sklearn":
                self.realtime_sklearn(data)
            case _:
                print("[ModelSignal][ERROR] Brak obsługi tego typu modelu.")


    def sklearn(self):
        features = self.model.feature_names_in_
        x = self.data[features].values
        probs = self.model.predict_proba(x)
        classes = [str(c) for c in self.model.classes_]
        class_map = {label: idx for idx, label in enumerate(classes)}
        return probs, class_map

    def realtime_sklearn(self,df):

        self.data.append(df)
        features = self.model.feature_names_in_

        df_row = pd.DataFrame([df])

        x = df_row[features].values

        prob = self.model.predict_proba(x)

        self.predictions.append(prob[0])
        self.open_times.append(df["open_time"])
        self.close_prices.append(df["close"])

        self.close.append(df["close"])
        self.high.append(df["high"])
        self.low.append(df["low"])
        self.open.append(df["open"])


    def lstm_price_predictor(self):
        preds = self.model.predict_batch(
            self.data,
            batch_size=256,
            include_signals=True # ← Wyłącz sygnały techniczne!
        )

        # Prosty fallback - oblicz z price_change
        signals = []
        prices = []
        for p in preds:
            buy = p["signals"]["buy_signal"]
            hold = 0.0
            sell = p["signals"]["sell_signal"]
            signals.append([buy, hold, sell])
            # log_ret = np.log(predicted / current)
            # conf = min(abs(log_ret) * 50, 1.0)
            # if log_ret > 0:
            #     signals.append([conf, 1 - conf, 0.0])
            # elif log_ret < 0:
            #     signals.append([0.0, 1 - conf, conf])
            # else:
            #     signals.append([0.0, 1.0, 0.0])
            prices.append(p["predicted_price"])

        classes = ["buy", "hold", "sell"]
        self.predicted_prices = prices
        return np.array(signals), classes




    def check_model_type(self):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.model_name)
            if os.path.exists(candidate_path):
                self.model_path = candidate_path
                self.model_type = subfolder.lower()
                return subfolder.lower()  # 'sklearn' lub 'lstm'

        raise ValueError(f"Nie znaleziono modelu '{self.model_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")