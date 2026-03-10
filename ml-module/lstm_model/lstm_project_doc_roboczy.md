# LSTM Price Prediction - Dokumentacja Projektowa

#### W dokumentacji mogą występować błędy

## 1. Sposób użycia

### 1.1 Użycie przez RabbitMQ (API)

#### 1.1.1 LSTM Prepare Data

**Minimalne wywołanie:**
```json
POST /ml/lstm_prepare_data
{
  "data_path": "btc_1h_data.csv"
}
```

**Maksymalne wywołanie:**
```json
POST /ml/lstm_prepare_data
{
  "data_path": "btc_1h_data.csv",
  "feature_set": "technical",
  "train_months": 3,
  "agent_months": 1,
  "test_months": 1,
  "train_pct": 70.0,
  "agent_pct": 10.0,
  "test_pct": 20.0,
  "lookback": 48,
  "horizon": 1,
  "scaler_features": "standard",
  "scaler_price": "standard",
  "use_log_prices": true
}
```

**Parametry:**
- `data_path` (required) - Ścieżka do pliku CSV z danymi OHLCV
- `feature_set` (default: "full") - Zestaw cech: "minimal", "technical", "full", "returns_only", "no_time", "momentum", "volatility_focused"
- `train_months` (optional) - Liczba miesięcy dla danych treningowych (tryb "months")
- `agent_months` (optional) - Liczba miesięcy dla danych agenta/walidacji
- `test_months` (optional) - Liczba miesięcy dla danych testowych (tryb "months")
- `train_pct` (default: 70.0) - Procent danych treningowych (tryb "percentage")
- `agent_pct` (default: 0.0) - Procent danych agenta
- `test_pct` (default: 30.0) - Procent danych testowych (tryb "percentage")
- `lookback` (default: 48) - Liczba timestepów historycznych w sekwencji
- `horizon` (default: 1) - Horyzont predykcji (t+1, t+3, etc.)
- `scaler_features` (default: "standard") - Typ skalera dla cech: "standard", "minmax", "robust"
- `scaler_price` (default: "standard") - Typ skalera dla cen
- `use_log_prices` (default: true) - Czy używać logarytmu naturalnego dla cen

---

#### 1.1.2 LSTM Tune

**Minimalne wywołanie:**
```json
POST /ml/lstm_tune
{
  "train_path": "B001_prep_btc_data_full_1m"
}
```

**Maksymalne wywołanie:**
```json
POST /ml/lstm_tune
{
  "train_path": "B001_prep_btc_data_full_1m",
  "n_trials": 100,
  "timeout": 3600,
  "study_name": "btc_tuning_experiment_01",
  "subset_size": 200000,
  "feature_set": "technical",
  "custom_features": ["return_1", "return_5", "rsi", "macd"],
  "validate_features": true,
  "generate_features": true,
  "fixed_horizon": 1
}
```

**Parametry:**
- `train_path` (required) - Ścieżka do przygotowanych danych lub CSV
- `n_trials` (default: 100) - Liczba prób optymalizacji Optuna
- `timeout` (optional) - Maksymalny czas w sekundach (None = bez limitu)
- `study_name` (optional) - Nazwa studium (auto-generated jeśli None)
- `subset_size` (default: 200000) - Rozmiar podzbioru danych przy `use_subset=True`
- `feature_set` (optional) - Zestaw cech do użycia
- `custom_features` (optional) - Lista custom cech (nadpisuje feature_set)
- `validate_features` (default: true) - Czy walidować dostępność cech
- `generate_features` (default: true) - Czy generować cechy z raw data
- `fixed_horizon` (optional) - Stały horyzont (jeśli None, horyzont jest tunowany)

---

#### 1.1.3 LSTM Train

**Minimalne wywołanie:**
```json
POST /ml/lstm_train
{
  "train_path": "B001_prep_btc_data_full_1m"
}
```

**Maksymalne wywołanie:**
```json
POST /ml/lstm_train
{
  "train_path": "B001_prep_btc_data_full_1m",
  "lookback": 48,
  "horizon": 1,
  "hidden_size": 128,
  "num_layers": 2,
  "dropout": 0.2,
  "epochs": 100,
  "batch_size": 256,
  "lr": 0.0005,
  "weight_decay": 1e-5,
  "use_subset": false,
  "subset_size": 100000,
  "feature_set": "technical",
  "custom_features": null,
  "validate_features": true,
  "generate_features": true,
  "scheduler_type": "cosine",
  "use_log_prices": true,
  "early_stopping_patience": 25,
  "load_optuna_params": "O001_optuna_tuning_results_B001_0312/best_params.json",
  "load_optuna_dir": "O001_optuna_tuning_results_B001_0312",
  "train_all_top": false,
  "seed": 42
}
```

**Parametry:**
- `train_path` (required) - Ścieżka do przygotowanych danych lub CSV
- `lookback` (default: 48) - Liczba timestepów w sekwencji wejściowej
- `horizon` (default: 1) - Horyzont predykcji (t+horizon)
- `hidden_size` (default: 128) - Rozmiar warstwy ukrytej LSTM
- `num_layers` (default: 1) - Liczba warstw LSTM (1-3)
- `dropout` (default: 0.2) - Dropout rate (0.0-0.5)
- `epochs` (default: 100) - Maksymalna liczba epok
- `batch_size` (default: 256) - Rozmiar batcha
- `lr` (default: 0.0005) - Learning rate
- `weight_decay` (default: 1e-5) - Weight decay dla AdamW
- `use_subset` (default: false) - Czy używać podzbioru danych
- `subset_size` (default: 100000) - Rozmiar podzbioru
- `feature_set` (optional) - Zestaw cech (gdy train_path to CSV)
- `custom_features` (optional) - Custom lista cech
- `validate_features` (default: true) - Walidacja cech
- `generate_features` (default: true) - Generowanie cech z raw data
- `scheduler_type` (default: "none") - Scheduler: "none", "cosine", "plateau"
- `use_log_prices` (default: true) - Logarytm cen
- `early_stopping_patience` (default: 25) - Cierpliwość early stopping
- `load_optuna_params` (optional) - Ścieżka do best_params.json z tuningu
- `load_optuna_dir` (optional) - Katalog z wynikami tuningu
- `train_all_top` (default: false) - Czy trenować top 3 konfiguracje z tuningu
- `seed` (default: 42) - Random seed dla reprodukowalności

---

#### 1.1.4 LSTM Test

**Minimalne wywołanie:**
```json
POST /ml/lstm_test
{
  "model_dir": "L001_btc_model_B001_0312"
}
```

**Maksymalne wywołanie:**
```json
POST /ml/lstm_test
{
  "model_dir": "L001_btc_model_B001_0312",
  "test_path": "B001_prep_btc_data_full_1m",
  "batch_size": 1024
}
```

**Parametry:**
- `model_dir` (optional) - Nazwa katalogu modelu (auto-detect latest jeśli None)
- `test_path` (optional) - Ścieżka do danych testowych (auto-detect z modelu jeśli None)
- `batch_size` (default: 1024) - Rozmiar batcha dla inferencji

---

### 1.2 Użycie przez konsolę

#### 1.2.1 Prepare Data

**Minimalne wywołanie:**
```bash
cd ml-module/lstm_model
python prepare_data.py ../../shared/files/raw_data/btc_1h_data.csv
```

**Maksymalne wywołanie:**
```bash
python prepare_data.py ../../shared/files/raw_data/btc_1h_data.csv \
  --output_dir ../../shared/files/lstm_prepared_data \
  --feature_set technical \
  --train_months 3 \
  --agent_months 1 \
  --test_months 1 \
  --lookback 48 \
  --horizon 1 \
  --scaler_features standard \
  --scaler_price standard \
  --no_log_prices
```

**Parametry:**
- `data_path` (positional, required) - Ścieżka do pliku CSV
- `--output_dir` (default: lstm_prepared_data) - Katalog wyjściowy
- `--feature_set` (default: full) - Zestaw cech
- `--train_months` (default: 3) - Miesiące treningowe
- `--agent_months` (default: 0) - Miesiące agenta
- `--test_months` (default: 1) - Miesiące testowe
- `--lookback` (default: 48) - Lookback window
- `--horizon` (default: 1) - Horyzont predykcji
- `--scaler_features` (default: standard) - Skaler cech
- `--scaler_price` (default: standard) - Skaler cen
- `--no_log_prices` (flag) - Wyłącza logarytm cen

---

#### 1.2.2 Hyperparameter Tuning

**Minimalne wywołanie:**
```bash
python lstm_hyperparameter_tuning.py
```

**Maksymalne wywołanie:**
```bash
python lstm_hyperparameter_tuning.py \
  --n_trials 100 \
  --timeout 3600 \
  --study_name my_btc_study \
  --subset_size 200000 \
  --feature_set technical \
  --custom_features return_1 return_5 rsi macd \
  --no_validate_features \
  --no_generate_features \
  --train_path ../../shared/files/lstm_prepared_data/B001_prep_btc_data_full_1m \
  --fixed_horizon 1
```

**Parametry:**
- `--n_trials` (default: z TuningConfig) - Liczba prób Optuna
- `--timeout` (default: None) - Timeout w sekundach
- `--study_name` (optional) - Nazwa studium
- `--subset_size` (default: z TuningConfig) - Rozmiar podzbioru
- `--feature_set` (optional) - Zestaw cech
- `--custom_features` (optional) - Lista custom cech
- `--no_validate_features` (flag) - Wyłącza walidację cech
- `--no_generate_features` (flag) - Wyłącza generowanie cech
- `--train_path` (default: z TuningConfig) - Ścieżka do danych
- `--fixed_horizon` (optional) - Stały horyzont

---

#### 1.2.3 Train Model

**Minimalne wywołanie:**
```bash
python lstm_train_price.py
```

**Maksymalne wywołanie:**
```bash
python lstm_train_price.py \
  --train_path ../../shared/files/lstm_prepared_data/B001_prep_btc_data_full_1m \
  --lookback 48 \
  --horizon 1 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout 0.2 \
  --epochs 100 \
  --batch_size 256 \
  --lr 0.0005 \
  --weight_decay 1e-5 \
  --use_subset \
  --subset_size 100000 \
  --feature_set technical \
  --custom_features return_1 return_5 rsi \
  --no_validate_features \
  --no_generate_features \
  --scheduler_type cosine \
  --use_log_prices \
  --early_stopping_patience 25 \
  --load_optuna_params ../../shared/models/lstm_data/O001_optuna_tuning/best_params.json \
  --load_optuna_dir ../../shared/models/lstm_data/O001_optuna_tuning \
  --train_all_top \
  --seed 42
```

**Parametry:** (analogiczne do API, użyj `--` przed nazwą parametru)

---

#### 1.2.4 Test Model

**Minimalne wywołanie:**
```bash
python lstm_test_price.py
```

**Maksymalne wywołanie:**
```bash
python lstm_test_price.py \
  --model_dir ../../shared/models/lstm_models/L001_btc_model_B001_0312 \
  --test_path ../../shared/files/lstm_prepared_data/B001_prep_btc_data_full_1m \
  --batch_size 1024
```

**Parametry:**
- `--model_dir` (optional) - Ścieżka do katalogu modelu
- `--test_path` (optional) - Ścieżka do danych testowych
- `--batch_size` (default: 1024) - Rozmiar batcha

---

## 2. Opis kodu i struktury

### 2.1 Pliki główne (wywoływane)

#### **prepare_data.py**
Przygotowuje surowe dane OHLCV do treningu LSTM.

**Działanie:**
1. Wczytuje CSV z danymi OHLCV (open, high, low, close, volume, open_time)
2. Generuje cechy techniczne przez `feature_engineering.py`
3. Czyści dane (usuwa NaN, inf)
4. Dzieli dane na train/agent/test według miesięcy LUB procentów
5. Skaluje cechy i ceny (opcjonalnie log-transform)
6. Tworzy sekwencje 3D dla LSTM: `[samples, lookback, features]`
7. Zapisuje przygotowane dane, scalery i metadata

**Zastosowanie:**
- Pierwszy krok pipeline'u
- Automatyczna detekcja symbolu (BTC/ETH) i interwału
- Generuje dane 3D (dla treningu) i 2D (dla agenta)
- Tworzy unikalny data_id (np. B001, E001)

---

#### **lstm_hyperparameter_tuning.py**
Optymalizacja hiperparametrów używając Optuna.

**Działanie:**
1. Definiuje search space dla hiperparametrów (hidden_size, layers, dropout, lr, etc.)
2. Dla każdego trialu:
   - Sugeruje wartości hiperparametrów
   - Trenuje mały model (max 40 epok)
   - Ewaluuje na walidacji
   - Opcjonalnie prune'uje słabe trialy (MedianPruner)
3. Zapisuje top 3 konfiguracje (best, second_best, third_best)
4. Generuje wizualizacje (history, importances, distribution)

**Zastosowanie:**
- Znajdowanie optymalnych parametrów przed pełnym treningiem
- Time Series Split dla walidacji (zapobiega data leakage)
- Zapisuje best_params.json do użycia w treningu
- Może tunować zarówno architekturę jak i lookback/horizon

---

#### **lstm_train_price.py**
Trenuje model LSTM do predykcji ceny.

**Działanie:**
1. Ładuje dane (prepared NPY lub raw CSV)
2. Tworzy model LSTM: LSTM → FC → BatchNorm → ReLU → Dropout → FC
3. Trenuje z:
   - AdamW optimizer (weight decay)
   - MSE loss
   - Gradient clipping (max_norm=1.0)
   - Early stopping (patience domyślnie 25)
   - Opcjonalny scheduler (cosine, plateau)
4. Zapisuje najlepszy model, scalery, config i training curves

**Zastosowanie:**
- Główny trening modelu
- Może załadować parametry z Optuna
- Opcja `train_all_top` trenuje 3 modele z tuningu
- Generuje model_id (L001, L002, etc.)

---

#### **lstm_test_price.py**
Testuje wytrenowany model na danych testowych.

**Działanie:**
1. Ładuje model i jego konfigurację
2. Auto-detect danych testowych z metadata modelu
3. Generuje predykcje w batchach
4. Oblicza metryki:
   - Price metrics: MAE, RMSE, MAPE, R²
   - Directional accuracy (czy przewidział wzrost/spadek)
   - Error statistics
5. Tworzy wizualizacje:
   - Time series (actual vs predicted)
   - Scatter plot
   - Error distribution
   - Directional confusion matrix

**Zastosowanie:**
- Ewaluacja wytrenowanego modelu
- Auto-detection danych testowych z train metadata
- Comprehensive metrics dla finansów
- Generuje test_id (T001, T002, etc.)

---

### 2.2 Feature Engineering (`feature_engineering.py`)

**Główna funkcja: `prepare_features(df)`**

Generuje cechy techniczne z danych OHLCV:

**1. Returns (Log-returns):**
- `return_1`, `return_5`, `return_15`, `return_60` - momentum na różnych skalach czasowych
- Formuła: `log(close_t / close_{t-n})`
- Zastosowanie: Najważniejszy sygnał dla predykcji cen

**2. Moving Averages (SMA ratios):**
- `close_sma_7`, `close_sma_14`, `close_sma_30` - pozycja ceny względem średnich
- Formuła: `close / SMA(close, window)`
- Zastosowanie: Wykrywanie trendów, poziomy wsparcia/oporu

**3. MACD (Moving Average Convergence Divergence):**
- `macd` = EMA(12) - EMA(26)
- `macd_signal` = EMA(macd, 9)
- `macd_diff` = macd - macd_signal
- Zastosowanie: Momentum i punkty zwrotne

**4. RSI (Relative Strength Index):**
- `rsi` - zakres 0-100
- Formuła: 100 - (100 / (1 + RS)), gdzie RS = avg_gain / avg_loss
- Zastosowanie: Wykrywanie wykupienia/wyprzedania

**5. Volatility:**
- `volatility_7`, `volatility_14`, `volatility_30` - rolling std of returns
- Zastosowanie: Ryzyko i zmienność rynku

**6. Volume:**
- `relative_volume` = volume / SMA(volume, 20)
- Zastosowanie: Siła ruchu cenowego

**7. Bollinger Bands:**
- `bb_position` = (close - lower_band) / (upper_band - lower_band)
- Zastosowanie: Pozycja ceny w paśmie, wykrywanie ekstremów

**8. Price Range:**
- `hl_range` = (high - low) / close
- Zastosowanie: Intraday volatility

**9. Time Features (Cyclical):**
- `hour_sin`, `hour_cos` - godzina dnia (0-23)
- `dow_sin`, `dow_cos` - dzień tygodnia (0-6)
- Zastosowanie: Sezonalność, wzorce czasowe w krypto

**Predefiniowane zestawy cech:**

- **minimal** (6 cech): Podstawowe - returns, SMA, RSI, volatility, volume
- **technical** (14 cech): Bez czasu - wszystkie techniczne + BB + range
- **full** (28 cech): Wszystkie cechy włącznie z czasem
- **returns_only** (4 cechy): Tylko returns na różnych skalach
- **no_time** (24 cechy): Wszystkie bez cech czasowych
- **momentum** (13 cech): Focus na returns, MACD, RSI, SMA
- **volatility_focused** (8 cech): Focus na volatility, BB, range, volume

---

### 2.3 Pliki pomocnicze (common/)

#### **data_utils.py**
- `load_data()` - ładuje prepared data lub CSV
- `create_sequences()` - tworzy 3D sequences dla LSTM
- `prepare_raw_data()` - pipeline przygotowania raw CSV

#### **id_utils.py**
- `get_next_model_id()` - generuje L001, L002, etc.
- `get_next_optuna_id()` - generuje O001, O002, etc.
- `get_next_test_id()` - generuje T001, T002, etc.
- `extract_data_id()` - wyciąga data_id z metadata

#### **metrics.py**
- `calculate_metrics()` - MAE, RMSE, MAPE, R², error stats
- `calculate_directional_accuracy()` - accuracy kierunku ruchu
- `print_metrics_summary()` - formatowane wydruki
- `interpret_metrics()` - interpretacja wyników

#### **visualization.py**
- `plot_training_curves()` - train/val loss
- `plot_test_results()` - comprehensive test visualizations
- `plot_optuna_results()` - Optuna optimization plots

---

### 2.4 Serwis (`lstm_service.py`)

**Klasa LSTMService** - wrapper integrujący LSTM z aplikacją.

**Główne metody:**
- `train_model()` - wrapper dla `lstm_train_price.train_model()`
- `test_model()` - wrapper dla `lstm_test_price.test_model()`
- `prepare_data()` - wrapper dla `prepare_data.prepare_data()`
- `tune_hyperparameters()` - wrapper dla `lstm_hyperparameter_tuning.run_tuning()`

**Funkcjonalność:**
- Automatyczne rozwiązywanie ścieżek (`_resolve_data_path`)
- Przełączanie working directory
- Dynamiczne importy modułów LSTM
- Konwersja wyników do formatu RabbitMQ
- Error handling i logging

---

## 3. Zapisywane pliki

### 3.1 Prepare Data
**Lokalizacja:** `shared/files/lstm_prepared_data/[DATA_ID]_prep_[symbol]_data_[feature_set]_[interval]/`

**Pliki:**
- `X_train.npy` - Sekwencje treningowe 3D `[samples, lookback, features]`
- `y_train.npy` - Target treningowy `[samples]`
- `X_test_3d.npy` - Sekwencje testowe 3D
- `y_test_3d.npy` - Target testowy
- `test_2d.npy` - Dane testowe 2D `[samples, features+1]` (bez sekwencji)
- `agent_data.npy` - Dane agenta 2D (jeśli agent_months > 0)
- `scaler_features.pkl` - Fitted scaler dla cech
- `scaler_price.pkl` - Fitted scaler dla cen
- `metadata.json` - Kompletna metadata (data_id, symbol, features, split info, etc.)
- `preparation.log` - Log przygotowania

**Przykład nazwy:** `B001_prep_btc_data_full_1m/`

---

### 3.2 Hyperparameter Tuning
**Lokalizacja:** `shared/models/lstm_data/[OPTUNA_ID]_optuna_tuning_results_[DATA_ID]_[date]/`

**Pliki:**
- `best_params.json` - Najlepsze parametry (rank 1)
- `second_best_params.json` - Drugie najlepsze (rank 2)
- `third_best_params.json` - Trzecie najlepsze (rank 3)
- `study.pkl` - Cały Optuna study object
- `trials.csv` - DataFrame wszystkich triali
- `tuning.log` - Log optymalizacji
- `summary.txt` - Podsumowanie wyników
- `history.png` - Optimization history plot
- `importances.png` - Parameter importances plot
- `distribution.png` - Trial results distribution

**Przykład nazwy:** `O001_optuna_tuning_results_B001_0312/`

---

### 3.3 Train Model
**Lokalizacja:** `shared/models/lstm_models/[MODEL_ID]_[symbol]_model_[DATA_ID]_[optuna_name]_[date]/`

**Pliki:**
- `lstm_price_model.pth` - Wytrenowany model (state_dict)
- `feature_scaler.pkl` - Scaler cech z treningu
- `price_scaler.pkl` - Scaler cen z treningu
- `model_config.pkl` - Pełna konfiguracja modelu
- `training_curves.png` - Train/val loss curves
- `training.log` - Log treningu

**Przykład nazwy:** `L001_btc_model_B001_best_0312/`

---

### 3.4 Test Model
**Lokalizacja:** `shared/models/lstm_data/[TEST_ID]_test_results_[MODEL_ID]_[DATA_ID]_[date]/`

**Pliki:**
- `results.csv` - Detailowe wyniki (actual, predicted, errors, directions)
- `summary.txt` - Podsumowanie metryk
- `results.png` - 4-panel visualization (time series, scatter, errors)
- `timeseries.png` - Pełna seria czasowa
- `directional.png` - Directional accuracy analysis

**Przykład nazwy:** `T001_test_results_L001_B001_0312/`

---

## 4. Ograniczenia kodu

### 4.1 Format danych wejściowych

**Wymagane kolumny CSV:**
- `open_time` - timestamp (Unix ms LUB parseable date string)
- `open` - cena otwarcia
- `high` - najwyższa cena
- `low` - najniższa cena
- `close` - cena zamknięcia
- `volume` - wolumen (opcjonalnie, ale zalecane)

**Opcjonalne kolumny:**
- `close_time` - timestamp zamknięcia
- Inne kolumny Binance API są ignorowane

**Format timestampu:**
- Unix milliseconds (np. 1609459200000)
- LUB parseable string (np. "2021-01-01 00:00:00")

---

### 4.2 Minimalne wymagania danych

**Dla prepare_data:**
- Minimum 72 records (dla generowania cech technicznych)
- Zalecane: 1000+ records dla stabilnych metryk
- Po usunięciu NaN musi zostać: `lookback + horizon + 100` records

**Dla treningu:**
- Train split: minimum `lookback + horizon` sequences
- Val split: minimum 50 sequences
- Zalecane: 10,000+ sequences dla dobrej generalizacji

**Dla feature engineering:**
- Niektóre features wymagają 60+ timestepów (np. return_60, SMA_30)
- NaN są tworzone na początku serii (przez rolling operations)
- Kod automatycznie usuwa NaN

---

### 4.3 Ograniczenia techniczne

**Model architecture:**
- Input: `[batch, lookback, n_features]`
- Output: single value (następna cena)
- Nie wspiera multi-step prediction (tylko horizon=1,3,6,12 pojedynczo)

**Memory:**
- Duże datasety (500k+ samples) mogą wymagać `use_subset=True`
- GPU memory dla batch_size=256: ~2GB
- Tuning z n_trials=100 może trwać kilka godzin

**Scheduler limitations:**
- `cosine` scheduler może nie działać z very short training
- `plateau` scheduler wymaga stabilnych metryk

---

### 4.4 Inne ograniczenia

**Symbol detection:**
- Automatyczne tylko dla: BTC, ETH, BNB, SOL, ADA, XRP, DOGE, DOT, MATIC, AVAX
- Inne symbole oznaczone jako "crypto"

**Interval detection:**
- Rozpoznawane: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- Inne oznaczone jako "[seconds]s"

**Log prices:**
- Wymaga `close > 0` (nie może być 0 ani ujemne)
- Zalecane dla stabilności treningu
- Może nie działać dla bardzo małych wartości (close < 0.01)

---

## 5. Przepływ danych

### 5.1 Pełny Pipeline (recommended workflow)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. RAW DATA                                                 │
│    - CSV z Binance API (OHLCV + timestamp)                 │
│    - Lokalizacja: shared/files/raw_data/                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPARE DATA                                             │
│    - Feature engineering (28 technical indicators)          │
│    - Train/Agent/Test split (months or %)                  │
│    - Scaling (features + prices)                           │
│    - Sequence creation [samples, lookback, features]       │
│    Output: shared/files/lstm_prepared_data/[DATA_ID]_prep/ │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. HYPERPARAMETER TUNING (optional but recommended)         │
│    - Optuna optimization (100 trials)                       │
│    - Search space: architecture, learning params, lookback │
│    - TimeSeriesSplit validation                            │
│    Output: shared/models/lstm_data/[OPTUNA_ID]_tuning/     │
│    Key file: best_params.json                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TRAIN MODEL                                              │
│    - Load prepared data (3D sequences)                     │
│    - Optional: load params from tuning                     │
│    - LSTM training with early stopping                     │
│    - Save best model + scalers + config                   │
│    Output: shared/models/lstm_models/[MODEL_ID]_model/     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. TEST MODEL                                               │
│    - Load model + config                                   │
│    - Auto-detect test data from metadata                   │
│    - Generate predictions + metrics                        │
│    - Create visualizations                                 │
│    Output: shared/models/lstm_data/[TEST_ID]_test_results/ │
└─────────────────────────────────────────────────────────────┘
```

---

### 5.2 Przepływ danych szczegółowy

#### **Etap 1: Raw → Prepared**

```
Raw CSV (N rows)
    ↓
[prepare_features()]
    ↓ +60 NaN rows (rolling features)
Clean (N-60 rows)
    ↓
[split_by_months/percentage]
    ↓
Train DF    Agent DF    Test DF
(70%)       (10%)       (20%)
    ↓           ↓           ↓
[StandardScaler.fit()]
    ↓
[transform all splits]
    ↓
[log(prices)] if use_log_prices
    ↓
[create_sequences(lookback, horizon)]
    ↓
X: [samples, lookback, features]
y: [samples] (price at t+horizon)
    ↓
Save as NPY + metadata
```

**Data transformations:**
1. Features: `StandardScaler` fitted TYLKO na train
2. Prices: log → StandardScaler fitted TYLKO na train
3. Test używa train scalers (prevent leakage)

---

#### **Etap 2: Prepared → Tuning**

```
Prepared Data
    ↓
[TimeSeriesSplit(n_splits=3)]
    ↓
For each trial (100x):
    ├─ Suggest hyperparameters
    ├─ Create model
    ├─ Train (max 40 epochs)
    ├─ Evaluate on validation
    └─ Prune if poor
    ↓
Best trial → best_params.json
    ↓
Train can load these params
```

**Optuna search space:**
- Architecture: hidden_size [32-256], num_layers [1-3]
- Regularization: dropout [0.0-0.5], weight_decay [1e-6, 1e-3]
- Training: lr [1e-4, 1e-2], batch_size [128, 256, 512]
- Optional: lookback [12-72], horizon [1-12]

---

#### **Etap 3: Prepared → Training**

```
Prepared Data (or Raw CSV)
    ↓
[load X_train, y_train, X_val, y_val]
    ↓
[Create DataLoader]
    ↓
For each epoch:
    ├─ Forward pass (LSTM)
    ├─ Calculate MSE loss
    ├─ Backward + gradient clipping
    ├─ Update weights (AdamW)
    ├─ Validate
    └─ Early stopping check
    ↓
Save best model (lowest val loss)
    ↓
Save scalers + config
```

**Model architecture flow:**
```
Input [batch, lookback, features]
    ↓
LSTM(input, hidden, num_layers, dropout)
    ↓
Take last timestep output [:, -1, :]
    ↓
FC(hidden → hidden/2)
    ↓
BatchNorm1d
    ↓
ReLU
    ↓
Dropout
    ↓
FC(hidden/2 → 1)
    ↓
Output [batch] (predicted price)
```

---

#### **Etap 4: Model → Testing**

```
Load Model + Config
    ↓
Load Test Data (auto-detect or specified)
    ↓
[Batch inference]
    ↓
Predictions (scaled)
    ↓
[inverse_transform with saved scalers]
    ↓
Real prices (actual + predicted)
    ↓
[Calculate metrics]
    ├─ MAE, RMSE, MAPE, R²
    ├─ Error statistics
    └─ Directional accuracy
    ↓
[Create visualizations]
    ├─ Time series plot
    ├─ Scatter plot
    ├─ Error distributions
    └─ Confusion matrix (direction)
    ↓
Save results + plots
```

---

### 5.3 Data IDs tracking

Cały pipeline używa ID dla trackingu:

```
DATA_ID (B001, E001, etc.)
    ↓ prepare_data
    ↓
OPTUNA_ID (O001, O002, etc.)
    ↓ tune (references DATA_ID)
    ↓
MODEL_ID (L001, L002, etc.)
    ↓ train (references DATA_ID, optional OPTUNA_ID)
    ↓
TEST_ID (T001, T002, etc.)
    ↓ test (references MODEL_ID, DATA_ID)
```

**Metadata chain:**
- Każdy plik zawiera `data_id` w nazwie lub metadata
- Model config zawiera `data_id` treningu
- Test results zawierają `model_id` i `data_id`
- Pełna traceability od raw data do wyników

---

## 6. Serwisy i konfiguracja RabbitMQ

### 6.1 Architektura serwisów

```
┌─────────────────────────────────────────────────────────────┐
│                         GATEWAY                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ FastAPI Routes (ml_routes.py)                          │ │
│  │  - POST /ml/lstm_prepare_data                          │ │
│  │  - POST /ml/lstm_tune                                  │ │
│  │  - POST /ml/lstm_train                                 │ │
│  │  - POST /ml/lstm_test                                  │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │ MlService (ml_service.py)                              │ │
│  │  - Tworzy request_payload                              │ │
│  │  - Generuje rabbit_msg_id                              │ │
│  │  - Zapisuje w SQLite (state="created")                 │ │
│  └────────────────────┬───────────────────────────────────┘ │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼ ML_REQUEST_QUEUE
┌─────────────────────────────────────────────────────────────┐
│                      RABBITMQ                               │
│                                                             │
│  ML_REQUEST_QUEUE  ────────────▶  ML_RESPONSE_QUEUE        │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ consume
┌─────────────────────────────────────────────────────────────┐
│                      ML-MODULE                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ RabbitMQService (rabbitmq_service.py)                  │ │
│  │  - consume(ML_REQUEST_QUEUE)                           │ │
│  │  - publish(ML_RESPONSE_QUEUE)                          │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │ RabbitMQRequestService (rabbitmq_request_service.py)   │ │
│  │  - handle_request(data)                                │ │
│  │  - Match action:                                       │ │
│  │    • "lstm_prepare_data"                               │ │
│  │    • "lstm_tune"                                       │ │
│  │    • "lstm_train"                                      │ │
│  │    • "lstm_test"                                       │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │ LSTMService (lstm_service.py)                          │ │
│  │  - prepare_data()                                      │ │
│  │  - tune_hyperparameters()                              │ │
│  │  - train_model()                                       │ │
│  │  - test_model()                                        │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       │                                      │
│  ┌────────────────────▼───────────────────────────────────┐ │
│  │ LSTM Model (lstm_model/)                               │ │
│  │  - prepare_data.py                                     │ │
│  │  - lstm_hyperparameter_tuning.py                       │ │
│  │  - lstm_train_price.py                                 │ │
│  │  - lstm_test_price.py                                  │ │
│  └────────────────────┬───────────────────────────────────┘ │
└────────────────────────┼────────────────────────────────────┘
                         │ result
                         ▼ ML_RESPONSE_QUEUE
┌─────────────────────────────────────────────────────────────┐
│                      GATEWAY                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ MlService.handle_response()                            │ │
│  │  - Update SQLite (state="success"/"failed")            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

### 6.2 LSTMService szczegóły

**Inicjalizacja:**
```python
class LSTMService:
    def __init__(self, shared_dir: str = None):
        # Find project root (where /shared exists)
        # Setup directories:
        # - shared/models/lstm_models/
        # - shared/models/lstm_data/
        # - shared/files/lstm_prepared_data/
        # - shared/files/raw_data/
        
        # Add lstm_model/ to sys.path for imports
```

**Metody core:**

1. **`_resolve_data_path(path)`**
   - Próbuje: prepared_data → raw_data → absolute path
   - Automatyczne znajdowanie danych

2. **`prepare_data(rabbit_msg_id, ...)`**
   - Zmienia cwd → `lstm_prepared_data/`
   - Wywołuje `prepare_data.prepare_data()`
   - Zwraca: data_id, output_path

3. **`tune_hyperparameters(rabbit_msg_id, ...)`**
   - Konfiguruje `TuningConfig`
   - Zmienia cwd → `lstm_data/`
   - Wywołuje `run_tuning()`
   - Zwraca: best_trial, best_params

4. **`train_model(rabbit_msg_id, ...)`**
   - Zmienia cwd → `models/`
   - Wywołuje `train_model()`
   - Obsługuje single model LUB train_all_top
   - Zwraca: model_id, output_dir, best_val_loss

5. **`test_model(rabbit_msg_id, ...)`**
   - Zmienia cwd → `models/`
   - Wywołuje `test_model()`
   - Auto-detect test data
   - Zwraca: metrics (MAE, RMSE, MAPE, R², dir_acc)

**Error handling:**
```python
try:
    # Operation
    return {
        "rabbit_msg_id": rabbit_msg_id,
        "result": "success",
        "message": "...",
        "data": {...}
    }
except Exception as e:
    return {
        "rabbit_msg_id": rabbit_msg_id,
        "result": "failed",
        "message": str(e)
    }
```

---

### 6.3 RabbitMQ flow szczegóły

**Request format (Gateway → ML-Module):**
```json
{
  "action": "lstm_train",
  "rabbit_msg_id": "uuid-1234-5678",
  "train_path": "B001_prep_btc_data_full_1m",
  "epochs": 50,
  "horizon": 1,
  ...
}
```

**Response format (ML-Module → Gateway):**
```json
{
  "rabbit_msg_id": "uuid-1234-5678",
  "result": "success",
  "message": "Successfully trained LSTM model",
  "model_id": "L001",
  "model_dir": "/path/to/model",
  "best_val_loss": 0.001234,
  "timestamp": "2025-03-12T10:30:00"
}
```

**SQLite tracking:**
1. Gateway tworzy request → zapisuje jako "created"
2. ML-Module przetwarza → zwraca result
3. Gateway odbiera response → update jako "success"/"failed"

---

### 6.4 Konfiguracja RabbitMQ

**Queues:**
- `ML_REQUEST_QUEUE` - zadania od Gateway do ML-Module
- `ML_RESPONSE_QUEUE` - wyniki od ML-Module do Gateway

**Connection params (w `rabbitmq_configuration.py`):**
```python
RABBITMQ_HOST = "rabbitmq"  # Docker service name
RABBITMQ_PORT = 5672
RABBITMQ_USERNAME = "admin"
RABBITMQ_PASSWORD = "admin"
```

**Inicjalizacja (`services/__init__.py`):**
```python
rabbitmq_service = RabbitMQService(
    host=RABBITMQ_HOST,
    port=RABBITMQ_PORT,
    username=RABBITMQ_USERNAME,
    password=RABBITMQ_PASSWORD
)

# Attach services
lstm_service = LSTMService()

rabbitmq_request_service = RabbitMQRequestService(
    rabbitmq_service=rabbitmq_service,
    lstm_service=lstm_service,
    # ... other services
)

rabbitmq_service.request_service = rabbitmq_request_service

# Start consuming
rabbitmq_service.consume(ML_REQUEST_QUEUE, 
                         rabbitmq_service.handle_request)
```

**Consumer thread:**
- Daemon thread w tle
- Auto-reconnect on connection loss
- Message acknowledgment po przetworzeniu
- Graceful error handling

---

### 6.5 Autonomia kodu

**Kod LSTM działa niezależnie:**

```bash
# Console usage (bez RabbitMQ):
cd ml-module/lstm_model
python prepare_data.py data.csv
python lstm_train_price.py --train_path lstm_prepared_data/B001_prep/
python lstm_test_price.py --model_dir lstm_models/L001_model/
```

**Przez RabbitMQ:**
```bash
# API usage:
curl -X POST http://localhost:8000/ml/lstm_train \
  -H "Content-Type: application/json" \
  -d '{"train_path": "B001_prep_btc_data_full_1m"}'
```

**Oba tryby:**
- Używają TYCH SAMYCH plików kodu
- Zapisują w TYCH SAMYCH lokalizacjach
- Generują TAKIE SAME wyniki
- Żadnych modyfikacji w oryginalnym kodzie LSTM

---

## 7. Podsumowanie

**LSTM Price Prediction System** to kompleksowy pipeline do predykcji cen kryptowalut używający LSTM neural networks.

**Główne features:**
- ✅ Automatyczne feature engineering (28 technical indicators)
- ✅ Optuna hyperparameter tuning
- ✅ Early stopping + gradient clipping
- ✅ Comprehensive evaluation (price + directional metrics)
- ✅ Full traceability (data_id → model_id → test_id)
- ✅ Dual mode: Console + RabbitMQ API
- ✅ Auto-detection (symbol, interval, test data)

**Zalecany workflow:**
1. Prepare → 2. Tune → 3. Train → 4. Test

**Integration:**
- RabbitMQ dla async processing
- SQLite dla task tracking
- Docker dla containerization
- Shared storage dla cross-module access