# RFModelPackage — Instrukcja użytkownika
*może zawierać błędy* 😘
## Instalacja i wymagania

Package jest samodzielny — nie wymaga instalacji `rf_model` ani żadnego innego modułu z projektu.
Wymagane zależności: `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`.

Struktura package'u:
```
model_package/
├── rf_model_package.py      ← główny moduł
└── feature_engineering.py  ← lokalna kopia universal FE (nie modyfikować)
```

---

## Szybki start

```python
from rf_model_package import RFModelPackage, SignalConfig

# 1. Konfiguracja sygnałów
cfg = SignalConfig(threshold=0.001, window_sigma=100)

# 2. Załadowanie modelu
pkg = RFModelPackage('rf_models/R001_eth_rf_model_E104_2203', signal_config=cfg)

# 3a. Sygnał dla całego DataFrame (backtest / symulacja)
df_signals = pkg.predict_batch(df)

# 3b. Sygnał dla ostatniej świecy (live)
signal = pkg.predict_live(df_buffer)
```

---

## SignalConfig — parametry sygnałów

```python
@dataclasses.dataclass
class SignalConfig:
    threshold:    float = 0.001   # minimalny log-return do wygenerowania sygnału
    window_sigma: int   = 100     # okno rolling std do szacowania sigma (liczba świec)
```

### `threshold`
Minimalna ekonomicznie istotna zmiana ceny (w log-returnach). Pokrywa prowizję i spread.

| Wartość | Odpowiednik % | Kiedy używać |
|---|---|---|
| `0.0005` | ~0.05% | Bardzo niskie koszty transakcji |
| `0.001` | ~0.1% | Standardowe prowizje (domyślny) |
| `0.002` | ~0.2% | Wyższe koszty / większa ostrożność |
| `0.005` | ~0.5% | Konserwatywny — tylko duże ruchy |

### `window_sigma`
Liczba świec do obliczenia rolling std log-returnów (`sigma` = szacowana zmienność rynku).
Wyższe = stabilniejsza sigma, wolniej reaguje na zmiany reżimu zmienności.
Zalecane dla danych 1m: `50`–`200`. Domyślnie `100`.

---

## RFModelPackage — konstruktor

```python
pkg = RFModelPackage(
    model_dir,             # str: ścieżka do folderu modelu RF
    signal_config=None     # SignalConfig | None (używa domyślnych wartości)
)
```

### Wymagane pliki w `model_dir`

| Plik | Wymagany | Opis |
|---|---|---|
| `rf_model.pkl` | TAK | Wagi modelu (joblib) |
| `model_config.json` | TAK | Konfiguracja: features, lookback, horizon, flattening |
| `feature_scaler.pkl` lub `scaler_features.pkl` | TAK | Skaler cech |
| `target_scaler.pkl` | NIE | Skaler targetu (ładowany z JSON jeśli dostępny) |

---

## Metody publiczne

### `predict_batch(df, data_mode='auto') → pd.DataFrame`

Przetwarza cały DataFrame i zwraca sygnały dla każdej ważnej próbki.

```python
df_signals = pkg.predict_batch(df)
# df_signals.columns: ['timestamp', 'mu', 'sigma', 'signal_buy', 'signal_sell']
```

**Kolumny wynikowe:**

| Kolumna | Typ | Opis |
|---|---|---|
| `timestamp` | wartość z `open_time` | Czas świecy t |
| `mu` | float | Przewidziany log-return po inverse_transform |
| `sigma` | float | Rolling std log-returnów (zmienność rynku) |
| `signal_buy` | float [0–100] | P(r > threshold) × 100 |
| `signal_sell` | float [0–100] | P(r < −threshold) × 100 |

Wiersze gdzie `sigma` jest NaN (pierwsze `window_sigma−1` świec) są automatycznie usuwane.

---

### `predict_live(df_buffer, data_mode='auto') → dict`

Zwraca sygnał dla ostatniej świecy. Wewnętrznie wywołuje `predict_batch` i zwraca ostatni wiersz.

```python
signal = pkg.predict_live(df_buffer)
# {'timestamp': ..., 'mu': 0.00123, 'sigma': 0.00089, 'signal_buy': 67.4, 'signal_sell': 5.2}
```

Caller musi utrzymywać rolling buffer o minimalnej liczbie wierszy:
```python
min_rows = pkg.min_rows_required   # lookback + window_sigma + 100 (warmup FE)
```

---

### `get_model_info() → dict`

Zwraca metadane załadowanego modelu.

```python
info = pkg.get_model_info()
# Kluczowe pola:
# info['model_id'], info['symbol'], info['lookback'], info['horizon']
# info['flattening_method'], info['n_features'], info['n_flat_features']
# info['criterion'], info['n_estimators']
# info['val_metrics']        → {'mae': ..., 'r2': ..., 'dir_acc': ...}
# info['oob_score']          → OOB R² z treningu (lub None)
# info['dir_acc_threshold']  → próg z konfiguracji modelu (lub None)
# info['signal_threshold']   → aktywny próg SignalConfig
# info['signal_window_sigma'] → aktywne okno sigma
```

---

### `min_rows_required` (property)

```python
n = pkg.min_rows_required
# = config['lookback'] + signal_config.window_sigma + 100
```

Minimalna liczba wierszy w buforze wejściowym dla `predict_live`.

---

## Tryby danych (`data_mode`)

| Tryb | Opis | Wymagane kolumny |
|---|---|---|
| `'auto'` (domyślny) | Auto-detect na podstawie obecności kolumn cech | — |
| `'raw'` | Surowe dane OHLCV z Binance — FE generowane automatycznie | `open_time, open, high, low, close, volume` |
| `'prepared'` | Dane po feature engineering i skalowaniu (z `data_preparation`) | kolumny cech + `close` + `open_time` |

**Auto-detect:** sprawdza czy pierwsza cecha z konfiguracji modelu istnieje w DataFrame. Jeśli tak → tryb `prepared`, jeśli nie → tryb `raw`.

### Minimalna liczba wierszy

| Tryb | Min wierszy |
|---|---|
| `raw` | `lookback + window_sigma + 100` (warmup FE) |
| `prepared` | `lookback + window_sigma` |

---

## Model sygnałów (matematyka)

```
mu    = przewidziany log-return (po inverse_transform skalera targetu)
sigma = rolling_std(log(close[t]/close[t-1]), window=window_sigma)
T     = threshold

signal_buy  = (1 - Phi((T  - mu) / sigma)) * 100   ∈ [0, 100]
signal_sell = Phi((-T - mu) / sigma) * 100           ∈ [0, 100]
```

Gdzie `Phi` to dystrybuanta rozkładu normalnego (N(0,1)).

**Właściwości:**
- `signal_buy + signal_sell < 100` zawsze (różnica = strefa hold)
- Przy `mu = 0, sigma = threshold`: `signal_buy ≈ signal_sell ≈ 15.9`
- Przy `mu >> threshold`: `signal_buy → 100, signal_sell → 0`
- `sigma → 0`: sygnały degenerują do 0 lub 100 (sigma clipped do `1e-8`)

---

## Implementacja — scenariusze

### Backtest / symulacja

```python
import pandas as pd
from rf_model_package import RFModelPackage, SignalConfig

cfg = SignalConfig(threshold=0.001, window_sigma=100)
pkg = RFModelPackage('rf_models/R025_eth_rf_model_E104_2203', signal_config=cfg)

# Załaduj dane historyczne (surowe OHLCV lub prepared CSV)
df = pd.read_csv('data/eth_1m_2024.csv')

# Wygeneruj sygnały dla całego okresu
signals = pkg.predict_batch(df, data_mode='auto')

print(f"Samples: {len(signals)}")
print(signals[['timestamp', 'mu', 'signal_buy', 'signal_sell']].head())
```

---

### Live trading — rolling buffer

```python
from collections import deque
import pandas as pd
from rf_model_package import RFModelPackage, SignalConfig

cfg = SignalConfig(threshold=0.001, window_sigma=100)
pkg = RFModelPackage('rf_models/R025_eth_rf_model_E104_2203', signal_config=cfg)

min_rows = pkg.min_rows_required  # np. 197 przy lookback=48, window_sigma=100, warmup=49
buffer = deque(maxlen=min_rows)

def on_new_candle(candle_dict):
    """Wywoływane przy każdej nowej świecy."""
    buffer.append(candle_dict)

    if len(buffer) < min_rows:
        return None   # za mało danych

    df_buf = pd.DataFrame(list(buffer))
    signal = pkg.predict_live(df_buf)

    return signal
    # signal = {
    #   'timestamp': ...,
    #   'mu':          0.00123,   # oczekiwany log-return
    #   'sigma':       0.00089,   # bieżąca zmienność
    #   'signal_buy':  67.4,      # P(wzrost > threshold) %
    #   'signal_sell': 5.2        # P(spadek > threshold) %
    # }
```

---

### Tryb prepared data (szybszy — brak FE)

```python
# Dane z data_preparation — cechy już obliczone i zaskalowane
df_prep = pd.read_csv('data_preparation/prepared_data/E104_prep_eth_universal_full_1m/test.csv')

signals = pkg.predict_batch(df_prep, data_mode='prepared')
```

Tryb `prepared` jest szybszy — pomija `prepare_features()`. Wymaga żeby dane były zaskalowane tym samym `feature_scaler` co podczas treningu.

---

### Zmiana parametrów sygnałów bez przeładowania modelu

```python
# Załaduj raz
pkg = RFModelPackage('rf_models/R025_eth_rf_model_E104_2203')

# Zmień konfigurację na nowy obiekt
pkg._sig_cfg = SignalConfig(threshold=0.002, window_sigma=50)

# Wszystkie kolejne predict_* używają nowej konfiguracji
signals = pkg.predict_batch(df)
```

---

## Interpretacja sygnałów

```python
signal = pkg.predict_live(df_buffer)

buy  = signal['signal_buy']
sell = signal['signal_sell']
hold = 100 - buy - sell   # strefa niepewności

# Przykładowa logika progowa
if buy > 70:
    action = 'BUY'
elif sell > 70:
    action = 'SELL'
else:
    action = 'HOLD'
```

**Orientacyjne wartości progów:**

| Próg | Znaczenie |
|---|---|
| `signal_buy > 50` | Model lekko faworyzuje wzrost |
| `signal_buy > 70` | Silny sygnał kupna |
| `signal_buy > 85` | Bardzo silny sygnał kupna |
| `signal_sell > 70` | Silny sygnał sprzedaży |

Progi należy kalibrować empirycznie na danych historycznych dla konkretnego modelu i pary.

---

## Obsługa błędów

```python
from rf_model_package import RFModelPackage, SignalConfig

try:
    pkg = RFModelPackage('rf_models/R025_eth_rf_model_E104_2203')
except FileNotFoundError as e:
    print(f"Brak pliku modelu: {e}")
    # Sprawdź czy rf_model.pkl i model_config.json istnieją w podanym folderze

try:
    signals = pkg.predict_batch(df)
except ValueError as e:
    print(f"Problem z danymi: {e}")
    # Typowe przyczyny:
    # - za mało wierszy w df
    # - brakujące kolumny (tryb raw: open_time, open, high, low, close, volume)
    # - ujemne lub zerowe wartości close
    # - NaN w kolumnie close

# Sprawdź minimalną liczbę wierszy przed wywołaniem
print(f"Wymagane minimum: {pkg.min_rows_required} wierszy")
print(f"Dostępne wiersze: {len(df)}")
```

---

## Ograniczenia i uwagi

**Leakage:** package nie wprowadza leakage — flattening używa `shift(1)` na wszystkich cechach, `sigma` obliczana jest z danych do świecy `t` włącznie (nie `t+1`).

**Flattening `signal`:** package obsługuje wszystkie 4 metody flatteningu (`lag`, `agg`, `hybrid`, `signal`) — metoda jest czytana automatycznie z `model_config.json`.

**Target scaler:** odtwarzany priorytetowo z `target_scaler_params` w `model_config.json` (nie wymaga obecności modułu `rf_model.common`). Fallback na `target_scaler.pkl` jeśli params brak. Obsługuje oba typy: `robust` (RobustScaler) i `standard` (StandardScaler) z pełnym ustawieniem `n_features_in_` — zgodny z nowszymi wersjami sklearn.

**Feature scaler:** akceptuje obie konwencje nazewnictwa: `feature_scaler.pkl` i `scaler_features.pkl`.

**Wątek-bezpieczeństwo:** instancja `RFModelPackage` nie jest thread-safe — przy wielowątkowym użyciu twórz osobną instancję per wątek.
