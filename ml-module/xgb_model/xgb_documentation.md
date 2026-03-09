# XGBoost Cryptocurrency Prediction — Dokumentacja

## Spis treści

1. [Przegląd systemu](#1-przegląd-systemu)
2. [Architektura i struktura plików](#2-architektura-i-struktura-plików)
3. [Przepływ danych](#3-przepływ-danych)
4. [Feature Engineering](#4-feature-engineering)
5. [Konfiguracja modelu (xgb_config.py)](#5-konfiguracja-modelu-xgb_configpy)
6. [Trening modelu (xgb_train.py)](#6-trening-modelu-xgb_trainpy)
7. [Tuning hiperparametrów (xgb_hyperparameter_tuning.py)](#7-tuning-hiperparametrów-xgb_hyperparameter_tuningpy)
8. [Testowanie modelu (xgb_test.py)](#8-testowanie-modelu-xgb_testpy)
9. [Metryki i ich interpretacja](#9-metryki-i-ich-interpretacja)
10. [Przykłady użycia](#10-przykłady-użycia)
11. [System identyfikatorów](#11-system-identyfikatorów)
12. [Pliki wyjściowe](#12-pliki-wyjściowe)

---

## 1. Przegląd systemu

Model XGBoost przewiduje **log-return** (logarytmiczną stopę zwrotu) kryptowaluty dla zadanego horyzontu czasowego. Jest to model regresji tablicowej (flat/tabular), który:

- przyjmuje dane OHLCV w formacie 1-minutowym (lub innych interwałach),
- generuje cechy techniczne (feature engineering),
- "spłaszcza" sekwencję historyczną (`lookback` świec) do pojedynczego wektora cech (`flattening`),
- przewiduje `log(close[t+horizon] / close[t])` — czyli oczekiwany procentowy ruch ceny.

**Target: log-return, nie cena absolutna.** Wartość `+0.001` oznacza oczekiwany wzrost o ~0.1%, wartość `-0.002` oznacza oczekiwany spadek o ~0.2%.

---

## 2. Architektura i struktura plików

```
ml/
├── feature_engineering.py          ← UNIVERSAL — generuje cechy OHLCV
│
├── xgb_model/
│   ├── xgb_config.py               ← Konfiguracja (hiperparametry, ścieżki, cele)
│   ├── xgb_train.py                ← Trening modelu
│   ├── xgb_test.py                 ← Testowanie modelu
│   ├── xgb_hyperparameter_tuning.py ← Optuna tuning
│   │
│   ├── common/
│   │   ├── data_utils.py           ← Ładowanie danych, flattening, scalery targetu
│   │   ├── metrics.py              ← Obliczanie metryk (log-return space)
│   │   ├── training_utils.py       ← Pętla treningowa, zapis modelu
│   │   ├── id_utils.py             ← System identyfikatorów (X001, XO001, XT001)
│   │   └── visualization.py        ← Wykresy wyników i feature importance
│   │
│   ├── xgb_models/                 ← Zapisane modele (X001_btc_xgb_model_...)
│   ├── xgb_data/                   ← Wyniki tuningu i testów
│   └── xgb_prepared_data/          ← Dane przygotowane przez data_preparation/
│
└── data_preparation/
    └── prepare_data.py             ← Przygotowanie danych (CSV → prepared dir)
```

---

## 3. Przepływ danych

### 3.1 Schemat ogólny

```
CSV / Prepared Dir
       │
       ▼
feature_engineering.prepare_features()   ← generuje ~23 cechy techniczne
       │
       ▼
flatten_features()                        ← sekwencja lookback → wektor
  ['lag' | 'agg' | 'hybrid']
       │
       ▼
create_log_return_target()               ← y = log(close[t+h] / close[t])
       │
       ▼
StandardScaler (cechy) + TargetScaler (target)
       │
       ▼
XGBoost.train()   ←→   early stopping na zbiorze walidacyjnym
       │
       ▼
Predykcja → inverse_transform → log-return → metryki
```

### 3.2 Ładowanie danych

`load_data()` w `common/data_utils.py` obsługuje dwa tryby:

**Tryb 1 — Prepared directory** (zalecany):

Folder przygotowany przez `data_preparation/prepare_data.py` zawiera:
- `train.csv` — dane treningowe (po feature engineering i skalowaniu cech)
- `test.csv` — dane testowe
- `scaler_features.pkl` — skaler cech
- `metadata.json` — konfiguracja (lookback, horizon, features, itp.)

```
xgb_prepared_data/
└── XB001_prep_btc_full_1m/
    ├── train.csv
    ├── test.csv
    ├── scaler_features.pkl
    └── metadata.json
```

**Tryb 2 — Raw CSV** (on-the-fly):

Surowy plik CSV z kolumnami OHLCV. Feature engineering i skalowanie wykonywane są automatycznie podczas ładowania.

### 3.3 Flattening — spłaszczanie sekwencji

XGBoost nie obsługuje danych sekwencyjnych (jak LSTM). Sekwencja `lookback` świec jest przekształcana w pojedynczy wiersz cech jedną z trzech metod:

**`lag`** — proste opóźnienia:
Dla każdej cechy tworzy kolumny `feature_lag_1`, `feature_lag_2`, ..., `feature_lag_{lookback}`.
Wynikowy wymiar: `n_features × lookback`. Przy 23 cechach i lookback=48 → 1104 kolumny.
Zachowuje pełną informację o kolejności, ale tworzy dużo cech — wolniejszy.

**`agg`** (domyślny) — agregacje:
Dla każdej cechy tworzy 5 statystyk z okna `[t-lookback, t-1]`: mean, std, min, max, last.
Wynikowy wymiar: `n_features × 5`. Przy 23 cechach → 115 kolumn.
Szybki, kompaktowy, dobry punkt startowy.

**`hybrid`** — kombinacja strategicznych opóźnień i agregacji wielookienkowych:
- opóźnienia: `[1, 2, 3, 5, 10, 24]` (domyślnie)
- okna agregacji: mean i std dla `[6, 12, 24, 48]`
- różnice pierwszego rzędu: `diff_1`, `diff_5`
Wynikowy wymiar: zależy od konfiguracji; zwykle 200–400 kolumn.
Najlepszy kompromis między szczegółowością a rozmiarem.

### 3.4 Target: log-return

```python
y = log(close[t + horizon] / close[t])
```

Gdzie `t` to moment na który patrzymy (ostatnia znana świeca), a `close[t+horizon]` to cena za `horizon` świec w przyszłość.

Wartości:
- `+0.002` → wzrost o ~0.2%
- `-0.001` → spadek o ~0.1%
- `0.0` → brak zmiany

---

## 4. Feature Engineering

Źródło: `feature_engineering.py` (universal — wspólny dla LSTM i XGBoost).

### 4.1 Generowane cechy

Wszystkie cechy generowane są przez `prepare_features(df)` na podstawie kolumn OHLCV.

**Log-returns** (4 cechy):

| Cecha | Opis |
|---|---|
| `return_1` | log(close[t] / close[t-1]) — zmiana 1-świecowa |
| `return_5` | log(close[t] / close[t-5]) |
| `return_15` | log(close[t] / close[t-15]) |
| `return_60` | log(close[t] / close[t-60]) |

**SMA Ratios** (3 cechy):

| Cecha | Opis |
|---|---|
| `close_sma_7` | close / SMA(7) — pozycja ceny względem 7-świecowej średniej |
| `close_sma_14` | close / SMA(14) |
| `close_sma_30` | close / SMA(30) |

Wartość > 1.0 = cena powyżej średniej, < 1.0 = poniżej.

**MACD** (3 cechy):

| Cecha | Opis |
|---|---|
| `macd` | EMA(12) - EMA(26) |
| `macd_signal` | EMA(9) z MACD |
| `macd_diff` | macd - macd_signal (histogram) |

**RSI** (1 cecha):

| Cecha | Opis |
|---|---|
| `rsi` | Relative Strength Index (14-period). Zakres [0, 100]. >70 = wykupienie, <30 = wyprzedanie |

**Volatility** (3 cechy):

| Cecha | Opis |
|---|---|
| `volatility_7` | rolling std of return_1, window=7 |
| `volatility_14` | rolling std of return_1, window=14 |
| `volatility_30` | rolling std of return_1, window=30 |

**Volatility Regime** (1 cecha):

| Cecha | Opis |
|---|---|
| `vol_regime` | z-score bieżącej zmienności (30-bar) na tle 500-barowego okna |

Interpretacja: 0.0 = medianowy reżim, +2.0 = podwyższona zmienność, -1.0 = spokojny rynek. Wymaga ~530 świec do stabilizacji.

**Volume** (1 cecha):

| Cecha | Opis |
|---|---|
| `relative_volume` | volume / SMA(20) volume — relatywny wolumen |

Wartość > 1.0 = większy niż przeciętny wolumen.

**Buy Pressure** (opcjonalna, 1 cecha):

| Cecha | Opis |
|---|---|
| `buy_pressure` | taker_buy_base_asset_volume / total_volume ∈ [0, 1] |

Dostępna tylko gdy dane zawierają kolumnę `taker_buy_base_asset_volume` (pełny eksport Binance). Jeden z najsilniejszych predyktorów krótkoterminowych ruchów.

**Bollinger Bands** (1 cecha):

| Cecha | Opis |
|---|---|
| `bb_position` | (close - BB_lower) / (BB_upper - BB_lower) ∈ [0, 1] |

0.0 = dolna banda, 1.0 = górna banda, 0.5 = środek.

**High-Low Range** (1 cecha):

| Cecha | Opis |
|---|---|
| `hl_range` | (high - low) / close — relatywny zakres świecy |

**Time features** (4 cechy):

| Cecha | Opis |
|---|---|
| `hour_sin` | sin(2π × hour / 24) |
| `hour_cos` | cos(2π × hour / 24) |
| `dow_sin` | sin(2π × day_of_week / 7) |
| `dow_cos` | cos(2π × day_of_week / 7) |

Cykliczne kodowanie godziny i dnia tygodnia (zachowuje ciągłość 23h→0h, piątek→poniedziałek).

**Łącznie: 22–23 cech** (bez / z buy_pressure).

### 4.2 Zestawy cech (feature sets)

#### Zestawy uniwersalne (LSTM i XGBoost)

| Zestaw | Cechy | Opis |
|---|---|---|
| `full` | wszystkie (22–23) | Kompletny zestaw |
| `technical` | 15 | Bez time features i vol_regime — bezpieczny dla krótkich okien |
| `technical_bp` | 16 | `technical` + buy_pressure |
| `minimal` | 6 | return_1, return_5, close_sma_14, rsi, volatility_14, relative_volume |

#### Zestawy dedykowane XGBoost

| Zestaw | Cechy | Kiedy używać |
|---|---|---|
| `xgb_momentum` | 11–12 | Trend/momentum z vol_regime; standardowy wybór |
| `xgb_volatility` | 9 | Fokus na reżim zmienności i rozstępy cen |
| `xgb_returns` | 4 | Tylko returny — maksymalna przenośność danych |

#### Zestawy dedykowane LSTM (nie używać w XGBoost)

`lstm_momentum`, `lstm_volatility`, `lstm_price` — zawierają time features, które po flattening stają się redundantne w XGBoost.

#### Aliasy legacy

| Alias | Mapuje do |
|---|---|
| `momentum` | `xgb_momentum` |
| `volatility_focused` | `xgb_volatility` |

---

## 5. Konfiguracja modelu (xgb_config.py)

### 5.1 Parametry danych

```python
DEFAULT_LOOKBACK = 48          # liczba świec historycznych (sekwencja wejściowa)
DEFAULT_HORIZON  = 12          # horyzont predykcji (świece w przyszłość)
DEFAULT_FLATTENING = 'agg'     # metoda spłaszczania: 'lag' | 'agg' | 'hybrid'
DEFAULT_FEATURE_SET = 'full'   # zestaw cech
```

`lookback=48` przy danych 1-minutowych = ostatnie 48 minut historii.
`horizon=12` = przewidujemy cenę za 12 minut.

### 5.2 Funkcja straty (OBJECTIVE)

```python
OBJECTIVE = 'reg:pseudohubererror_large'
HUBER_SLOPE = 10.0
```

Dostępne funkcje straty:

| Wartość | Opis | Kiedy używać |
|---|---|---|
| `reg:pseudohubererror_large` | Pseudo-Huber δ=10 (domyślna) | Odporność na outliery, bliżej MSE — polecana |
| `reg:pseudohubererror` | Pseudo-Huber δ=1 | Silniejsza odporność, bliżej MAE |
| `reg:squarederror` | MSE — średnia kwadratów błędów | Wrażliwa na outliery; dobra gdy dane czyste |
| `reg:absoluteerror` | MAE — medianowa regresja | Tendencja do "spłaszczania" predykcji do 0 |

**Niezgodne** (nie używać z log-return targetem):
- `reg:squaredlogerror` — wymaga etykiet > -1
- `rank:ndcg` — wymaga nieujemnych etykiet całkowitych

`reg:pseudohubererror_large` to wirtualna nazwa obsługiwana wewnętrznie — XGBoost otrzymuje `reg:pseudohubererror` z parametrem `huber_slope=10`.

### 5.3 Skalowanie targetu

```python
SCALE_TARGET = True
TARGET_SCALER_TYPE = 'robust'   # 'standard' | 'robust'
```

Log-returny mają małe wartości (rząd 0.001), co powoduje problemy z gradientami XGBoost. Skalowanie normalizuje target do ~N(0,1) przed treningiem i odwraca po predykcji.

| Typ | Opis | Kiedy używać |
|---|---|---|
| `standard` | StandardScaler — mean=0, std=1 | Dane bez ekstremalnych outlierów |
| `robust` | RobustScaler — mediana i IQR | Dane z flash crashami / pump&dump (zalecany) |

Skalowanie jest **odwracalne** — metryki testowe zawsze raportowane są w oryginalnej przestrzeni log-return.

### 5.4 Winsoryzacja targetu

```python
WINSORIZE_TARGET = False
WINSORIZE_PERCENTILE = 99.0
```

Obcina skrajne wartości log-return (np. flash crashe). Granice wyznaczane **tylko** na danych treningowych (brak data leakage). Val/test obcinane tymi samymi granicami.

Przykład z `WINSORIZE_PERCENTILE=99.0`: obcina zewnętrzne 1% rozkładu (dolne 0.5% i górne 0.5%).

Kiedy włączyć: gdy dane zawierają ekstremalne zdarzenia rynkowe, które są niereprezentatywne dla normalnego zachowania.

### 5.5 Cel optymalizacji Optuna

```python
OPTUNA_OPTIMIZE_TARGET = 'train'   # 'val' | 'train'
```

Kontroluje, na którym zbiorze Optuna minimalizuje stratę:
- `'val'` — zalecane w produkcji: wybiera parametry generalizujące na nowe dane
- `'train'` — przydatne do diagnozy underfittingu

### 5.6 Hiperparametry domyślne XGBoost

```python
DEFAULT_PARAMS = {
    'n_estimators':     2222,   # liczba drzew
    'learning_rate':    0.05,   # krok uczenia
    'max_depth':        6,      # głębokość drzewa
    'subsample':        0.8,    # próbkowanie wierszy per drzewo
    'colsample_bytree': 0.8,    # próbkowanie kolumn per drzewo
    'colsample_bylevel': 0.8,   # próbkowanie kolumn per poziom
    'min_child_weight': 3,      # min suma wag w liściu
    'gamma':            0.0,    # min redukcja straty do podziału
    'reg_alpha':        0.0,    # regularyzacja L1
    'reg_lambda':       1,      # regularyzacja L2
    'tree_method':      'hist', # algorytm budowy drzew (szybki)
}
EARLY_STOPPING_ROUNDS = 20000  # zatrzymanie gdy brak poprawy przez N rund
USE_BEST_ITERATION = False      # False=ostatnia iteracja, True=best_iteration
```

### 5.7 USE_BEST_ITERATION

| Wartość | Zachowanie |
|---|---|
| `False` (domyślne) | Zapisuje model z ostatniej iteracji (wszystkie `n_estimators` drzew) |
| `True` | Zapisuje model z `best_iteration` (punkt early stopping) |

`False` jest domyślny, bo `EARLY_STOPPING_ROUNDS=20000` praktycznie wyłącza early stopping — model trenuje do końca.

---

## 6. Trening modelu (xgb_train.py)

### 6.1 Uruchomienie z linii poleceń

**Podstawowe (z przygotowanymi danymi):**
```bash
python xgb_train.py --data_path xgb_prepared_data/XB001_prep_btc_full_1m
```

**Z surowym CSV:**
```bash
python xgb_train.py --data_path data/btc_1m.csv
```

**Z parametrami Optuna:**
```bash
python xgb_train.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --load_optuna_params xgb_data/XO001_xgb_optuna_XB001_0103/best_params.json
```

**Z ręcznymi nadpisaniami:**
```bash
python xgb_train.py \
    --data_path data/btc_1m.csv \
    --lookback 72 \
    --horizon 6 \
    --flattening_method hybrid \
    --feature_set xgb_momentum \
    --n_estimators 3000 \
    --learning_rate 0.01 \
    --max_depth 8 \
    --subsample 0.7 \
    --colsample_bytree 0.7 \
    --scale_target \
    --target_scaler_type robust
```

### 6.2 Opis parametrów CLI

**Dane:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--data_path` | wymagany | Ścieżka do prepared dir lub pliku .csv |
| `--load_optuna_params` | None | Ścieżka do `best_params.json` z Optuna |
| `--lookback` | 48 | Długość okna historycznego (liczba świec) |
| `--horizon` | 12 | Horyzont predykcji (liczba świec w przód) |
| `--flattening_method` | `agg` | Metoda spłaszczania: `lag`, `agg`, `hybrid` |
| `--feature_set` | `full` | Zestaw cech (patrz sekcja 4.2) |

**Hiperparametry XGBoost:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--n_estimators` | 2222 | Liczba drzew |
| `--max_depth` | 6 | Maksymalna głębokość drzewa (3–10) |
| `--learning_rate` | 0.05 | Krok uczenia (0.001–0.5) |
| `--subsample` | 0.8 | Frakcja próbkowania wierszy (0.6–1.0) |
| `--colsample_bytree` | 0.8 | Frakcja kolumn per drzewo (0.6–1.0) |
| `--colsample_bylevel` | 0.8 | Frakcja kolumn per poziom (0.6–1.0) |
| `--min_child_weight` | 3 | Min suma wag w liściu (1–10) |
| `--gamma` | 0.0 | Min redukcja straty do podziału węzła (0–2) |
| `--reg_alpha` | 0.0 | Regularyzacja L1 (0–5) |
| `--reg_lambda` | 1 | Regularyzacja L2 (0.01–5) |
| `--objective` | z config | Funkcja straty |
| `--early_stopping_rounds` | 20000 | Patience early stopping |

**Target:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--scale_target` | True (z config) | Włącz skalowanie targetu |
| `--no_scale_target` | — | Wyłącz skalowanie targetu |
| `--target_scaler_type` | `robust` | Typ skalera: `standard`, `robust` |
| `--winsorize_target` | False | Włącz winsoryzację targetu |
| `--winsorize_percentile` | 99.0 | Percentyl winsoryzacji (90–99.9) |

**Inne:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--seed` | 42 | Ziarno losowości |

### 6.3 Priorytet parametrów

Gdy podano `--load_optuna_params`, parametry z pliku JSON są ładowane jako baza. Każdy parametr CLI podany **dodatkowo** nadpisuje wartość z Optuna. Nienadpisane parametry biorą wartości z Optuna, a jeśli brak — z `XGBConfig.DEFAULT_PARAMS`.

### 6.4 Podział train/val

Podział jest **chronologiczny** (80%/20%) — brak shuffle, brak data leakage:
- Train: pierwsze 80% sekwencji
- Val: ostatnie 20% sekwencji

Early stopping monitoruje stratę na zbiorze walidacyjnym.

### 6.5 API Python

```python
from xgb_train import train

model, feature_scaler, config = train(
    data_path='xgb_prepared_data/XB001_prep_btc_full_1m',
    lookback=48,
    horizon=12,
    flattening_method='hybrid',
    feature_set='xgb_momentum',
    n_estimators=3000,
    learning_rate=0.01,
    scale_target=True,
    target_scaler_type='robust',
)

print(config['val_metrics'])   # {'mae': ..., 'r2': ..., 'dir_acc': ...}
print(config['output_dir'])    # ścieżka do zapisanego modelu
```

---

## 7. Tuning hiperparametrów (xgb_hyperparameter_tuning.py)

### 7.1 Uruchomienie

```bash
python xgb_hyperparameter_tuning.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --n_trials 100 \
    --feature_set xgb_momentum \
    --optimize_target val
```

**Z tunowaniem lookback i horizon:**
```bash
python xgb_hyperparameter_tuning.py \
    --data_path data/btc_1m.csv \
    --n_trials 200 \
    --tune_lookback \
    --tune_horizon \
    --flattening_method hybrid
```

### 7.2 Parametry CLI tuningu

| Parametr | Domyślny | Opis |
|---|---|---|
| `--data_path` | wymagany | Ścieżka do danych |
| `--n_trials` | 100 | Liczba prób Optuna |
| `--timeout` | None | Limit czasu w sekundach |
| `--feature_set` | `full` | Zestaw cech (stały podczas tuningu) |
| `--lookback` | 48 | Lookback (stały, jeśli nie `--tune_lookback`) |
| `--horizon` | 12 | Horizon (stały, jeśli nie `--tune_horizon`) |
| `--flattening_method` | None | Fiksuje metodę flatteningu (domyślnie tunowane) |
| `--tune_lookback` | False | Tuninguj lookback jako hiperparametr |
| `--tune_horizon` | False | Tuninguj horizon jako hiperparametr |
| `--scale_target` | True | Włącz skalowanie targetu |
| `--no_scale_target` | — | Wyłącz skalowanie targetu |
| `--target_scaler_type` | `robust` | Typ skalera |
| `--optimize_target` | `train` | Co minimalizuje Optuna: `val` lub `train` |

### 7.3 Przestrzeń hiperparametrów (domyślna)

| Hiperparametr | Typ | Zakres |
|---|---|---|
| `max_depth` | int | [3, 10] |
| `min_child_weight` | int | [1, 7] |
| `gamma` | float | [0.0, 2.0] |
| `subsample` | float | [0.6, 1.0] |
| `colsample_bytree` | float | [0.6, 1.0] |
| `colsample_bylevel` | float | [0.6, 1.0] |
| `learning_rate` | float (log) | [0.00005, 0.5] |
| `n_estimators` | int | [400, 4000] |
| `reg_alpha` | float | [0.0, 5.0] |
| `reg_lambda` | float | [0.01, 5.0] |
| `flattening_method` | categorical | `lag`, `agg`, `hybrid` |
| `lookback` | categorical | [24, 36, 48, 72, 96, 120] (jeśli `--tune_lookback`) |
| `horizon` | categorical | [1, 3, 6, 12, 24] (jeśli `--tune_horizon`) |

### 7.4 Algorytm i pruning

- **Sampler:** TPE (Tree-structured Parzen Estimator) — bayesowski algorytm próbkowania
- **Pruner:** MedianPruner — przerywa próby gorsze od mediany ukończonych
  - `n_startup_trials=5` — pierwsze 5 prób nie jest przycinanych
  - `n_warmup_steps=10` — pierwsze 10 epok każdej próby też nie jest przycinanych

### 7.5 Wyniki tuningu

Zapisywane w `xgb_data/XO001_xgb_optuna_{data_id}_{data}/`:

```
XO001_xgb_optuna_XB001_0103/
├── best_params.json           ← parametry zwycięskiej próby
├── second_best_params.json    ← 2. miejsce
├── third_best_params.json     ← 3. miejsce
├── study.pkl                  ← obiekt Optuna Study
├── trials.csv                 ← wszystkie próby z wynikami
├── tuning.log                 ← log procesu
└── summary.txt                ← podsumowanie tekstowe
```

### 7.6 Trening po tuningu

```bash
# Użyj najlepszych parametrów
python xgb_train.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --load_optuna_params xgb_data/XO001_xgb_optuna_XB001_0103/best_params.json

# Lub przez xgb_config.py
python xgb_config.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --optuna_dir xgb_data/XO001_xgb_optuna_XB001_0103 \
    --params_name best
```

---

## 8. Testowanie modelu (xgb_test.py)

### 8.1 Uruchomienie

**Auto-detect modelu i danych (używa najnowszego modelu i tego samego folderu prepared):**
```bash
python xgb_test.py
```

**Wskazanie konkretnego modelu:**
```bash
python xgb_test.py --model_dir xgb_models/X001_btc_xgb_model_XB001_0103
```

**Wskazanie danych testowych:**
```bash
python xgb_test.py \
    --model_dir xgb_models/X001_btc_xgb_model_XB001_0103 \
    --test_path xgb_prepared_data/XB001_prep_btc_full_1m
```

**Test na surowym CSV:**
```bash
python xgb_test.py \
    --model_dir xgb_models/X001_btc_xgb_model_XB001_0103 \
    --test_path data/btc_test.csv
```

### 8.2 Parametry CLI

| Parametr | Domyślny | Opis |
|---|---|---|
| `--model_dir` | najnowszy w `xgb_models/` | Folder z wytrenowanym modelem |
| `--test_path` | auto-detect | Folder prepared dir lub plik .csv |

### 8.3 Auto-detect danych testowych

Gdy `--test_path` nie jest podany, `test_model()` szuka danych w kolejności:

1. Jeśli `train_path` z konfiguracji modelu to prepared dir → użyj tego samego folderu (zawiera `test.csv`)
2. Szukaj w `xgb_prepared_data/` po `data_id` z konfiguracji modelu
3. Błąd z instrukcją co podać

### 8.4 Pliki testowanego modelu

```
xgb_models/X001_btc_xgb_model_XB001_0103/
├── xgb_model.json          ← model XGBoost (wagi drzew)
├── model_config.json       ← pełna konfiguracja modelu
├── feature_scaler.pkl      ← skaler cech (StandardScaler)
├── target_scaler.pkl       ← skaler targetu (RobustScaler), jeśli był użyty
├── feature_importance.json ← ważność cech (gain, weight, cover)
├── training.log            ← log treningu
└── summary.txt             ← podsumowanie treningu
```

### 8.5 Pliki wynikowe testu

Zapisywane w `xgb_data/XT001_test_results_{model_id}_{data_id}_{data}/`:

```
XT001_test_results_X001_XB001_0103/
├── results.csv             ← predykcje i błędy dla każdej próbki
├── metrics.json            ← metryki w formacie JSON
├── summary.txt             ← podsumowanie tekstowe
├── results.png             ← wykresy: log-return actual vs predicted
├── timeseries.png          ← pełna seria czasowa
└── feature_importance.png  ← top-30 ważności cech (jeśli dostępne)
```

`results.csv` zawiera kolumny:
```
actual_log    ← prawdziwy log-return
predicted_log ← przewidziany log-return
error         ← predicted_log - actual_log
```

---

## 9. Metryki i ich interpretacja

### 9.1 Przestrzeń metryk

**Ważne:** XGBoost przewiduje **log-return**, nie cenę absolutną. Wszystkie metryki liczone są w przestrzeni log-return, nie w dolarach. Wartości są małe (rząd 0.001–0.01).

### 9.2 Opis metryk

**MAE (Mean Absolute Error):**
```
MAE = mean(|y_pred - y_true|)
```
Średni bezwzględny błąd predykcji log-returnu. Przykład: MAE = 0.002 oznacza, że model myli się przeciętnie o 0.2 punktu procentowego w przewidywaniu ruchu ceny.

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(mean((y_pred - y_true)²))
```
Bardziej wrażliwy na duże błędy niż MAE. Wartość zawsze ≥ MAE.

**R² (współczynnik determinacji):**
```
R² = 1 - SS_res / SS_tot
```
Mierzy jaka frakcja wariancji targetu jest wyjaśniona przez model. Dla rynków finansowych nawet R² = 0.05–0.10 jest dobry wynik (rynki są bliskie losowym).

| R² | Interpretacja |
|---|---|
| > 0.10 | Dobry fit |
| 0.05–0.10 | Umiarkowany fit |
| < 0.05 | Słaby fit (ale uwaga: może być ok na rynkach) |

**Pseudo-Huber Loss (δ=1):**
```
PH(δ=1) = mean(δ² × (sqrt(1 + ((y_pred - y_true)/δ)²) - 1))
```
Metryka odpowiadająca funkcji straty `reg:pseudohubererror`. Odporna na outliery (duże błędy są karane mniej niż w MSE). Niższy = lepszy.

**Pseudo-Huber Loss (δ=10):**
Odpowiada funkcji straty `reg:pseudohubererror_large` (domyślnej). Przy δ=10 zachowuje się bardziej jak MSE dla typowych wartości log-return, ale jest odporny na ekstremalne outliery.

**Dir Acc (Directional Accuracy):**
```
Dir Acc = mean(sign(y_true) == sign(y_pred)) × 100%
```
Procent przypadków, w których model poprawnie przewidział **kierunek** ruchu ceny (wzrost/spadek). Obliczany bezpośrednio ze znaku log-returnu.

| Dir Acc | Interpretacja |
|---|---|
| > 55% | Dobry — realna przewaga |
| 50–55% | Niewielka przewaga |
| < 50% | Brak przewagi (gorszy niż losowy) |

### 9.3 Przykładowy output testu

```
METRICS SUMMARY
===============

LOG RETURN METRICS:
  MAE:                   0.001823
  RMSE:                  0.002641
  R²:                    0.0731
  Pseudo-Huber (δ=1):    0.001654
  Pseudo-Huber (δ=10):   0.002498

ERROR STATS:
  Mean:   0.000012
  Median: 0.000003
  Std:    0.002641

DIRECTIONAL:
  Accuracy: 54.37%

INTERPRETATION
==============
  ✓ Good fit: R² = 0.0731
  ○ Slight edge: 54.4%
```

### 9.4 Co oznaczają Error Stats

- **Mean** (błąd średni): wartość bliska 0 = model nie ma systematycznego biasu (nie zawsze przewiduje za wysoko ani za nisko)
- **Median** (błąd medianowy): bardziej odporny na outliery niż mean; powinien być bliski 0
- **Std** (odchylenie standardowe błędów): rozrzut błędów; im mniejszy tym lepiej

Gdy `Mean >> 0`: model systematycznie przeszacowuje log-returny.
Gdy `Mean << 0`: model systematycznie niedoszacowuje.

### 9.5 Feature Importance

Zapisywana w `feature_importance.json` z trzema typami ważności:

| Typ | Opis |
|---|---|
| `gain` | Średnia redukcja straty przez splits danej cechy — **najważniejsza** |
| `weight` | Liczba razy cecha użyta do podziału |
| `cover` | Średnia liczba próbek dotknięta przez split danej cechy |

`top20_gain` zawiera 20 najważniejszych cech wg gain — wystarczy do analizy.

Wykres `feature_importance.png` pokazuje top-30 cech wg gain po spłaszczeniu (np. `return_1_lag_1`, `rsi_mean_24`).

---

## 10. Przykłady użycia

### Scenariusz 1: Szybki start z surowym CSV

```bash
# Trening na surowym CSV z domyślną konfiguracją
python xgb_train.py --data_path data/btc_1m_2024.csv

# Test (dla CSV trzeba podać test_path ręcznie)
python xgb_test.py \
    --model_dir xgb_models/X001_btc_xgb_model_RAW_0103 \
    --test_path data/btc_1m_2024_test.csv
```

### Scenariusz 2: Pełny pipeline z prepared data

```bash
# Krok 1: Przygotuj dane
python data_preparation/prepare_data.py btc_1m_2024.csv \
    --feature_set full \
    --train_pct 70 \
    --test_pct 30 \
    --lookback 48 \
    --horizon 12

# Krok 2: Tuning
python xgb_model/xgb_hyperparameter_tuning.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --n_trials 150 \
    --feature_set xgb_momentum \
    --optimize_target val

# Krok 3: Trening z najlepszymi parametrami
python xgb_model/xgb_train.py \
    --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
    --load_optuna_params xgb_data/XO001_xgb_optuna_XB001_0103/best_params.json

# Krok 4: Test
python xgb_model/xgb_test.py \
    --model_dir xgb_models/X001_btc_xgb_model_XB001_0103
```

### Scenariusz 3: Eksperymenty z feature setami

```bash
# Porównanie feature setów (każdy tworzy osobny model z unikalnym ID)
for fs in xgb_momentum xgb_volatility technical full; do
    python xgb_train.py \
        --data_path xgb_prepared_data/XB001_prep_btc_full_1m \
        --feature_set $fs \
        --flattening_method hybrid
done
```

### Scenariusz 4: Użycie buy_pressure (dane Binance pełne)

```bash
# Dane muszą zawierać kolumnę taker_buy_base_asset_volume
python xgb_train.py \
    --data_path data/btc_1m_binance_full.csv \
    --feature_set technical_bp \
    --flattening_method hybrid
```

---

## 11. System identyfikatorów

| Prefix | Format | Przykład | Znaczenie |
|---|---|---|---|
| `X` | X001 | X001, X002 | Model ID (w nazwie folderu modelu) |
| `XB` | XB001 | XB001, XB002 | Data ID (prepared dataset — BTC) |
| `XE` | XE001 | XE001, XE002 | Data ID (prepared dataset — ETH) |
| `XO` | XO001 | XO001, XO002 | Optuna tuning ID |
| `XT` | XT001 | XT001, XT002 | Test run ID |

Identyfikatory są automatycznie inkrementowane na podstawie istniejących folderów.

Przykładowe nazwy folderów:
```
xgb_models/X001_btc_xgb_model_XB001_0103/    ← model X001 na danych XB001
xgb_data/XO001_xgb_optuna_XB001_0103/         ← tuning XO001
xgb_data/XT001_test_results_X001_XB001_0103/  ← test XT001
```

---

## 12. Pliki wyjściowe

### Model (xgb_models/X001_.../):

| Plik | Opis |
|---|---|
| `xgb_model.json` | Wagi modelu XGBoost — do wczytania przez `xgb.Booster().load_model()` |
| `model_config.json` | Pełna konfiguracja: features, lookback, horizon, hiperparametry, metryki val |
| `feature_scaler.pkl` | Skaler cech (StandardScaler) — do transformacji danych wejściowych |
| `target_scaler.pkl` | Skaler targetu (RobustScaler) — do inverse transform predykcji |
| `feature_importance.json` | Gain/weight/cover dla wszystkich flat features |
| `training.log` | Szczegółowy log procesu treningu |
| `summary.txt` | Czytelne podsumowanie: hiperparametry + metryki train/val |

### Tuning (xgb_data/XO001_.../):

| Plik | Opis |
|---|---|
| `best_params.json` | Najlepsze parametry — bezpośredni input do `--load_optuna_params` |
| `second_best_params.json` | 2. najlepszy zestaw parametrów |
| `third_best_params.json` | 3. najlepszy zestaw parametrów |
| `study.pkl` | Obiekt Optuna Study — do dalszej analizy w Python |
| `trials.csv` | Wszystkie próby z wartościami metryk i parametrów |
| `tuning.log` | Log procesu tuningu |
| `summary.txt` | Podsumowanie top-3 prób |

### Test (xgb_data/XT001_.../):

| Plik | Opis |
|---|---|
| `results.csv` | `actual_log`, `predicted_log`, `error` dla każdej próbki |
| `metrics.json` | MAE, RMSE, R², Pseudo-Huber (δ=1), Pseudo-Huber (δ=10), Dir Acc |
| `summary.txt` | Podsumowanie tekstowe |
| `results.png` | Actual vs predicted log-return (pierwsze 500 próbek + scatter) |
| `timeseries.png` | Pełna seria log-return: actual (niebieski) vs predicted (czerwony) |
| `feature_importance.png` | Wykres słupkowy top-30 cech wg gain |
