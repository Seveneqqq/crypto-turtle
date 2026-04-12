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
9. [Ważenie próbek i funkcje celu](#9-ważenie-próbek-i-funkcje-celu)
10. [Próg sygnałowy (Signal Threshold)](#10-próg-sygnałowy-signal-threshold)
11. [Metryki i ich interpretacja](#11-metryki-i-ich-interpretacja)
12. [Przykłady użycia](#12-przykłady-użycia)
13. [System identyfikatorów](#13-system-identyfikatorów)
14. [Pliki wyjściowe](#14-pliki-wyjściowe)

---

## 1. Przegląd systemu

Model XGBoost przewiduje **log-return** (logarytmiczną stopę zwrotu) kryptowaluty dla zadanego horyzontu czasowego. Jest to model regresji tablicowej (flat/tabular), który:

- przyjmuje dane OHLCV w formacie 1-minutowym (lub innych interwałach),
- generuje cechy techniczne (feature engineering),
- „spłaszcza" sekwencję historyczną (`lookback` świec) do pojedynczego wektora cech (`flattening`),
- przewiduje `log(close[t+horizon] / close[t])` — czyli oczekiwany procentowy ruch ceny.

**Target: log-return, nie cena absolutna.** Wartość `+0.001` oznacza oczekiwany wzrost o ~0.1%, wartość `-0.002` oznacza oczekiwany spadek o ~0.2%.

---

## 2. Architektura i struktura plików

```
ml/
├── feature_engineering.py              ← UNIVERSAL — generuje cechy OHLCV
│
├── xgb_model/
│   ├── xgb_config.py                   ← Konfiguracja (hiperparametry, ścieżki, cele)
│   ├── xgb_train.py                    ← Trening modelu
│   ├── xgb_test.py                     ← Testowanie modelu
│   ├── xgb_hyperparameter_tuning.py    ← Optuna tuning
│   ├── feature_engineering.py          ← Bridge do universal FE (via importlib)
│   │
│   ├── common/
│   │   ├── data_utils.py               ← Ładowanie danych, scalery targetu
│   │   ├── flattening.py               ← Metody spłaszczania sekwencji
│   │   ├── metrics.py                  ← Obliczanie metryk (log-return space)
│   │   ├── training_utils.py           ← Pętla treningowa, zapis modelu
│   │   ├── sample_weighting.py         ← Wagi próbek, custom objective, signal threshold
│   │   ├── log_utils.py                ← Funkcje logowania (train/test/tuning)
│   │   ├── id_utils.py                 ← System identyfikatorów (X001, XO001, XT001)
│   │   └── visualization.py            ← Wykresy wyników i feature importance
│   │
│   ├── (../shared/models/)xgb_models/      ← Zapisane modele (X001_btc_xgb_model_...)
│   ├── xgb_data/                           ← Wyniki tuningu i testów
│   └── (../shared/models/)prepared_data/   ← Dane przygotowane przez data_preparation/
│
└── (../shared/models/)data_preparation/
    └── prepare_data.py                 ← Przygotowanie danych (CSV → prepared dir)
```

---

## 3. Przepływ danych

### 3.1 Schemat ogólny

```
CSV / Prepared Dir
       │
       ▼
feature_engineering.prepare_features()   ← generuje cechy techniczne
       │
       ▼
flatten_features()                        ← sekwencja lookback → wektor
  ['lag' | 'agg' | 'hybrid' | 'signal' | 'none']
       │
       ▼
create_log_return_target()               ← y = log(close[t+h] / close[t])
       │
       ▼
StandardScaler (cechy) + TargetScaler (target, opcjonalnie)
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
prepared_data/
└── XB001_prep_btc_full_1m/
    ├── train.csv
    ├── test.csv
    ├── scaler_features.pkl
    └── metadata.json
```

**Tryb 2 — Raw CSV** (on-the-fly):

Surowy plik CSV z kolumnami OHLCV. Feature engineering i skalowanie wykonywane są automatycznie podczas ładowania.

### 3.3 Flattening — spłaszczanie sekwencji

XGBoost nie obsługuje danych sekwencyjnych (jak LSTM). Sekwencja `lookback` świec jest przekształcana w pojedynczy wiersz cech jedną z pięciu metod:

**`lag`** — proste opóźnienia:
Dla każdej cechy tworzy kolumny `feature_lag_1`, `feature_lag_2`, ..., `feature_lag_{lookback}`.
Wynikowy wymiar: `n_features × lookback`. Przy 15 cechach i lookback=48 → 720 kolumn.
Zachowuje pełną informację o kolejności, ale tworzy dużo cech.

**`agg`** — agregacje:
Dla każdej cechy tworzy 5 statystyk z okna `[t-lookback, t-1]`: mean, std, min, max, last.
Wynikowy wymiar: `n_features × 5`. Przy 15 cechach → 75 kolumn.
Szybki, kompaktowy, dobry punkt startowy.

**`hybrid`** (domyślny) — kombinacja strategicznych opóźnień i agregacji wielookienkowych:
- opóźnienia dla punktów: `[1, 2, 3, 5, 10, 24]`
- okna agregacji mean i std: `[6, 12, 24, 48]`
- różnice pierwszego rzędu: `diff_1`, `diff_5`
Najlepszy kompromis między szczegółowością a rozmiarem.

**`signal`** — cechy zorientowane na sygnał/momentum:
Nie agreguje statystyk — generuje cechy bezpośrednio kodujące kierunek i trend:
- `trend_short_long` — mean(okno_krótkie) - mean(okno_długie) dla par okien [3,6,12] × [24,48]
- `last_vs_mean` — ostatnia wartość - mean(lookback)
- `sign_frac` — frakcja wartości > 0 w oknie (szczególnie informacyjna dla return_*)
- `diff_{lag}` — różnice przy opóźnieniach [1, 3, 5, 12]
- `accel_1_5` — diff(1) - diff(5): krótkoterminowe przyspieszenie vs średnioterminowe

**`none`** — bez kontekstu czasowego (baseline):
Przekazuje tylko bieżące wartości cech (t-1) bez żadnej historii.
Przydatne jako punkt odniesienia — mierzy ile wnosi kontekst historyczny.

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
| `technical` | ~15 | Bez time features i vol_regime — bezpieczny dla krótkich okien; **domyślny** |
| `technical_bp` | ~16 | `technical` + buy_pressure |
| `minimal` | ~6 | return_1, return_5, close_sma_14, rsi, volatility_14, relative_volume |

#### Zestawy dedykowane XGBoost

| Zestaw | Cechy | Kiedy używać |
|---|---|---|
| `xgb_momentum` | ~11–12 | Trend/momentum z vol_regime; standardowy wybór |
| `xgb_volatility` | ~9 | Fokus na reżim zmienności i rozstępy cen |
| `xgb_returns` | 4 | Tylko returny — maksymalna przenośność danych |

#### Aliasy legacy

| Alias | Mapuje do |
|---|---|
| `momentum` | `xgb_momentum` |
| `volatility_focused` | `xgb_volatility` |

---

## 5. Konfiguracja modelu (xgb_config.py)

### 5.1 Parametry danych

```python
DEFAULT_LOOKBACK    = 48        # liczba świec historycznych (sekwencja wejściowa)
DEFAULT_HORIZON     = 12        # horyzont predykcji (świece w przyszłość)
DEFAULT_FLATTENING  = 'hybrid'  # metoda spłaszczania: 'lag' | 'agg' | 'hybrid' | 'signal' | 'none'
DEFAULT_FEATURE_SET = 'technical'  # zestaw cech
```

`lookback=48` przy danych 1-minutowych = ostatnie 48 minut historii.
`horizon=12` = przewidujemy cenę za 12 minut.

### 5.2 Funkcja straty (OBJECTIVE)

```python
OBJECTIVE    = 'reg:pseudohubererror_large'
HUBER_SLOPE  = 10.0
```

Dostępne funkcje straty:

| Wartość | Opis | Kiedy używać |
|---|---|---|
| `reg:pseudohubererror_large` | Pseudo-Huber δ=10 (domyślna) | Odporność na outliery, bliżej MSE — polecana |
| `reg:pseudohubererror` | Pseudo-Huber δ=1 | Silniejsza odporność, bliżej MAE |
| `reg:squarederror` | MSE — średnia kwadratów błędów | Wrażliwa na outliery; dobra gdy dane czyste |
| `reg:absoluteerror` | MAE — medianowa regresja | Tendencja do „spłaszczania" predykcji do 0 |

**Niezgodne** (nie używać z log-return targetem):
- `reg:squaredlogerror` — wymaga etykiet > -1 (skalowane log-returny mogą być ujemne)
- `rank:ndcg` — wymaga nieujemnych etykiet całkowitych

`reg:pseudohubererror_large` to wirtualna nazwa obsługiwana wewnętrznie — XGBoost otrzymuje `reg:pseudohubererror` z parametrem `huber_slope=10`.

### 5.3 Skalowanie targetu

```python
SCALE_TARGET       = True
TARGET_SCALER_TYPE = 'robust'   # 'standard' | 'robust'
```

Log-returny mają małe wartości (rząd 0.001), co powoduje problemy z gradientami XGBoost. Skalowanie normalizuje target do ~N(0,1) przed treningiem i odwraca po predykcji.

| Typ | Opis | Kiedy używać |
|---|---|---|
| `standard` | StandardScaler — mean=0, std=1 | Dane bez ekstremalnych outlierów |
| `robust` | RobustScaler — mediana i IQR | Dane z flash crashami / pump&dump (zalecany) |

Skalowanie jest **odwracalne** — metryki testowe zawsze raportowane są w oryginalnej przestrzeni log-return.

> **Ważne:** gdy aktywna jest custom objective (`USE_CUSTOM_OBJECTIVE=True`), `base_score` jest automatycznie ustawiane na `mean(y_train)` aby uniknąć problemów z gradientami przy niskiej skali log-returnów.

### 5.4 Winsoryzacja targetu

```python
WINSORIZE_TARGET       = False
WINSORIZE_PERCENTILE   = 99.0
```

Obcina skrajne wartości log-return (np. flash crashe). Granice wyznaczane **tylko** na danych treningowych (brak data leakage). Val/test obcinane tymi samymi granicami.

Przykład z `WINSORIZE_PERCENTILE=99.0`: obcina zewnętrzne 1% rozkładu (dolne 0.5% i górne 0.5%).

Kiedy włączyć: gdy dane zawierają ekstremalne zdarzenia rynkowe, które są niereprezentatywne dla normalnego zachowania.

### 5.5 Cel optymalizacji Optuna

```python
OPTUNA_OPTIMIZE_TARGET = 'val'         # 'val' | 'train'
OPTUNA_OPTIMIZE_METRIC = 't_dir_acc'   # patrz lista poniżej
```

`OPTUNA_OPTIMIZE_TARGET` — który zbiór Optuna minimalizuje:
- `'val'` (domyślne) — minimalizuje stratę na zbiorze walidacyjnym; zalecane w produkcji
- `'train'` — minimalizuje na treningowym; przydatne do diagnozy underfittingu

`OPTUNA_OPTIMIZE_METRIC` — która metryka jest optymalizowana:

| Wartość | Opis |
|---|---|
| `mae` | Mean Absolute Error (minimalizuj) |
| `rmse` | Root Mean Squared Error (minimalizuj) |
| `pseudo_huber` | Pseudo-Huber Loss δ=1 (minimalizuj) |
| `dir_acc` | Directional Accuracy (maksymalizuj — Optuna neguje) |
| `t_mae` | Thresholded MAE na próbkach \|y\| ≥ threshold (minimalizuj) |
| `t_rmse` | Thresholded RMSE (minimalizuj) |
| `t_pseudo_huber` | Thresholded Pseudo-Huber (minimalizuj) |
| `t_dir_acc` | Thresholded Dir Acc — **domyślna** (maksymalizuj — Optuna neguje) |
| `t_r2` | Thresholded R² (maksymalizuj — Optuna neguje) |

Metryki z prefiksem `t_` wymagają aktywnego `SIGNAL_THRESHOLD_ENABLED=True`. Optuna zawsze **minimalizuje** — metryki wyższy=lepszy (dir_acc, t_dir_acc, t_r2) są automatycznie negowane przed przekazaniem do Optuny i odwracane z powrotem w logach i wynikach.

### 5.6 Hiperparametry domyślne XGBoost

```python
DEFAULT_PARAMS = {
    'n_estimators':      3222,   # liczba drzew
    'learning_rate':     0.05,   # krok uczenia
    'max_depth':         6,      # głębokość drzewa
    'subsample':         0.8,    # próbkowanie wierszy per drzewo
    'colsample_bytree':  0.8,    # próbkowanie kolumn per drzewo
    'colsample_bylevel': 0.8,    # próbkowanie kolumn per poziom
    'min_child_weight':  3,      # min suma wag w liściu
    'gamma':             0.0,    # min redukcja straty do podziału
    'reg_alpha':         0.0,    # regularyzacja L1
    'reg_lambda':        1,      # regularyzacja L2
    'tree_method':       'hist', # algorytm budowy drzew (szybki)
}
EARLY_STOPPING_ROUNDS = 1000    # zatrzymanie gdy brak poprawy przez N rund
USE_BEST_ITERATION    = False   # False=ostatnia iteracja, True=best_iteration
```

### 5.7 USE_BEST_ITERATION

| Wartość | Zachowanie |
|---|---|
| `False` (domyślne) | Zapisuje model z ostatniej iteracji (wszystkie `n_estimators` drzew) |
| `True` | Zapisuje model z `best_iteration` (punkt early stopping) |

---

## 6. Trening modelu (xgb_train.py)

### 6.1 Uruchomienie z linii poleceń

**Podstawowe (z przygotowanymi danymi):**
```bash
python xgb_train.py --data_path prepared_data/XB001_prep_btc_full_1m
```

**Z surowym CSV:**
```bash
python xgb_train.py --data_path data/btc_1m.csv
```

**Z parametrami Optuna:**
```bash
python xgb_train.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
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
| `--flattening_method` | `hybrid` | Metoda spłaszczania: `lag`, `agg`, `hybrid`, `signal`, `none` |
| `--feature_set` | `technical` | Zestaw cech (patrz sekcja 4.2) |

**Hiperparametry XGBoost:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--n_estimators` | 3222 | Liczba drzew |
| `--max_depth` | 6 | Maksymalna głębokość drzewa (3–10) |
| `--learning_rate` | 0.05 | Krok uczenia (0.001–0.5) |
| `--subsample` | 0.8 | Frakcja próbkowania wierszy (0.6–1.0) |
| `--colsample_bytree` | 0.8 | Frakcja kolumn per drzewo (0.6–1.0) |
| `--colsample_bylevel` | 0.8 | Frakcja kolumn per poziom (0.6–1.0) |
| `--min_child_weight` | 3 | Min suma wag w liściu (1–10) |
| `--gamma` | 0.0 | Min redukcja straty do podziału węzła (0–2) |
| `--reg_alpha` | 0.0 | Regularyzacja L1 (0–5) |
| `--reg_lambda` | 1 | Regularyzacja L2 (0.01–5) |
| `--objective` | z config | Funkcja straty (`reg:pseudohubererror`, `reg:pseudohubererror_large`, `reg:squarederror`, `reg:absoluteerror`) |
| `--early_stopping_rounds` | 1000 | Patience early stopping |

**Target:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--scale_target` | True (z config) | Włącz skalowanie targetu |
| `--no_scale_target` | — | Wyłącz skalowanie targetu |
| `--target_scaler_type` | `robust` | Typ skalera: `standard`, `robust` |
| `--winsorize_target` | False | Włącz winsoryzację targetu |
| `--winsorize_percentile` | 99.0 | Percentyl winsoryzacji (90–99.9) |

**Ważenie próbek (podejście A — DMatrix weight):**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--sample_weight_mode` | None | Schemat ważenia: `abs`, `abs_clip`, `rank`, `percentile`; None = jednolite |
| `--sample_weight_alpha` | 1.0 | Wykładnik dla trybów `abs` / `abs_clip` |
| `--sample_weight_min_w` | 0.1 | Floor frakcja dla `abs_clip` |
| `--sample_weight_percentile` | 70.0 | Percentyl progu dla trybu `percentile` |
| `--sample_weight_low_w` | 0.1 | Waga poniżej progu w trybie `percentile` |

**Kara za błędny kierunek (podejście A — misdir):**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--reweight_misdir` | False | Włącz karę za błędny kierunek (drugi trening) |
| `--misdir_approach_a_variant` | `boost` | Wariant: `retrain` = restart od zera, `boost` = kontynuacja od checkpointu |
| `--misdir_penalty` | 2.0 | Mnożnik wag dla próbek z błędnym kierunkiem |
| `--misdir_min_quantile` | 0.7 | Minimalny kwantyl \|y\| kwalifikujący do kary |

**Custom objective (podejście B):**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--use_custom_objective` | True (z config) | Włącz custom objective |
| `--custom_obj_type` | `weighted_huber` | Typ: `weighted_huber`, `dead_zone` |
| `--custom_obj_sample_weights` | True | Składnik ważenia próbek w custom obj |
| `--no_custom_obj_sample_weights` | — | Wyłącz ważenie próbek w custom obj |
| `--custom_obj_misdir_penalty` | True | Składnik kary misdir w custom obj |
| `--no_custom_obj_misdir_penalty` | — | Wyłącz karę misdir w custom obj |
| `--custom_obj_huber_delta` | `auto` | Delta Pseudo-Hubera: `auto` lub wartość float |
| `--dead_zone_weight` | 0.1 | Mnożnik gradientu w dead zone (tryb `dead_zone`) |

**Próg sygnałowy:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--signal_threshold_enabled` | True (z config) | Włącz próg sygnałowy |
| `--signal_threshold_mode` | `fixed` | Tryb: `fixed` = stała wartość, `empirical` = percentyl \|y_train\| |
| `--signal_threshold_value` | 0.001 | Wartość progu (tryb `fixed`) |
| `--signal_threshold_percentile` | 50.0 | Percentyl \|y_train\| (tryb `empirical`) |
| `--use_threshold_penalties` | True (z config) | Próg wpływa na kary w objective/wagach |
| `--threshold_smooth_transition` | False | Gładkie przejście sigmoid zamiast twardego odcięcia |
| `--threshold_smooth_width` | 2.0 | Szerokość sigmoidy (1=ostry, 5=łagodny) |

**Inne:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--seed` | 42 | Ziarno losowości |

### 6.3 Priorytet parametrów

Gdy podano `--load_optuna_params`, parametry z pliku JSON są ładowane jako baza. Każdy parametr CLI podany **dodatkowo** nadpisuje wartość z Optuna. Nienadpisane parametry biorą wartości z Optuna, a jeśli brak — z `XGBConfig`.

### 6.4 Podział train/val

Podział jest **chronologiczny** (80%/20%) — brak shuffle, brak data leakage:
- Train: pierwsze 80% sekwencji
- Val: ostatnie 20% sekwencji

Early stopping monitoruje stratę na zbiorze walidacyjnym.

### 6.5 API Python

```python
from xgb_train import train

model, feature_scaler, config = train(
    data_path='prepared_data/XB001_prep_btc_full_1m',
    lookback=48,
    horizon=12,
    flattening_method='hybrid',
    feature_set='technical',
    n_estimators=3000,
    learning_rate=0.01,
    scale_target=True,
    target_scaler_type='robust',
)

print(config['val_metrics'])   # {'mae': ..., 'r2': ..., 'dir_acc': ..., 't_dir_acc': ...}
print(config['output_dir'])    # ścieżka do zapisanego modelu
```

---

## 7. Tuning hiperparametrów (xgb_hyperparameter_tuning.py)

### 7.1 Uruchomienie

```bash
python xgb_hyperparameter_tuning.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --n_trials 100 \
    --feature_set technical \
    --optimize_target val \
    --optimize_metric t_dir_acc
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
| `--feature_set` | `technical` | Zestaw cech (stały podczas tuningu) |
| `--lookback` | 48 | Lookback (stały, jeśli nie `--tune_lookback`) |
| `--horizon` | 12 | Horizon (stały, jeśli nie `--tune_horizon`) |
| `--flattening_method` | None | Fiksuje metodę flatteningu (domyślnie tunowane) |
| `--tune_lookback` | False | Tuninguj lookback jako hiperparametr |
| `--tune_horizon` | False | Tuninguj horizon jako hiperparametr |
| `--scale_target` | True | Włącz skalowanie targetu |
| `--no_scale_target` | — | Wyłącz skalowanie targetu |
| `--target_scaler_type` | `robust` | Typ skalera |
| `--optimize_target` | `val` | Co minimalizuje Optuna: `val` lub `train` |
| `--optimize_metric` | `t_dir_acc` | Metryka do optymalizacji (patrz sekcja 5.5) |
| `--signal_threshold_mode` | `fixed` | Tryb progu sygnałowego |
| `--signal_threshold_value` | 0.001 | Wartość progu (tryb `fixed`) |
| `--use_threshold_penalties` | True (z config) | Włącz kary progowe |
| `--custom_obj_type` | `weighted_huber` | Typ custom objective |
| `--dead_zone_weight` | 0.1 | Waga gradientu w dead zone |

> **Uwaga:** `REWEIGHT_MISDIR` nie jest stosowany podczas tuningu (podwajałby czas każdego trialu). Wagi DMatrix (podejście A) oraz custom objective (podejście B) są stosowane.

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
| `n_estimators` | int | [400, 1000] |
| `reg_alpha` | float | [0.0, 5.0] |
| `reg_lambda` | float | [0.01, 5.0] |
| `flattening_method` | categorical | `lag`, `agg`, `hybrid`, `signal`, `none` |
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
├── best_params.json           ← parametry zwycięskiej próby (+ horizon, lookback, flattening_method)
├── second_best_params.json    ← 2. miejsce
├── third_best_params.json     ← 3. miejsce
├── study.pkl                  ← obiekt Optuna Study
├── trials.csv                 ← wszystkie próby z wynikami
├── tuning.log                 ← log procesu
├── summary.txt                ← podsumowanie top-3 prób z informacją o metryce i negacji
├── history.png                ← wykres historii optymalizacji (Optuna)
├── importances.png            ← ważność hiperparametrów (Optuna)
├── distribution.png           ← rozkład wartości prób
└── convergence.png            ← konwergencja (kumulatywne minimum)
```

`_metadata` w plikach `best_params.json` zawiera:
- `best_value` — wartość przekazana do Optuny (może być zanegowana)
- `best_value_display` — wartość do wyświetlenia (po odwróceniu negacji)
- `best_value_meaning` — opis metryki z adnotacją `(negated — higher=better)` gdy dotyczy
- `optimize_metric` — nazwa optymalizowanej metryki

### 7.6 Trening po tuningu

```bash
# Użyj najlepszych parametrów
python xgb_train.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --load_optuna_params xgb_data/XO001_xgb_optuna_XB001_0103/best_params.json

# Lub przez xgb_config.py
python xgb_config.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --optuna_dir xgb_data/XO001_xgb_optuna_XB001_0103 \
    --params_name best
```

---

## 8. Testowanie modelu (xgb_test.py)

### 8.1 Uruchomienie

**Auto-detect modelu i danych (używa najnowszego modelu):**
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
    --test_path prepared_data/XB001_prep_btc_full_1m
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

1. Jeśli `train_path` z konfiguracji modelu to prepared dir z `test.csv` → użyj tego samego folderu
2. Szukaj w `xgb_model/prepared_data/` po `data_id` z konfiguracji modelu
3. Szukaj w `data_preparation/prepared_data/` po `data_id`
4. Szukaj w `rf_model/rf_prepared_data/` po `data_id`
5. Błąd z instrukcją co podać

### 8.4 Obsługa progu sygnałowego podczas testu

Próg sygnałowy jest wczytywany z konfiguracji modelu (`signal_threshold_resolved`). Jeśli model był trenowany z `SIGNAL_THRESHOLD_ENABLED=True` i próg jest zapisany, metryki `t_*` są automatycznie obliczane.

**Fallback dla starszych modeli:** jeśli model nie ma zapisanego progu, a konfiguracja wskazuje na tryb `empirical`, system automatycznie przełącza się na tryb `fixed` z oryginalną wartością progową (unikając data leakage przez obliczanie percentyla z danych testowych).

### 8.5 Pliki modelu

```
xgb_models/X001_btc_xgb_model_XB001_0103/
├── xgb_model.json          ← model XGBoost (wagi drzew)
├── model_config.json       ← pełna konfiguracja modelu
├── feature_scaler.pkl      ← skaler cech (StandardScaler)
├── target_scaler.pkl       ← skaler targetu (RobustScaler), jeśli był użyty
├── feature_importance.json ← ważność cech (gain, weight, cover, top20_gain)
├── training.log            ← log treningu
└── summary.txt             ← podsumowanie treningu z metrykami t_* jeśli aktywne
```

### 8.6 Pliki wynikowe testu

Zapisywane w `xgb_data/XT001_test_results_{model_id}_{data_id}_{data}/`:

```
XT001_test_results_X001_XB001_0103/
├── results.csv                     ← predykcje i błędy dla każdej próbki
├── metrics.json                    ← wszystkie metryki (w tym t_* jeśli aktywne)
├── summary.txt                     ← podsumowanie z metrykami t_* jeśli aktywne
├── log_returns_timeseries.png      ← pełna seria log-return: actual vs predicted
├── log_returns_scatter.png         ← scatter: actual vs predicted
├── error_distribution.png         ← histogram błędów
├── combined_results.png            ← 2×2: timeseries + scatter + histogram + rolling dir acc
└── feature_importance.png         ← top-30 ważności cech (jeśli dostępne)
```

`results.csv` zawiera kolumny:
```
actual_log    ← prawdziwy log-return
predicted_log ← przewidziany log-return
error         ← predicted_log - actual_log
```

---

## 9. Ważenie próbek i funkcje celu

System oferuje dwa podejścia do skupienia modelu na ważnych ruchach rynkowych. Aktywacja obu jednocześnie dla tego samego mechanizmu jest blokowana przez `validate_approach_config()`.

### 9.1 Podejście A — Natywne wagi DMatrix

Wagi per-próbka obliczane z `|log_return|` i przekazywane do `xgb.DMatrix(weight=...)`. XGBoost mnoży hesjan każdej próbki przez jej wagę.

```python
SAMPLE_WEIGHT_MODE       = None    # None = jednolite; 'abs' | 'abs_clip' | 'rank' | 'percentile'
SAMPLE_WEIGHT_ALPHA      = 1.0     # wykładnik dla 'abs' / 'abs_clip'
SAMPLE_WEIGHT_MIN_W      = 0.1     # floor frakcja dla 'abs_clip'
SAMPLE_WEIGHT_PERCENTILE = 70.0    # percentyl progu dla 'percentile'
SAMPLE_WEIGHT_LOW_W      = 0.1     # waga poniżej progu w 'percentile'
```

| Tryb | Wzór | Opis |
|---|---|---|
| `None` | w = 1.0 | Jednolite wagi (domyślne XGBoost) |
| `abs` | w = \|y\|^alpha | Waga proporcjonalna do wielkości ruchu |
| `abs_clip` | w = clip(\|y\|^alpha, floor, ∞) | `abs` z minimalną wagą |
| `rank` | w = rank(\|y\|) / N | Skalowanie rangowe — niezależne od skali |
| `percentile` | w = 1.0 jeśli \|y\| ≥ pct, inaczej low_w | Binarne: duże ruchy vs. reszta |

#### Kara za błędny kierunek (misdir penalty, podejście A)

Po pierwszym treningu identyfikuje próbki gdzie model przewidział zły kierunek dla dużych ruchów (`|y| ≥ MISDIR_MIN_QUANTILE`) i podnosi ich wagi, po czym trenuje ponownie.

```python
REWEIGHT_MISDIR           = False   # domyślnie wyłączone
MISDIR_APPROACH_A_VARIANT = 'boost' # 'retrain' = restart od zera, 'boost' = kontynuacja
MISDIR_PENALTY            = 2.0     # mnożnik dla próbek z błędnym kierunkiem
MISDIR_MIN_QUANTILE       = 0.7     # próg |y| kwalifikujący do kary (top 30% ruchów)
```

### 9.2 Podejście B — Custom Objective

Zastępuje standardowy `params['objective']`. W odróżnieniu od podejścia A, modyfikacja gradientu jest aktywna **w każdej iteracji boostingu** — model adaptuje się ciągle.

```python
USE_CUSTOM_OBJECTIVE      = True             # master switch
CUSTOM_OBJ_TYPE           = 'weighted_huber' # 'weighted_huber' | 'dead_zone'
CUSTOM_OBJ_SAMPLE_WEIGHTS = True             # składnik wag
CUSTOM_OBJ_MISDIR_PENALTY = True             # składnik kary misdir
CUSTOM_OBJ_HUBER_DELTA    = 'auto'           # delta Pseudo-Hubera ('auto' lub float)
```

**`weighted_huber`** — Pseudo-Huber z ważeniem i karą:
Gradient skalowany przez kombinację: wagi z |y_raw|, kary za błędny kierunek, overlay progu sygnałowego. Każdy składnik jest niezależny i może być włączony/wyłączony osobno.

**`dead_zone`** — Pseudo-Huber z dead-zone poniżej progu:
Tłumi gradienty dla `|y| < threshold` (sygnał uważany za szum). Powyżej progu: pełny gradient + opcjonalna kara misdir. Wymaga aktywnego `SIGNAL_THRESHOLD_ENABLED=True`.

**Tryb `auto` dla `CUSTOM_OBJ_HUBER_DELTA`:**
- gdy `target_scaler` jest aktywny → delta = 1.0 (dane już w ~N(0,1))
- gdy brak skalera → delta = std(y_train) automatycznie dostosowuje się do skali danych

### 9.3 Zabezpieczenie przed konfliktami

```python
ALLOW_COMBINED_APPROACHES = False  # blokuje jednoczesne użycie A+B dla tego samego mechanizmu
```

Gdy `False` (domyślne): aktywacja `SAMPLE_WEIGHT_MODE` i `CUSTOM_OBJ_SAMPLE_WEIGHTS=True` jednocześnie → `ValueError`. Analogicznie dla `REWEIGHT_MISDIR` i `CUSTOM_OBJ_MISDIR_PENALTY`. Ustaw `True` tylko dla eksperymentów.

---

## 10. Próg sygnałowy (Signal Threshold)

Centralny próg sygnałowy definiuje granicę między „szumem" (małe ruchy bez realnego zysku po opłatach) a „sygnałem" (ruchy warte handlu).

### 10.1 Konfiguracja

```python
SIGNAL_THRESHOLD_ENABLED    = True    # master switch
SIGNAL_THRESHOLD_MODE       = 'fixed' # 'fixed' | 'empirical'
SIGNAL_THRESHOLD_VALUE      = 0.001   # wartość progu (tryb 'fixed')
SIGNAL_THRESHOLD_PERCENTILE = 50.0    # percentyl |y_train| (tryb 'empirical')
USE_THRESHOLD_PENALTIES     = True    # próg wpływa na kary (nie tylko metryki)
THRESHOLD_SMOOTH_TRANSITION = False   # sigmoid zamiast twardego odcięcia
THRESHOLD_SMOOTH_WIDTH      = 2.0     # szerokość sigmoidy
```

**Tryb `fixed`:** próg = stała wartość `SIGNAL_THRESHOLD_VALUE` (np. 0.001 = 0.1% ruchu).

**Tryb `empirical`:** próg = percentyl `SIGNAL_THRESHOLD_PERCENTILE` z rozkładu `|y_train|`. Np. 50.0 = mediana |y_train| — próg dynamicznie dostosowuje się do danych.

### 10.2 Zastosowanie progu

**Tylko metryki** (`USE_THRESHOLD_PENALTIES=False`): próg używany wyłącznie do obliczania metryk progowych `t_*` — nie wpływa na trening.

**Metryki + kary** (`USE_THRESHOLD_PENALTIES=True`):
- W DMatrix weights (podejście A): wagi próbek poniżej progu mnożone przez `DEAD_ZONE_WEIGHT`
- W custom objective (podejście B): gradient próbek poniżej progu skalowany przez `DEAD_ZONE_WEIGHT`

> **Ważne:** gdy aktywna jest custom objective, overlay progowy w wagach DMatrix jest automatycznie pomijany, żeby uniknąć podwójnego karania (DMatrix weight × mnożnik gradient = np. 0.1 × 0.1 = 0.01).

### 10.3 Próg sygnałowy a metryki

Gdy `SIGNAL_THRESHOLD_ENABLED=True`, metryki są obliczane w dwóch przestrzeniach:
1. **Standardowe** — na wszystkich próbkach
2. **Progowe (t_*)** — tylko na próbkach gdzie `|y_true| ≥ threshold`

Metryki progowe informują o jakości predykcji **tam gdzie to ma znaczenie** — na ruchach wystarczająco dużych by zarabiać po opłatach.

---

## 11. Metryki i ich interpretacja

### 11.1 Przestrzeń metryk

**Ważne:** XGBoost przewiduje **log-return**, nie cenę absolutną. Wszystkie metryki liczone są w przestrzeni log-return, nie w dolarach. Wartości są małe (rząd 0.001–0.01).

### 11.2 Metryki standardowe

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
Mierzy jaka frakcja wariancji targetu jest wyjaśniona przez model. Dla rynków finansowych nawet R² = 0.05–0.10 jest dobry wynik.

| R² | Interpretacja |
|---|---|
| > 0.10 | Dobry fit |
| 0.05–0.10 | Umiarkowany fit |
| < 0.05 | Słaby fit |

**Pseudo-Huber Loss (δ=1):**
Metryka odpowiadająca funkcji straty `reg:pseudohubererror`. Odporna na outliery. Niższy = lepszy.

**Pseudo-Huber Loss (δ=10):**
Odpowiada funkcji straty `reg:pseudohubererror_large` (domyślnej). Przy δ=10 zachowuje się bardziej jak MSE dla typowych wartości log-return, ale jest odporny na ekstremalne outliery.

**Dir Acc (Directional Accuracy):**
```
Dir Acc = mean(sign(y_true) == sign(y_pred)) × 100%
```
Procent przypadków, w których model poprawnie przewidział **kierunek** ruchu ceny (wzrost/spadek).

| Dir Acc | Interpretacja |
|---|---|
| > 55% | Dobry — realna przewaga |
| 50–55% | Niewielka przewaga |
| < 50% | Brak przewagi |

### 11.3 Metryki progowe (t_*)

Obliczane wyłącznie na próbkach spełniających `|y_true| ≥ signal_threshold`. Aktywne gdy `SIGNAL_THRESHOLD_ENABLED=True` **i** liczba kwalifikujących próbek ≥ 10.

| Metryka | Opis |
|---|---|
| `t_mae` | MAE na próbkach z dużymi ruchami |
| `t_rmse` | RMSE na próbkach z dużymi ruchami |
| `t_r2` | R² na próbkach z dużymi ruchami |
| `t_pseudo_huber` | Pseudo-Huber δ=1 na próbkach z dużymi ruchami |
| `t_pseudo_huber_large` | Pseudo-Huber δ=10 na próbkach z dużymi ruchami |
| `t_dir_acc` | Directional Accuracy na próbkach z dużymi ruchami — **najważniejsza** |
| `t_n_samples` | Liczba próbek powyżej progu |
| `t_pct_samples` | Procent próbek powyżej progu |
| `t_threshold` | Wartość użytego progu |

`t_dir_acc` jest domyślną metryką optymalizowaną przez Optunę — mierzy zdolność modelu do przewidywania kierunku ruchów **które mają znaczenie handlowe**.

### 11.4 Przykładowy output testu

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

THRESHOLDED METRICS (|y| ≥ 0.001000, n=18432 / 61.5%):
  MAE:                   0.002901
  RMSE:                  0.003812
  R²:                    0.0842
  Pseudo-Huber (δ=1):    0.002543
  Pseudo-Huber (δ=10):   0.003701
  Dir Acc:               57.84%

INTERPRETATION
==============
  ✓ Good fit: R² = 0.0731
  ○ Slight edge: 54.4%
  ✓ Good thresholded direction (|y| ≥ 0.0010): 57.8%
```

### 11.5 Co oznaczają Error Stats

- **Mean** (błąd średni): wartość bliska 0 = model nie ma systematycznego biasu
- **Median** (błąd medianowy): bardziej odporny na outliery niż mean
- **Std** (odchylenie standardowe błędów): rozrzut błędów; im mniejszy tym lepiej

### 11.6 Feature Importance

Zapisywana w `feature_importance.json` z trzema typami ważności:

| Typ | Opis |
|---|---|
| `gain` | Średnia redukcja straty przez splits danej cechy — **najważniejsza** |
| `weight` | Liczba razy cecha użyta do podziału |
| `cover` | Średnia liczba próbek dotknięta przez split danej cechy |

`top20_gain` zawiera 20 najważniejszych cech wg gain.

Wykres `feature_importance.png` pokazuje top-30 cech wg gain po spłaszczeniu (np. `return_1_lag_1`, `rsi_mean_24`).

---

## 12. Przykłady użycia

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
    --lookback 48 \
    --horizon 12

# Krok 2: Tuning z optymalizacją t_dir_acc
python xgb_model/xgb_hyperparameter_tuning.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --n_trials 150 \
    --feature_set technical \
    --optimize_target val \
    --optimize_metric t_dir_acc

# Krok 3: Trening z najlepszymi parametrami
python xgb_model/xgb_train.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --load_optuna_params xgb_data/XO001_xgb_optuna_XB001_0103/best_params.json

# Krok 4: Test
python xgb_model/xgb_test.py \
    --model_dir xgb_models/X001_btc_xgb_model_XB001_0103
```

### Scenariusz 3: Eksperymenty z feature setami

```bash
# Porównanie feature setów (każdy tworzy osobny model z unikalnym ID)
for fs in technical xgb_momentum xgb_volatility full; do
    python xgb_train.py \
        --data_path prepared_data/XB001_prep_btc_full_1m \
        --feature_set $fs \
        --flattening_method hybrid
done
```

### Scenariusz 4: Eksperymenty z metodami flatteningu

```bash
# Porównanie metod flatteningu
for fm in lag agg hybrid signal none; do
    python xgb_train.py \
        --data_path prepared_data/XB001_prep_btc_full_1m \
        --flattening_method $fm \
        --feature_set technical
done
```

### Scenariusz 5: Trening bez custom objective (standard XGBoost)

```bash
python xgb_train.py \
    --data_path prepared_data/XB001_prep_btc_full_1m \
    --objective reg:pseudohubererror \
    --no_custom_obj_sample_weights \
    --no_custom_obj_misdir_penalty
```

### Scenariusz 6: Użycie buy_pressure (dane Binance pełne)

```bash
# Dane muszą zawierać kolumnę taker_buy_base_asset_volume
python xgb_train.py \
    --data_path data/btc_1m_binance_full.csv \
    --feature_set technical_bp \
    --flattening_method hybrid
```

---

## 13. System identyfikatorów

| Prefix | Format | Przykład | Znaczenie |
|---|---|---|---|
| `X` | X001 | X001, X002 | Model ID (w nazwie folderu modelu) |
| `XB` | XB001 | XB001, XB002 | Data ID (prepared dataset) |
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

## 14. Pliki wyjściowe

### Model (xgb_models/X001_.../):

| Plik | Opis |
|---|---|
| `xgb_model.json` | Wagi modelu XGBoost — do wczytania przez `xgb.Booster().load_model()` |
| `model_config.json` | Pełna konfiguracja: features, lookback, horizon, hiperparametry, metryki val, threshold, custom obj |
| `feature_scaler.pkl` | Skaler cech (StandardScaler) — do transformacji danych wejściowych |
| `target_scaler.pkl` | Skaler targetu (RobustScaler) — do inverse transform predykcji |
| `feature_importance.json` | Gain/weight/cover dla wszystkich flat features + top20_gain |
| `training.log` | Szczegółowy log procesu treningu z metrykami skalowanymi i oryginalnymi |
| `summary.txt` | Podsumowanie: hiperparametry + metryki train/val (standardowe i progowe t_*) |

`model_config.json` zawiera m.in.:
- `signal_threshold_resolved` — dokładna wartość progu użyta podczas treningu
- `signal_threshold_enabled`, `signal_threshold_mode`, `signal_threshold_value`
- `use_custom_objective`, `custom_obj_type`, `custom_obj_huber_delta`
- `sample_weight_mode`, `reweight_misdir`

### Tuning (xgb_data/XO001_.../):

| Plik | Opis |
|---|---|
| `best_params.json` | Najlepsze parametry z metadanymi (optimize_metric, best_value_display, negation info) |
| `second_best_params.json` | 2. najlepszy zestaw parametrów |
| `third_best_params.json` | 3. najlepszy zestaw parametrów |
| `study.pkl` | Obiekt Optuna Study — do dalszej analizy w Python |
| `trials.csv` | Wszystkie próby z wartościami metryk i parametrów |
| `tuning.log` | Log procesu tuningu |
| `summary.txt` | Podsumowanie top-3 prób z informacją o metryce i kierunku optymalizacji |
| `history.png` | Historia optymalizacji |
| `importances.png` | Ważność hiperparametrów |
| `distribution.png` | Rozkład wartości prób |
| `convergence.png` | Konwergencja (kumulatywne minimum per trial) |

### Test (xgb_data/XT001_.../):

| Plik | Opis |
|---|---|
| `results.csv` | `actual_log`, `predicted_log`, `error` dla każdej próbki |
| `metrics.json` | MAE, RMSE, R², Pseudo-Huber (δ=1 i δ=10), Dir Acc + metryki t_* jeśli aktywne |
| `summary.txt` | Podsumowanie tekstowe z metrykami standardowymi i progowymi |
| `log_returns_timeseries.png` | Pełna seria log-return: actual (niebieski) vs predicted (czerwony) |
| `log_returns_scatter.png` | Scatter: actual vs predicted |
| `error_distribution.png` | Histogram błędów |
| `combined_results.png` | 2×2: time series + scatter + histogram + rolling directional accuracy |
| `feature_importance.png` | Wykres słupkowy top-30 cech wg gain |
