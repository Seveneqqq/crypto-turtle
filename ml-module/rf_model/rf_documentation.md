# Random Forest Cryptocurrency Prediction — Dokumentacja
*może zawierać błędy* 😘

## Spis treści

1. [Przegląd systemu](#1-przegląd-systemu)
2. [Architektura i struktura plików](#2-architektura-i-struktura-plików)
3. [Przepływ danych](#3-przepływ-danych)
4. [Feature Engineering](#4-feature-engineering)
5. [Konfiguracja modelu (rf_config.py)](#5-konfiguracja-modelu-rf_configpy)
6. [Trening modelu (rf_train.py)](#6-trening-modelu-rf_trainpy)
7. [Tuning hiperparametrów (rf_hyperparameter_tuning.py)](#7-tuning-hiperparametrów-rf_hyperparameter_tuningpy)
8. [Testowanie modelu (rf_test.py)](#8-testowanie-modelu-rf_testpy)
9. [Metryki i ich interpretacja](#9-metryki-i-ich-interpretacja)
10. [Przykłady użycia](#10-przykłady-użycia)
11. [System identyfikatorów](#11-system-identyfikatorów)
12. [Pliki wyjściowe](#12-pliki-wyjściowe)

---

## 1. Przegląd systemu

Model Random Forest przewiduje **log-return** (logarytmiczną stopę zwrotu) kryptowaluty dla zadanego horyzontu czasowego. Jest to model regresji tablicowej (flat/tabular), który:

- przyjmuje dane OHLCV w formacie 1-minutowym (lub innych interwałach),
- generuje cechy techniczne (feature engineering),
- „spłaszcza" sekwencję historyczną (`lookback` świec) do pojedynczego wektora cech (`flattening`),
- przewiduje `log(close[t+horizon] / close[t])` — czyli oczekiwany procentowy ruch ceny.

**Target: log-return, nie cena absolutna.** Wartość `+0.001` oznacza oczekiwany wzrost o ~0.1%, wartość `-0.002` oznacza oczekiwany spadek o ~0.2%.

Model RF powstał jako port modelu XGBoost z zamianą XGBoost na `sklearn.ensemble.RandomForestRegressor`. Pipeline danych, metryki i konwencje nazewnictwa są identyczne. Kluczowe różnice to: brak early stoppingu (RF trenuje wszystkie drzewa), zapis modelu przez joblib (`.pkl`), OOB score jako proxy jakości na danych treningowych, oraz dodatkowy mechanizm ważenia próbek (`sample_weight`) i kary za błędny kierunek (`REWEIGHT_MISDIR`).

---

## 2. Architektura i struktura plików

```
ml/
├── feature_engineering.py          ← UNIVERSAL — generuje cechy OHLCV
│
├── rf_model/
│   ├── rf_config.py                ← Konfiguracja (hiperparametry, ścieżki, cele)
│   ├── rf_train.py                 ← Trening modelu
│   ├── rf_test.py                  ← Testowanie modelu
│   ├── rf_hyperparameter_tuning.py ← Optuna tuning
│   │
│   ├── common/
│   │   ├── data_utils.py           ← Ładowanie danych, flattening, scalery targetu
│   │   ├── metrics.py              ← Obliczanie metryk (log-return space)
│   │   ├── training_utils.py       ← Fit RF, sample weights, misdir penalty, zapis modelu
│   │   ├── id_utils.py             ← System identyfikatorów (R001, RO001, RT001)
│   │   └── visualization.py        ← Wykresy wyników i feature importance
│   │
│   └── rf_data/                            ← Wyniki tuningu i testów
│─── (../shared/models/)rf_models/          ← Zapisane modele (R001_btc_rf_model_...)
│─── (../shared/models/)rf_prepared_data/   ← Dane przygotowane przez data_preparation/
│
├── (../shared/models/)model_package/   ← Standalone inference (sygnały BUY/SELL)
│   ├── rf_model_package.py             ← Klasa RFModelPackage
│   └── feature_engineering.py          ← Lokalna kopia universal FE
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
  ['lag' | 'agg' | 'hybrid' | 'signal']
       │
       ▼
create_log_return_target()               ← y = log(close[t+h] / close[t])
       │
       ▼
(opcja) winsorize_targets()              ← przycięcie ekstremalnych log-returnów
       │
       ▼
StandardScaler (cechy) + TargetScaler (target)
       │
       ▼
(opcja) compute_sample_weights()         ← wagi próbek wg |y|
       │
       ▼
RandomForestRegressor.fit()
  └── (opcja) apply_misdir_penalty()     ← drugi fit z karą za błędny kierunek
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
rf_prepared_data/
└── E104_prep_eth_universal_full_1m/
    ├── train.csv
    ├── test.csv
    ├── scaler_features.pkl
    └── metadata.json
```

Dane z `data_preparation/` są **współdzielone** z modelem XGBoost — ten sam prepared dir może być używany przez oba modele.

**Tryb 2 — Raw CSV** (on-the-fly):

Surowy plik CSV z kolumnami OHLCV. Feature engineering i skalowanie wykonywane są automatycznie podczas ładowania.

### 3.3 Flattening — spłaszczanie sekwencji

Random Forest nie obsługuje danych sekwencyjnych. Sekwencja `lookback` świec jest przekształcana w pojedynczy wiersz cech jedną z czterech metod:

**`lag`** — proste opóźnienia:
Dla każdej cechy tworzy kolumny `feature_lag_1`, `feature_lag_2`, ..., `feature_lag_{lookback}`.
Wynikowy wymiar: `n_features × lookback`. Przy 23 cechach i lookback=48 → 1104 kolumny.
Zachowuje pełną informację o kolejności, ale tworzy dużo cech.

**`agg`** — agregacje:
Dla każdej cechy tworzy 5 statystyk z okna `[t-lookback, t-1]`: mean, std, min, max, last.
Wynikowy wymiar: `n_features × 5`. Przy 23 cechach → 115 kolumn.
Szybki, kompaktowy, dobry punkt startowy.

**`hybrid`** — kombinacja strategicznych opóźnień i agregacji wielookienkowych:
- opóźnienia: `[1, 2, 3, 5, 10, 24]` (domyślnie)
- okna agregacji: mean i std dla `[6, 12, 24, 48]`
- różnice pierwszego rzędu: `diff_1`, `diff_5`
Wynikowy wymiar: zależy od konfiguracji; zwykle 200–400 kolumn.
Kompromis między szczegółowością a rozmiarem.

**`signal`** — cechy kierunkowe i momentum:
Zamiast statystyk opisowych generuje cechy wprost enkodujące kierunek i trend:
- `trend_S_L` — `mean(S) - mean(L)` dla każdej pary krótkich/długich okien (np. mean_6 − mean_24)
- `last_vs_mean` — ostatnia wartość minus średnia całego okna
- `sign_frac` — frakcja wartości > 0 w oknie (szczególnie informacyjna dla cech `return_*`)
- `diff_N` — różnice przy strategicznych lagach (1, 3, 5, 12)
- `accel_1_5` — diff(1) − diff(5): krótkoterminowe przyspieszenie vs. średnioterminowe

Wszystkie wejścia są przesunięte o 1 (`shift(1)`) — brak look-ahead leakage.
Okna konfigurowane przez `SIGNAL_SHORT_WINDOWS`, `SIGNAL_LONG_WINDOWS`, `SIGNAL_DIFF_LAGS` w `rf_config.py`.

### 3.4 Target: log-return

```python
y = log(close[t + horizon] / close[t])
```

Gdzie `t` to moment na który patrzymy (ostatnia znana świeca), a `close[t+horizon]` to cena za `horizon` świec w przyszłość.

Wartości:
- `+0.002` → wzrost o ~0.2%
- `-0.001` → spadek o ~0.1%
- `0.0` → brak zmiany

### 3.5 Podział train/val

Podział jest **chronologiczny** (80%/20%) — brak shuffle, brak data leakage:
- Train: pierwsze 80% sekwencji
- Val: ostatnie 20% sekwencji

RF nie ma early stoppingu, więc zbiór walidacyjny służy wyłącznie do ewaluacji i tuningu Optuna.

---

## 4. Feature Engineering

Źródło: `feature_engineering.py` (universal — wspólny dla LSTM, XGBoost i Random Forest).

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

#### Zestawy uniwersalne (LSTM, XGBoost i RF)

| Zestaw | Cechy | Opis |
|---|---|---|
| `full` | wszystkie (22–23) | Kompletny zestaw |
| `technical` | 15 | Bez time features i vol_regime — bezpieczny dla krótkich okien |
| `technical_bp` | 16 | `technical` + buy_pressure |
| `minimal` | 6 | return_1, return_5, close_sma_14, rsi, volatility_14, relative_volume |

#### Zestawy dedykowane XGBoost/RF

| Zestaw | Cechy | Kiedy używać |
|---|---|---|
| `xgb_momentum` | 11–12 | Trend/momentum z vol_regime; standardowy wybór |
| `xgb_volatility` | 9 | Fokus na reżim zmienności i rozstępy cen |
| `xgb_returns` | 4 | Tylko returny — maksymalna przenośność danych |

Zestawy `xgb_*` działają identycznie dla RF — nazwy wynikają z historii (RF powstał z portu XGBoost).

#### Aliasy legacy

| Alias | Mapuje do |
|---|---|
| `momentum` | `xgb_momentum` |
| `volatility_focused` | `xgb_volatility` |

---

## 5. Konfiguracja modelu (rf_config.py)

### 5.1 Parametry danych

```python
DEFAULT_LOOKBACK    = 48        # liczba świec historycznych (sekwencja wejściowa)
DEFAULT_HORIZON     = 12        # horyzont predykcji (świece w przyszłość)
DEFAULT_FLATTENING  = 'hybrid'  # metoda spłaszczania: 'lag' | 'agg' | 'hybrid' | 'signal'
DEFAULT_FEATURE_SET = 'minimal' # zestaw cech
```

`lookback=48` przy danych 1-minutowych = ostatnie 48 minut historii.
`horizon=12` = przewidujemy cenę za 12 minut.

### 5.2 Funkcja straty (OBJECTIVE / criterion)

```python
OBJECTIVE = 'absolute_error'
```

Dostępne funkcje straty (sklearn criterion):

| Wartość | Opis | Kiedy używać |
|---|---|---|
| `absolute_error` | MAE — medianowa regresja (domyślna) | Odporność na outliery; zalecana dla log-returnów |
| `squared_error` | MSE — średnia kwadratów błędów | Wrażliwa na outliery; gdy dane czyste |
| `friedman_mse` | MSE z kryterium podziału Friedmana | Często lepsza jakość splitów niż squared_error |

**Niezgodne** (nie używać z log-return targetem):
- `poisson` — wymaga nieujemnych etykiet (log-returny mogą być ujemne)

### 5.3 Skalowanie targetu

```python
SCALE_TARGET       = True
TARGET_SCALER_TYPE = 'robust'   # 'standard' | 'robust'
```

Log-returny mają małe wartości (rząd 0.001). Skalowanie normalizuje target do ~N(0,1) przed treningiem i odwraca po predykcji. RF jest mniej wrażliwy niż XGBoost na skalę targetu (drzewa są odporne na monotoniczne transformacje), ale skalowanie ułatwia porównanie metryk i może poprawić sample weights.

| Typ | Opis | Kiedy używać |
|---|---|---|
| `robust` | RobustScaler — mediana i IQR (domyślny) | Dane z flash crashami / pump&dump |
| `standard` | StandardScaler — mean=0, std=1 | Dane bez ekstremalnych outlierów |

Skalowanie jest **odwracalne** — metryki testowe zawsze raportowane są w oryginalnej przestrzeni log-return.

### 5.4 Winsoryzacja targetu

```python
WINSORIZE_TARGET       = False
WINSORIZE_PERCENTILE   = 99.0
```

Obcina skrajne wartości log-return (np. flash crashe). Granice wyznaczane **tylko** na danych treningowych (brak data leakage). Val/test obcinane tymi samymi granicami.

Przykład z `WINSORIZE_PERCENTILE=99.0`: obcina zewnętrzne 1% rozkładu (dolne 0.5% i górne 0.5%).

Kiedy włączyć: gdy dane zawierają ekstremalne zdarzenia rynkowe niereprezentatywne dla normalnego zachowania.

### 5.5 Sample Weights — ważenie próbek

```python
SAMPLE_WEIGHT_MODE       = 'percentile'   # None | 'abs' | 'abs_clip' | 'rank' | 'percentile'
SAMPLE_WEIGHT_ALPHA      = 1.0            # wykładnik dla 'abs' / 'abs_clip'
SAMPLE_WEIGHT_MIN_W      = 0.1            # podłoga wagi dla 'abs_clip'
SAMPLE_WEIGHT_PERCENTILE = 70.0           # próg percentyla dla 'percentile'
SAMPLE_WEIGHT_LOW_W      = 0.1            # waga poniżej progu dla 'percentile'
```

Mechanizm ważenia sprawia, że model skupia się bardziej na próbkach z dużymi absolutnymi log-returnami (duże ruchy rynkowe) i mniej na próbkach z wartościami bliskimi zero (szum, płaski rynek).

Wagi są zawsze obliczane z **niezskalowanego** `y_train` — niezależnie od ustawień `SCALE_TARGET`.

| Tryb | Opis | Parametry |
|---|---|---|
| `None` | Brak ważenia — jednolite wagi (domyślne sklearn) | — |
| `abs` | `w = \|y\|^alpha` — proporcjonalne do absolutnego log-returnu | `SAMPLE_WEIGHT_ALPHA` |
| `abs_clip` | Jak `abs`, ale z podłogą `min_w × max(w)` — zapobiega wagom bliskim 0 | `SAMPLE_WEIGHT_ALPHA`, `SAMPLE_WEIGHT_MIN_W` |
| `rank` | `w = rank(\|y\|) / N` — rank-based, skalowalny | — |
| `percentile` | `w = 1.0` jeśli `\|y\| >= p-ty percentyl`, inaczej `low_w` | `SAMPLE_WEIGHT_PERCENTILE`, `SAMPLE_WEIGHT_LOW_W` |

**Przykład konfiguracji `percentile` (domyślny):**
```python
SAMPLE_WEIGHT_MODE       = 'percentile'
SAMPLE_WEIGHT_PERCENTILE = 70.0   # top 30% próbek wg |y| dostaje wagę 1.0
SAMPLE_WEIGHT_LOW_W      = 0.1    # dolne 70% dostaje wagę 0.1 (nie 0 — nie ignoruj)
```

Aby wyłączyć ważenie:
```python
SAMPLE_WEIGHT_MODE = None
```

### 5.6 Misdir Penalty — kara za błędny kierunek

```python
REWEIGHT_MISDIR     = False   # domyślnie wyłączone
MISDIR_PENALTY      = 2.0     # mnożnik wagi dla błędnie sklasyfikowanych (≥ 1.0)
MISDIR_MIN_QUANTILE = 0.7     # próg |y| — tylko top 30% ruchów kwalifikuje się do kary
```

Gdy `REWEIGHT_MISDIR = True`, trening przebiega w dwóch etapach:

1. **Fit 1** — normalny trening na `sample_weight` (jeśli aktywny).
2. **Predykcja na train** — model z Fit 1 przewiduje log-returny dla danych treningowych.
3. **apply_misdir_penalty** — próbki, gdzie `sign(y_pred) ≠ sign(y_true)` AND `|y_true| >= MISDIR_MIN_QUANTILE` percentyl, dostają wagi pomnożone przez `MISDIR_PENALTY`. Funkcja zwraca `(updated_weights, n_penalised)` — liczba ukaranych próbek logowana bezpośrednio.
4. **Fit 2** — nowy model trenowany z zaktualizowanymi wagami.

Wagi wyłącznie **rosną** — mechanizm nigdy nie zeruje wag, co chroni model przed spłaszczaniem predykcji do zera.

Koszt: czas treningu ×2 (dwa pełne fity RF).

| Parametr | Zalecany zakres | Opis |
|---|---|---|
| `MISDIR_PENALTY` | 1.5–3.0 | Wartości > 3.0 mogą przefitować model do dużych ruchów |
| `MISDIR_MIN_QUANTILE` | 0.6–0.8 | 0.7 = top 30% ruchów kwalifikuje; niżej = więcej próbek karanych |

### 5.7 Cel optymalizacji Optuna

```python
OPTUNA_OPTIMIZE_TARGET = 'train'   # 'val' | 'train'
```

Kontroluje, na którym zbiorze Optuna minimalizuje stratę:
- `'val'` — zalecane w produkcji: wybiera parametry generalizujące na nowe dane
- `'train'` — przydatne do diagnozy underfittingu

### 5.8 Hiperparametry domyślne Random Forest

```python
DEFAULT_PARAMS = {
    'n_estimators':             862,        # liczba drzew
    'max_depth':                15,         # maksymalna głębokość drzewa
    'max_features':             0.33,       # frakcja cech rozważanych przy każdym splicie
    'max_samples':              0.85,       # frakcja próbek do budowy każdego drzewa (bootstrap)
    'min_samples_leaf':         9,          # min próbek w liściu
    'min_samples_split':        2,          # min próbek do próby podziału węzła
    'min_impurity_decrease':    0.0001796,  # min redukcja impurity do akceptacji splitu
    'min_weight_fraction_leaf': 0.0,        # min ważona frakcja sumy wag w liściu
    'bootstrap':                True,       # losowanie ze zwracaniem
    'oob_score':                True,       # monitorowanie OOB R² (proxy treningu)
    'n_jobs':                   -1,         # wszystkie dostępne CPU
    'random_state':             42,
}
```

### 5.9 Opis kluczowych hiperparametrów

**`n_estimators`** — liczba drzew w lesie. Więcej = stabilniejsze predykcje, ale dłuższy trening. Powyżej ~500 drzew wzrost jakości jest coraz mniejszy.

**`max_depth`** — maksymalna głębokość drzewa. `None` = drzewa w pełni rozwinięte (ryzyko overfittingu). Zalecany zakres: 5–20 dla dużych danych krypto.

**`max_features`** — frakcja cech losowo rozważanych przy każdym splicie. Kluczowy parametr regularyzacji RF. Niższy = bardziej zróżnicowane drzewa (lepsza decorrelacja), wyższy = każde drzewo jest bardziej „kompetentne". Zakres: 0.1–1.0.

**`max_samples`** — frakcja próbek losowana (z powtórzeniami) do budowy każdego drzewa. Wymaga `bootstrap=True`. Zakres: 0.5–1.0.

**`min_samples_leaf`** — minimalna liczba próbek w każdym liściu. Podstawowy parametr regularyzacji — zapobiega zbyt szczegółowym podziałom. Przy dużych danych (800k próbek) sensowny zakres: 1–20.

**`min_samples_split`** — minimalna liczba próbek **w węźle** wymagana do próby podziału. Uzupełnia `min_samples_leaf` — kontroluje głębokość drzewa od góry. Przy dużych danych sensowny zakres: 2–20.

**`min_impurity_decrease`** — minimalna redukcja impurity wymagana do akceptacji podziału. Drobna regularyzacja — małe wartości (0.0–0.01).

**`min_weight_fraction_leaf`** — minimalna **ważona** frakcja sumy wag próbek w liściu. Aktywny i sensowny wyłącznie gdy `SAMPLE_WEIGHT_MODE is not None` — przy wagach `min_samples_leaf` liczy surowe próbki, a nie ich ważony wkład, co ta wartość naprawia. Przy wyłączonych wagach ustaw `0.0`. Zakres: 0.0–0.01.

**`oob_score`** — gdy `True` (i `bootstrap=True`), oblicza R² na próbkach Out-Of-Bag — darmowy estymator jakości na danych, których dane drzewo nie widziało, bez konieczności osobnego zbioru val. Koszt: trochę wolniejszy trening.

### 5.10 Konfiguracja flatteningu `signal`

```python
SIGNAL_SHORT_WINDOWS = [3, 6, 12]    # krótkie okna dla trend_S_L
SIGNAL_LONG_WINDOWS  = [24, 48]      # długie okna dla trend_S_L
SIGNAL_DIFF_LAGS     = [1, 3, 5, 12] # lagi różnic dla diff_N i accel_1_5
```

Okna > `lookback` są automatycznie filtrowane. Można dostosować do szybszych lub wolniejszych interwałów danych.

---

## 6. Trening modelu (rf_train.py)

### 6.1 Uruchomienie z linii poleceń

**Podstawowe (z przygotowanymi danymi):**
```bash
python rf_train.py --data_path rf_prepared_data/E104_prep_eth_universal_full_1m
```

**Z surowym CSV:**
```bash
python rf_train.py --data_path data/eth_1m.csv
```

**Z parametrami Optuna:**
```bash
python rf_train.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --load_optuna_params rf_data/RO001_rf_optuna_E104_2203/best_params.json
```

**Z ręcznymi nadpisaniami:**
```bash
python rf_train.py \
    --data_path data/eth_1m.csv \
    --lookback 72 \
    --horizon 6 \
    --flattening_method signal \
    --feature_set xgb_momentum \
    --n_estimators 500 \
    --max_depth 10 \
    --max_features 0.4 \
    --min_samples_leaf 5 \
    --min_samples_split 5 \
    --scale_target \
    --target_scaler_type robust \
    --sample_weight_mode percentile \
    --sample_weight_percentile 70.0 \
    --sample_weight_low_w 0.1
```

### 6.2 Opis parametrów CLI

**Dane:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--data_path` | wymagany | Ścieżka do prepared dir lub pliku .csv |
| `--load_optuna_params` | None | Ścieżka do `best_params.json` z Optuna |
| `--lookback` | 48 | Długość okna historycznego (liczba świec) |
| `--horizon` | 12 | Horyzont predykcji (liczba świec w przód) |
| `--flattening_method` | `hybrid` | Metoda spłaszczania: `lag`, `agg`, `hybrid`, `signal` |
| `--feature_set` | `minimal` | Zestaw cech (patrz sekcja 4.2) |

**Hiperparametry Random Forest:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--n_estimators` | 862 | Liczba drzew |
| `--max_depth` | 15 | Maksymalna głębokość drzewa |
| `--max_features` | 0.33 | Frakcja cech per split (0.1–1.0) |
| `--max_samples` | 0.85 | Frakcja próbek per drzewo (0.5–1.0) |
| `--min_samples_leaf` | 9 | Min próbek w liściu (1–20) |
| `--min_samples_split` | 2 | Min próbek do próby podziału węzła (2–20) |
| `--min_impurity_decrease` | 0.0001796 | Min redukcja impurity do podziału (0.0–0.01) |
| `--min_weight_fraction_leaf` | 0.0 | Min ważona frakcja sumy wag w liściu (0.0–0.01) |
| `--no_bootstrap` | — | Wyłącz bootstrap (oob_score automatycznie False) |
| `--criterion` | z config | Funkcja straty: `squared_error`, `absolute_error`, `friedman_mse` |

**Target:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--scale_target` | True (z config) | Włącz skalowanie targetu |
| `--no_scale_target` | — | Wyłącz skalowanie targetu |
| `--target_scaler_type` | `robust` | Typ skalera: `standard`, `robust` |
| `--winsorize_target` | False | Włącz winsoryzację targetu |
| `--winsorize_percentile` | 99.0 | Percentyl winsoryzacji (90–99.9) |

**Sample weights:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--sample_weight_mode` | `percentile` | Tryb ważenia: `abs`, `abs_clip`, `rank`, `percentile` lub pomiń |
| `--sample_weight_alpha` | 1.0 | Wykładnik dla trybów `abs` / `abs_clip` |
| `--sample_weight_min_w` | 0.1 | Podłoga wagi dla trybu `abs_clip` |
| `--sample_weight_percentile` | 70.0 | Próg percentyla dla trybu `percentile` |
| `--sample_weight_low_w` | 0.1 | Waga poniżej progu dla trybu `percentile` |

**Inne:**

| Parametr | Domyślny | Opis |
|---|---|---|
| `--seed` | 42 | Ziarno losowości |
| `--verbose` | 1 | Poziom logowania sklearn: 0=cicho, 1=postęp, 2=per-drzewo |

### 6.3 Priorytet parametrów

Gdy podano `--load_optuna_params`, parametry z pliku JSON są ładowane jako baza. Każdy parametr CLI podany **dodatkowo** nadpisuje wartość z Optuna. Nienadpisane parametry biorą wartości z Optuna, a jeśli brak — z `RFConfig.DEFAULT_PARAMS`.

Parametry `REWEIGHT_MISDIR`, `MISDIR_PENALTY`, `MISDIR_MIN_QUANTILE` są zawsze czytane z `rf_config.py` — nie są eksponowane przez CLI (modyfikacja w pliku konfiguracyjnym).

### 6.4 OOB Score

Gdy `bootstrap=True` i `oob_score=True`, RF oblicza R² na Out-Of-Bag samples — próbkach, których dane drzewo nie widziało podczas budowy. Jest to estymator jakości zbliżony do cross-validation, bez kosztu osobnego zbioru walidacyjnego. Logowany po treningu jako `OOB R2 (training proxy)`.

OOB score jest **wyłączany** podczas tuningu Optuna (`oob_score=False`) aby przyspieszyć każdą próbę.

Wartość `oob_score` w `params` jest sterowana flagą `RFConfig.USE_OOB_SCORE` — przy `bootstrap=True` przyjmuje jej wartość, przy `bootstrap=False` jest automatycznie wyłączana.

### 6.5 API Python

```python
from rf_train import train

model, feature_scaler, config = train(
    data_path='rf_prepared_data/E104_prep_eth_universal_full_1m',
    lookback=48,
    horizon=12,
    flattening_method='hybrid',
    feature_set='xgb_momentum',
    n_estimators=500,
    max_depth=10,
    max_features=0.4,
    min_samples_leaf=5,
    min_samples_split=5,
    scale_target=True,
    target_scaler_type='robust',
    sample_weight_mode='percentile',
    sample_weight_percentile=70.0,
    sample_weight_low_w=0.1,
)

print(config['val_metrics'])   # {'mae': ..., 'r2': ..., 'dir_acc': ...}
print(config['oob_score'])     # OOB R² (proxy jakości na train)
print(config['output_dir'])    # ścieżka do zapisanego modelu
```

---

## 7. Tuning hiperparametrów (rf_hyperparameter_tuning.py)

### 7.1 Uruchomienie

```bash
python rf_hyperparameter_tuning.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --n_trials 100 \
    --feature_set xgb_momentum \
    --optimize_target val
```

**Z fiksowanym flatteningiem i tunowaniem lookback:**
```bash
python rf_hyperparameter_tuning.py \
    --data_path data/eth_1m.csv \
    --n_trials 150 \
    --tune_lookback \
    --flattening_method hybrid \
    --scale_target \
    --target_scaler_type robust
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
| `--enable_pruning` | False | Włącz MedianPruner (domyślnie wyłączony — patrz 7.4) |

### 7.3 Metryki optymalizacji Optuna

RF oferuje cztery różne metryki do minimalizacji, konfigurowane przez `TuningConfig.OPTIMIZE_METRIC` w pliku tuningu:

| Metryka | Opis | Kiedy używać |
|---|---|---|
| `mae` | Średni błąd absolutny log-returnu | Podstawowa, stabilna metryka |
| `dir_acc` | `1 - directional_accuracy` (minimalizacja = max dir_acc) | Gdy celem jest przewidywanie kierunku |
| `composite` | `w × MAE_norm + (1-w) × (1 - dir_acc)` (domyślna) | Balans między dokładnością a kierunkiem |
| `sharpe` | `MAE / std(errors)` — penalizuje niespójność błędów | Faworyzuje modele z regularnym błędem |

Waga MAE w metryce `composite` kontrolowana przez `TuningConfig.COMPOSITE_MAE_WEIGHT` (domyślnie `0.7`):
```python
OPTIMIZE_METRIC      = 'composite'
COMPOSITE_MAE_WEIGHT = 0.7   # 70% MAE + 30% (1 - dir_acc)
```

### 7.4 Przestrzeń hiperparametrów (domyślna)

| Hiperparametr | Typ | Zakres |
|---|---|---|
| `n_estimators` | int | [200, 900] |
| `max_depth` | int | [3, 20] |
| `min_samples_leaf` | int | [1, 20] |
| `min_samples_split` | int | [2, 20] |
| `min_impurity_decrease` | float | [0.0, 0.01] |
| `max_features` | float | [0.1, 1.0] |
| `max_samples` | float | [0.5, 1.0] |
| `min_weight_fraction_leaf` | float | [0.0, 0.01] (tylko gdy `SAMPLE_WEIGHT_MODE is not None`) |
| `flattening_method` | categorical | `lag`, `agg`, `hybrid`, `signal` |
| `lookback` | categorical | [24, 36, 48, 72, 96, 120] (jeśli `--tune_lookback`) |
| `horizon` | categorical | [1, 3, 6, 12, 24] (jeśli `--tune_horizon`) |

Uwaga: `min_weight_fraction_leaf` wchodzi do search space **wyłącznie** gdy `TuningConfig.SAMPLE_WEIGHT_MODE is not None` — przy wyłączonych wagach jest stałe 0.0 i nie obciąża przestrzeni przeszukiwań.

### 7.5 Algorytm i pruning

- **Sampler:** TPE (Tree-structured Parzen Estimator) — bayesowski algorytm próbkowania
- **Pruner:** MedianPruner — domyślnie **wyłączony** (`ENABLE_PRUNING = False`)

RF raportuje wynik tylko raz po zakończeniu treningu (brak iteracyjnych epok jak w XGBoost/LSTM), więc MedianPruner może wyłącznie odrzucać próby, które zakończyły się wynikiem gorszym od mediany — bez możliwości przerwania w trakcie. Włącz go flagą `--enable_pruning` jeśli chcesz agresywnie odrzucać słabe konfiguracje po pierwszym ficie.

### 7.6 Wyniki tuningu

Zapisywane w `rf_data/RO001_rf_optuna_{data_id}_{data}/`:

```
RO001_rf_optuna_E104_2203/
├── best_params.json           ← parametry zwycięskiej próby
├── second_best_params.json    ← 2. miejsce
├── third_best_params.json     ← 3. miejsce
├── study.pkl                  ← obiekt Optuna Study
├── trials.csv                 ← wszystkie próby z wynikami
├── tuning.log                 ← log procesu
├── summary.txt                ← podsumowanie tekstowe
├── history.png                ← historia optymalizacji
├── importances.png            ← ważność hiperparametrów
├── distribution.png           ← rozkład wyników prób
└── convergence.png            ← krzywa zbieżności (cumulative best)
```

### 7.7 Trening po tuningu

```bash
python rf_train.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --load_optuna_params rf_data/RO001_rf_optuna_E104_2203/best_params.json
```

---

## 8. Testowanie modelu (rf_test.py)

### 8.1 Uruchomienie

**Auto-detect modelu i danych:**
```bash
python rf_test.py
```

**Wskazanie konkretnego modelu:**
```bash
python rf_test.py --model_dir rf_models/R001_eth_rf_model_E104_2203
```

**Wskazanie danych testowych:**
```bash
python rf_test.py \
    --model_dir rf_models/R001_eth_rf_model_E104_2203 \
    --test_path rf_prepared_data/E104_prep_eth_universal_full_1m
```

**Test na surowym CSV:**
```bash
python rf_test.py \
    --model_dir rf_models/R001_eth_rf_model_E104_2203 \
    --test_path data/eth_test.csv
```

### 8.2 Parametry CLI

| Parametr | Domyślny | Opis |
|---|---|---|
| `--model_dir` | najnowszy w `rf_models/` | Folder z wytrenowanym modelem |
| `--test_path` | auto-detect | Folder prepared dir lub plik .csv |

### 8.3 Auto-detect danych testowych

Gdy `--test_path` nie jest podany, `test_model()` szuka danych w kolejności:

1. Jeśli `train_path` z konfiguracji modelu to prepared dir → użyj tego samego folderu (zawiera `test.csv`)
2. Szukaj w `rf_prepared_data/` po `data_id` z konfiguracji modelu
3. Szukaj w `data_preparation/prepared_data/` (katalog współdzielony)
4. Szukaj w `xgb_model/xgb_prepared_data/` (prepared dir XGBoost w formacie kompatybilnym)
5. Błąd z instrukcją co podać

Kroki 3 i 4 oznaczają, że RF może automatycznie korzystać z danych przygotowanych dla XGBoost — format prepared dir jest wspólny.

### 8.4 Pliki testowanego modelu

```
rf_models/R001_eth_rf_model_E104_2203/
├── rf_model.pkl             ← model RF (joblib)
├── model_config.json        ← pełna konfiguracja modelu
├── feature_scaler.pkl       ← skaler cech
├── target_scaler.pkl        ← skaler targetu (jeśli był użyty)
├── feature_importance.json  ← MDI ważność cech
├── training.log             ← log treningu
└── summary.txt              ← podsumowanie treningu
```

### 8.5 Pliki wynikowe testu

Zapisywane w `rf_data/RT001_test_results_{model_id}_{data_id}_{data}/`:

```
RT001_test_results_R001_E104_2203/
├── results.csv                  ← predykcje i błędy dla każdej próbki
├── metrics.json                 ← metryki w formacie JSON
├── summary.txt                  ← podsumowanie tekstowe
├── log_returns_timeseries.png   ← pełna seria czasowa actual vs predicted
├── log_returns_scatter.png      ← scatter: actual vs predicted
├── error_distribution.png      ← rozkład błędów
├── combined_results.png         ← siatka 2×2: seria, scatter, błędy, rolling dir_acc
└── feature_importance.png       ← top-30 ważności cech MDI
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

**Ważne:** RF przewiduje **log-return**, nie cenę absolutną. Wszystkie metryki liczone są w przestrzeni log-return. Wartości są małe (rząd 0.001–0.01).

### 9.2 Opis metryk

**MAE (Mean Absolute Error):**
```
MAE = mean(|y_pred - y_true|)
```
Średni bezwzględny błąd predykcji log-returnu. MAE = 0.002 → model myli się przeciętnie o 0.2 p.p. w przewidywaniu ruchu ceny.

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(mean((y_pred - y_true)²))
```
Bardziej wrażliwy na duże błędy niż MAE. Wartość zawsze ≥ MAE. Duża różnica RMSE−MAE wskazuje na sporadyczne duże błędy.

**R² (współczynnik determinacji):**
```
R² = 1 - SS_res / SS_tot
```
Mierzy jaką frakcję wariancji targetu wyjaśnia model. Dla rynków finansowych nawet R² = 0.05–0.10 jest dobrym wynikiem (rynki są bliskie losowym).

| R² | Interpretacja |
|---|---|
| > 0.10 | Dobry fit |
| 0.05–0.10 | Umiarkowany fit |
| < 0.05 | Słaby fit (ale uwaga: na rynkach może być ok) |

**Pseudo-Huber Loss (δ=1 i δ=10):**
Metryki odporne na outliery, zachowane dla porównywalności z wynikami modelu XGBoost. RF nie optymalizuje bezpośrednio tej funkcji straty, ale wartości pozwalają zestawić oba modele na tej samej skali.

**Dir Acc (Directional Accuracy):**
```
Dir Acc = mean(sign(y_true) == sign(y_pred)) × 100%
```
Procent przypadków z poprawnie przewidzianym **kierunkiem** ruchu ceny. Kluczowa metryka dla strategii tradingowych.

| Dir Acc | Interpretacja |
|---|---|
| > 55% | Dobry — realna przewaga |
| 50–55% | Niewielka przewaga |
| < 50% | Brak przewagi (gorszy niż losowy) |

**OOB R² (training proxy):**
Obliczany podczas treningu na próbkach Out-Of-Bag (nie widzianych przez dane drzewo). Szybki estymator generalizacji bez osobnego zbioru walidacyjnego. Wartość zwykle niższa niż R² na val — jest bardziej konserwatywna, bliższa realistycznej ocenie.

### 9.3 Przykładowy output testu

```
METRICS SUMMARY
===============

LOG RETURN METRICS:
  MAE:                   0.001754
  RMSE:                  0.002589
  R²:                    0.0682
  Pseudo-Huber (δ=1):    0.001591
  Pseudo-Huber (δ=10):   0.002411

ERROR STATS:
  Mean:   0.000008
  Median: 0.000001
  Std:    0.002589

DIRECTIONAL:
  Accuracy: 53.91%

INTERPRETATION
==============
  [+] Good fit: R² = 0.0682
  [o] Slight edge: 53.9%
```

### 9.4 Feature Importance (MDI)

Random Forest udostępnia jedną miarę ważności cech — **MDI (Mean Decrease in Impurity)**, opartą na `model.feature_importances_`. Jest to bezpośredni wektor bez konieczności remap pozycjonalnego (inaczej niż w XGBoost Booster).

Zapisywana w `feature_importance.json`:
```json
{
  "mdi": {
    "return_1_last": 0.0412,
    "return_1_diff_1": 0.0387,
    ...
  },
  "top20_mdi": { ... }
}
```

Wykres `feature_importance.png` pokazuje top-30 cech wg MDI po spłaszczeniu (np. `return_1_last`, `rsi_mean_48`, `macd_trend_6_24`).

**Uwaga:** MDI jest podatne na bias w kierunku cech o dużej kardynalności (dużo unikalnych wartości). Dla bardziej wiarygodnej analizy ważności można użyć permutation importance (nie jest domyślnie obliczane — wymaga dodatkowego kodu).

---

## 10. Przykłady użycia

### Scenariusz 1: Szybki start z surowym CSV

```bash
# Trening na surowym CSV z domyślną konfiguracją
python rf_train.py --data_path data/eth_1m_2024.csv

# Test (dla CSV trzeba podać test_path ręcznie)
python rf_test.py \
    --model_dir rf_models/R001_eth_rf_model_RAW_0103 \
    --test_path data/eth_1m_2024_test.csv
```

### Scenariusz 2: Pełny pipeline z prepared data

```bash
# Krok 1: Przygotuj dane
cd data_preparation
python prepare_data.py eth_1m_2024.csv \
    --feature_set full \
    --train_pct 70 \
    --test_pct 30 \
    --lookback 48 \
    --horizon 12

# Krok 2: Tuning
cd ../rf_model
python rf_hyperparameter_tuning.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --n_trials 100 \
    --feature_set xgb_momentum \
    --optimize_target val

# Krok 3: Trening z najlepszymi parametrami
python rf_train.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --load_optuna_params rf_data/RO001_rf_optuna_E104_2203/best_params.json

# Krok 4: Test
python rf_test.py \
    --model_dir rf_models/R001_eth_rf_model_E104_2203
```

### Scenariusz 3: Konfiguracja sample weights i misdir penalty

Edytuj `rf_config.py`:
```python
# Ważenie próbek — tryb percentylowy
SAMPLE_WEIGHT_MODE       = 'percentile'
SAMPLE_WEIGHT_PERCENTILE = 70.0   # top 30% dużych ruchów
SAMPLE_WEIGHT_LOW_W      = 0.1    # małe ruchy z wagą 0.1

# Kara za błędny kierunek na dużych ruchach
REWEIGHT_MISDIR     = True
MISDIR_PENALTY      = 2.0    # mnożnik ×2 dla błędnie skierowanych
MISDIR_MIN_QUANTILE = 0.7    # aktywna tylko dla top 30% ruchów
```

Następnie trenuj normalnie:
```bash
python rf_train.py --data_path rf_prepared_data/E104_prep_eth_universal_full_1m
```

### Scenariusz 4: Eksperyment z flatteningiem `signal`

```bash
# Trening z nową metodą signal
python rf_train.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --flattening_method signal \
    --feature_set xgb_momentum

# Tuning z dopuszczeniem wszystkich 4 metod flatteningu
python rf_hyperparameter_tuning.py \
    --data_path rf_prepared_data/E104_prep_eth_universal_full_1m \
    --n_trials 100 \
    --feature_set xgb_momentum
# (signal wchodzi automatycznie do search space flatteningu)
```

### Scenariusz 5: Użycie danych przygotowanych dla XGBoost

Prepared dir XGBoost jest w pełni kompatybilny z modelem RF:
```bash
python rf_train.py \
    --data_path ../xgb_model/prepared_data/E104_prep_eth_universal_full_1m

python rf_test.py \
    --model_dir rf_models/R001_eth_rf_model_E104_2203 \
    --test_path ../xgb_model/prepared_data/E104_prep_eth_universal_full_1m
```

---

## 11. System identyfikatorów

| Prefix | Format | Przykład | Znaczenie |
|---|---|---|---|
| `R` | R001 | R001, R002 | Model ID (w nazwie folderu modelu) |
| `RB` | RB001 | RB001, RB002 | Data ID (prepared dataset w rf_prepared_data) |
| `RO` | RO001 | RO001, RO002 | Optuna tuning ID |
| `RT` | RT001 | RT001, RT002 | Test run ID |

Identyfikatory są automatycznie inkrementowane na podstawie istniejących folderów.

Dane z `data_preparation/` mają własne ID (np. `E104` dla ETH, `B103` dla BTC) i są współdzielone z XGBoost. Prefix `RB` jest zarezerwowany dla danych przygotowanych bezpośrednio w `rf_prepared_data/`.

Przykładowe nazwy folderów:
```
rf_models/R001_eth_rf_model_E104_2203/         ← model R001 na danych E104
rf_data/RO001_rf_optuna_E104_2203/              ← tuning RO001
rf_data/RT001_test_results_R001_E104_2203/      ← test RT001
```

---

## 12. Pliki wyjściowe

### Model (rf_models/R001_.../):

| Plik | Opis |
|---|---|
| `rf_model.pkl` | Wagi modelu RF (joblib) — do wczytania przez `joblib.load()` |
| `model_config.json` | Pełna konfiguracja: features, lookback, horizon, hiperparametry, metryki val, sample weight params, misdir params |
| `feature_scaler.pkl` | Skaler cech (StandardScaler lub RobustScaler) |
| `target_scaler.pkl` | Skaler targetu (jeśli `SCALE_TARGET=True`) |
| `feature_importance.json` | MDI ważność cech: `mdi` (wszystkie), `top20_mdi` |
| `training.log` | Szczegółowy log procesu treningu |
| `summary.txt` | Czytelne podsumowanie: hiperparametry, ważenie próbek, metryki train/val, OOB R² |

### Tuning (rf_data/RO001_.../):

| Plik | Opis |
|---|---|
| `best_params.json` | Najlepsze parametry — bezpośredni input do `--load_optuna_params` |
| `second_best_params.json` | 2. najlepszy zestaw parametrów |
| `third_best_params.json` | 3. najlepszy zestaw parametrów |
| `study.pkl` | Obiekt Optuna Study — do dalszej analizy w Python |
| `trials.csv` | Wszystkie próby z wartościami metryk i parametrów |
| `tuning.log` | Log procesu tuningu |
| `summary.txt` | Podsumowanie top-3 prób z metryką optymalizacji |
| `history.png` | Historia wartości celu przez kolejne próby |
| `importances.png` | Ważność hiperparametrów (Optuna fANOVA) |
| `distribution.png` | Rozkład wartości celu wszystkich prób |
| `convergence.png` | Krzywa zbieżności — najlepszy wynik per próba (cumulative min) |

### Test (rf_data/RT001_.../):

| Plik | Opis |
|---|---|
| `results.csv` | `actual_log`, `predicted_log`, `error` dla każdej próbki |
| `metrics.json` | MAE, RMSE, R², Pseudo-Huber (δ=1), Pseudo-Huber (δ=10), Dir Acc |
| `summary.txt` | Podsumowanie tekstowe z OOB R² (jeśli dostępny) |
| `log_returns_timeseries.png` | Pełna seria log-return: actual (niebieski) vs predicted (czerwony) |
| `log_returns_scatter.png` | Scatter: actual vs predicted log-return |
| `error_distribution.png` | Histogram błędów predykcji |
| `combined_results.png` | Siatka 2×2: seria, scatter, rozkład błędów, rolling dir_acc |
| `feature_importance.png` | Wykres słupkowy top-30 cech wg MDI |
