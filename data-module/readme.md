Moduł danych

# .env

Zapisujemy wszystkie zmienne środowiskowe w .env w formacie np: 

```
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
```

Zmienne potem są ładowane odpowiednio rabbitmq_configuration -> zmienne rabbita
configuration -> wszystkie pozostałe

# /configuration

Tutaj ładowane są zmienne z .env 

# /services
Folder zawierajacy serwisy (klasy grupujace logike, **klasą sterującą jest rabbitmq_service**)

**__init__** - Klasa główna w której tworzone są instancje, są dzięki temu łatwo dostepne z kazdego miejsca\ 

**rabbitmq_service** - Klasa sterująca która będzie wywoływać odpowiednie serwisy i metody
(Gdy przyjdzie wiadomość np: trenuj dane to ta klasa ją obsłuży i wywoła wasz kod który wykona tą operację)


**main.py** - plik w głównym folderze modułu, tutaj będą importowane endpointy i jest cały przepływ aplikacji.
**Reszta struktury wedle uznania ale zachęcam do zachowania czystej i przejrzystej struktury**

//-----------------------------------------------------------------------------------------
Surowe pliki pobrane z binance są zapisywanie w ../shared/files/binance_historical_raw_data
Dane są pobierane w czystej postaci i zapisywane w csv. Gdy plik zostanie zapisany wtedy inne serwisy mogą
otworzyc ten plik i zrobic z niego data frame.

1. przerobic generowanie dla lstm i sklearn zeby dalo sie wybrac feature (kombinacje)

http://localhost:8000/data/process-data-lstm
{
  "filename": "short_test.csv",
  "steps": ["load_data"],
  "preprocessing_method": "RAW",
  "output_folders": ["processed_trainmodel_data_lstm"],
  "technical_indicators_set": "set1",
  "selected_features": ["ma_5", "ma_10", "ma_20", "sma_10"]
}

http://localhost:8000/data/process-data
{
  "filename": "short_test.csv",
  "steps": ["load_data", "make_features", "add_technical_indicators"],
  "preprocessing_method": "RAW",
  "output_folders": ["processed_trainmodel_data"],
  "horizon": 5,
  "feature_sets": [
    ["ret1", "ret1_lag1", "RSI_14"],
    ["macd", "macd_sig", "EMA_20"]
  ]
}

http://localhost:8000/data/tune-data-lstm
{
  "action": "tune_lstm",
  "filename_candidates": [
    "short_test.csv"
  ],
  "steps": ["load_data"],
  "n_trials": 5,

  "technical_sets": ["combinations"],

  "preprocessing_methods": ["RAW"],
  "target_col": "close",

  "seq_len_range": [32, 32, 1],
  "scaler_types": ["standard", "minmax"],

  "lstm_units_range": [32, 32, 1],
  "lstm_layers_range": [1, 1],
  "dropout_range": [0.1, 0.1, 0.1],
  "lr_range": [0.001, 0.001],

  "batch_sizes": [32],
  "epochs": 1,
	"max_features_in_combo": 5,
  "top_k_combinations": 5
}

http://localhost:8000/data/tune-data
{
  "action": "tune_data",
  "filename_candidates": ["short_test.csv"],
  "steps": ["load_data", "make_features", "add_technical_indicators"],
  "n_trials": 50,
  "technical_sets": ["combinations"],
  "preprocessing_methods": ["RAW", "Z-SCORE", "ROBUST", "MINMAX"],
  "output_folders": ["processed_trainmodel_data"],   
  "metrics": ["mse"],                 
  "max_features_in_combo": 2,
  "top_k_combinations": 50
}
