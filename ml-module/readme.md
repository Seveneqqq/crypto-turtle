Moduł uczenia maszynowego

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


# Serwisy sklearn
**sklearn_data_prep_service.py** - tworzy dwa zestawy danych (treningowy i testowy) z wykorzystaniem futures engineering zapisanych w
ml-module/sklearn_model/config/features

pobiera dane z shared/files/binance_historical_raw_data

zapisuje w shared/files/processed_files

POST: 
http://localhost:8000/ml/prepare_features

BODY:

{
  "train_file": "btc_train.csv",

  "test_file": "btc_test.csv",

  "fe_version": "fe_v2",

  "feature_cols": ["open", "close", "high", "low", "volume"]
}

**sklearn_test_service.py** - testuje wytrenowany model w shared/models/sklearn
wyniki zapisywane są jako logi w shared/models/test_logs

POST:
http://localhost:8000/ml/test_model


BODY:

{
  "model_filename": "btc_model_simple_c_v1_2209_1.pkl",

  "test_filename": "test_20250922_214643.csv"
}


**sklearn_train_service.py** - trenuje model na danych z shared/files/processed_files i zapisuje do shared/models/sklearn

POST:
http://localhost:8000/ml/train

BODY:
{
  "df_filename": "train_20250922_214643.csv",

  "clf_version": "c_v1",

  "feature_cols": ["ma_5","ma_10","ma_20","sma_10","sma_30","stoch_k","stoch_d","macd","macd_signal","macd_diff","adx","atr","std_20","cumvol","volume_norm","momentum_10"]
}