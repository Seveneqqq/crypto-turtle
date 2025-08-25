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