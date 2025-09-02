# Zasady rozwijania modułów

Pull requesty, które zawierają kod niezgodny z założoną architekturą, **nie będą mergowane**
do momentu poprawy.

1. Aplikacja lokalnie ma być uruchamiana w środowisku wirtualnym **venv**  
   ```bash
   ./venv/Scripts/activate
   ```

2. Wszystkie paczki muszą być instalowane w **venv**, a następnie przed pushem należy wykonać:
   ```bash
   pip freeze > requirements.txt
   ```

3. Logika modułu ma być umieszczana w klasach (tzw. **Serwisach**).  
   Obiekty serwisów tworzymy w pliku `__init__.py` w katalogu `/services`.

4. Zmienne środowiskowe umieszczamy w pliku `.env` w głównym katalogu modułu.  
   - Plik `.env` **nie jest dodawany do repozytorium**.  
   - Wymiana pliku `.env` odbywa się przez Teamsy/Messengera.

5. Wszystkie parametry w funkcjach/metodach/klasach mają być przekazywane jako zmienne,  
   tak aby można je było wysłać przez **RabbitMQ**.

6. Wszystkie zasoby współdzielone między modułami (datasets, modele, inne pliki)  
   muszą być umieszczane w katalogu:
   ```
   /shared
   ```

7. Każda operacja na plikach (lub innych zasobach) musi być realizowana z użyciem **ścieżek obliczanych dynamicznie w Pythonie**.  
   Funkcja dostaje tylko **nazwę pliku**, a pełna ścieżka jest obliczana w funkcji.  

   **Przykład:**

   ```python
   import os

   def get_shared_file_path(folder: str, filename: str) -> str:
       """
       Zwraca pełną ścieżkę do pliku w katalogu /shared/files/{folder},
       niezależnie czy aplikacja działa lokalnie, czy w kontenerze Dockera.
       """

       # 1. Znajdujemy katalog główny projektu (tam gdzie jest /shared)
       project_root = os.path.dirname(os.path.abspath(__file__))
       while not os.path.exists(os.path.join(project_root, "shared")):
           project_root = os.path.dirname(project_root)

       # 2. Budujemy ścieżkę do pliku
       full_path = os.path.join(project_root, "shared", "files", folder, filename)

       # 3. Tworzymy katalog jeśli nie istnieje
       os.makedirs(os.path.dirname(full_path), exist_ok=True)

       return full_path


   # 🔹 Przykład użycia:
   model_path = get_shared_file_path("models", "ml-model.pkl")
   print(model_path)
   # np. /home/user/project/shared/files/models/ml-model.pkl
   # albo w dockerze /app/shared/files/models/ml-model.pkl
   ```

   ➡️ Dzięki temu jeden moduł może zapisać plik w `/shared/files/models/`,  
   a inny moduł (znając tylko nazwę pliku i folder) może go odczytać.
