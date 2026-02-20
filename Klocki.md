# Przygotowanie modułów w formie klocków pod plgrid
### Cel: Wywoływanie serwisów lokalnie i w plgridzie bez większych zmian w kodzie. Wywołanie poza strukturą konkretnego modułu. 
### Jakby to powiedział Paweł - gateway bieda edition.
### * wywołanie przez dockera będzie działało dalej tak samo
## 1. Plik ```module_loader.py ```
- pominięcie inita (inicjalizacji rabbita)
```
import importlib
import os

os.environ["SKIP_INIT"] = "1"
base_dir = os.path.dirname(os.path.abspath(__file__))
```
- funkcja ładująca serwis
```
def load(name: str, attr: str):
    module = importlib.import_module(name)
    return getattr(module, attr)
```
- funkcje wywołania serwisów
```
def block_train_ppo():
    TrainAgentPPOService = load(
        "rl-module.services.train_agent_ppo_service",
        "TrainAgentPPOService"
    )
    service = TrainAgentPPOService(base_dir)
    return service

def block_simulation():
    pass

def block_visualisation():
    pass
```
### base_dir jest tak na prawdę adresem shared, więc w wersji na plgrida należy podać adres scratch: `/net/afscra/people/[nick]`

## 2. Dostosowanie inita modułu:
- plik `init.py` w twoim module
- na początku należy dodać warunek: 
```
- if os.environ.get("SKIP_INIT") == "1":
    pass
else:
    # inicjalizacja serwisów itd.
```
### Ten warunek odróżnia wywołanie lokalnie/plgrid od wywołania przez dockera, przy wywołaniu przez dockera init się wykona.

## 3. Dostosowanie serwisów:
```
class TrainAgentPPOService:
    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
```
### jeśli nie jest podany base_dir -> wywołanie dockerowe - bez zmian
### jeśli jest podany -> wywołanie lokalne/plgrid
### jeśli macie jakieś inne pliki `.py` dołączane do serwisu, trzeba będzie przekazywać do nich base_dir

# Jeśli działa lokalnie - zadziała na plgridzie (z podmienionym adresem na scratch)
# Scratch: `"/net/afscra/people/plgniemyslniepytaj"` - twoja nazwa użytkownika