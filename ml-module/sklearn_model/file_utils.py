import os
from typing import List

def get_project_root() -> str:
    """
    Zwraca katalog główny projektu (tam gdzie jest folder /shared).
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(project_root, "shared")):
        project_root = os.path.dirname(project_root)
    return project_root

def get_model_path(relative_path: str) -> str:
    """
    Zwraca pełną ścieżkę do pliku w /shared/models/{relative_path}.
    Tworzy foldery jeśli nie istnieją.
    relative_path może zawierać podfoldery, np. "test/btc_model.pkl"
    """
    project_root = get_project_root()
    full_path = os.path.join(project_root, "shared", "models", relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def get_file_path(relative_path: str) -> str:
    """
    Zwraca pełną ścieżkę do pliku w /shared/files/{relative_path}.
    Tworzy foldery jeśli nie istnieją.
    relative_path może zawierać podfoldery, np. "train/btc_train.csv"
    """
    project_root = get_project_root()
    full_path = os.path.join(project_root, "shared", "files", relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def list_existing_models(base_name: str, subfolder: str = "") -> List[str]:
    """
    Zwraca listę istniejących modeli w katalogu shared/models[/subfolder] zaczynających się od base_name.
    """
    folder = get_model_path(subfolder)
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.startswith(base_name)]
