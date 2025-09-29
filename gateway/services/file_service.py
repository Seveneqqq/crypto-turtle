from collections import defaultdict
import os
import shutil
import csv
from fastapi import UploadFile
from pathlib import Path

class FileService:

    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent / "shared" / "files"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.binance_raw_path = self.base_path / "raw_data"
        self.binance_raw_path.mkdir(parents=True, exist_ok=True)

        self.project_root = Path(__file__).resolve().parent.parent  
        self.shared_path = self.project_root / "shared"
        self.plots_path = self.shared_path / "simulation" / "plots"

        self.models_path = self.shared_path / "models" / "ml"
        self.logs_path = self.shared_path / "simulation" / "logs"

    # multipart/form-data -> body multipart->file
    def upload_file(self, file: UploadFile):
        save_path = self.base_path / file.filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "filepath": str(save_path)
        }

    def get_files_by_key(self, key: str):
        """
        Zwraca pliki na podstawie klucza, który jest mapowany na konkretną ścieżkę.
        Działa zarówno w Dockerze, jak i lokalnie.
        """

        # mapa kluczy na foldery
        key_to_path = {
            "models": self.models_path,
            "logs": self.logs_path,
            "plots": self.plots_path,
            "binance_raw": self.binance_raw_path,
            "files": self.base_path,
            "agents_scikit": self.shared_path / "files" / "agents" / "scikit",
            "raw_data": self.shared_path / "files" / "raw_data",
            "processed_rlagent_data": self.shared_path / "files" / "processed_rlagent_data",
            "processed_simulation_data": self.shared_path / "files" / "processed_simulation_data",
            "processed_testmodel_data": self.shared_path / "files" / "processed_testmodel_data",
            "processed_trainmodel_data": self.shared_path / "files" / "processed_trainmodel_data",
            "models_ml": self.shared_path / "models" / "ml",
            "models_sklearn": self.shared_path / "models" / "sklearn",
        }

        if key not in key_to_path:
            return {"message": f"Unknown key '{key}'"}

        path = key_to_path[key]

        if not path.exists() or not path.is_dir():
            return {"message": f"Path for key '{key}' not found"}

        files = []
        for filename in os.listdir(path):
            file_path = path / filename
            if file_path.is_file():
                files.append({
                    "name": filename,
                    "path": str(file_path)
                })

        return {
            "key": key,
            "directory": str(path),
            "files": files
        }

    def get_models(self):
        """Zwraca wszystkie modele"""
        models = []
        for filename in os.listdir(self.models_path):
            file_path = self.models_path / filename
            if file_path.is_file():
                models.append({
                    "name": filename,
                    "path": str(file_path)
                })
        return models

    def get_simulation_results(self):
        """Zwraca sparowane logi i wykresy"""
        results = defaultdict(lambda: {"log": None, "plot": None})

        def extract_key(filename: str):
            parts = filename.split("_")
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}" 
            return None

        for filename in os.listdir(self.logs_path):
            key = extract_key(filename)
            if key:
                results[key]["log"] = str(self.logs_path / filename)

        # wykresy
        for filename in os.listdir(self.plots_path):
            key = extract_key(filename)
            if key:
                results[key]["plot"] = str(self.plots_path / filename)

        return [
            {
                "key": key,
                "log": files["log"],
                "plot": files["plot"]
            }
            for key, files in results.items()
            if files["log"] or files["plot"]
        ]

    def get_all_files(self):
        files = []
        for root, dirs, filenames in os.walk(self.base_path):
            for filename in filenames:
                files.append({
                    "name": filename,
                    "path": str(Path(root) / filename)
                })
        return files

    def delete_file(self, filename: str):
        file_path = self.base_path / filename
        if file_path.exists():
            file_path.unlink()
            return {"message": "File deleted successfully"}
        else:
            return {"message": "File not found"}

    def get_binance_raw_files(self):     
        files = []
        for filename in os.listdir(self.binance_raw_path):
            file_path = self.binance_raw_path / filename
            if file_path.is_file():
                files.append({
                    "name": filename,
                    "path": str(file_path)
                })
        return files

    def read_file_preview(self, filename: str, limit: int = 100):
        file_path = self.binance_raw_path / filename
        if not file_path.exists():
            return {"message": "File not found"}

        preview_rows = []
        total_rows = 0

        with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)  
            for i, row in enumerate(reader):
                if i < limit:
                    preview_rows.append(row)
                total_rows += 1

        return {
            "filename": filename,
            "preview": preview_rows,
            "total_rows": total_rows,
            "header": header
        }
