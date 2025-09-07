import os
import shutil
import csv
from fastapi import UploadFile
from pathlib import Path

class FileService:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent / "shared" / "files"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.binance_raw_path = self.base_path / "binance_historical_raw_data"
        self.binance_raw_path.mkdir(parents=True, exist_ok=True)

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
