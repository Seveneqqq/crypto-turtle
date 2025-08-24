import os
import shutil
from fastapi import UploadFile

class FileService:
    
    #multipart/form-data
    #body multipart->file
    def upload_file(self, file: UploadFile):
        save_path = f"../shared/files/{file.filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "filepath": f"/shared/files/{file.filename}"
        }

    def get_all_files(self):
        files = []
        for root, dirs, filenames in os.walk("../shared/files"):
            for filename in filenames:
                files.append({
                    "name": filename,
                    "path": f"/shared/files/{filename}"
                })
        return files
    
    def delete_file(self, filename: str):
        file_path = f"../shared/files/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": "File deleted successfully"}
        else:
            return {"message": "File not found"}