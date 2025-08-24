

import os
import shutil

from fastapi import UploadFile


class DataService:
    
    
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