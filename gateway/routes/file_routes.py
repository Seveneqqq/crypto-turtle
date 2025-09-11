from fastapi import APIRouter, File, UploadFile, Depends
from services import auth_service
from services import file_service

router = APIRouter(prefix="/file", tags=["file"])


@router.get("/private")
def private_data(user=Depends(auth_service.require_auth)):
    return {"msg": f"Witaj w data, {user.user_metadata.get('name')}"}


@router.post("")
def upload_file(
    user=Depends(auth_service.require_auth),
    file: UploadFile = File(...)
):
    return file_service.upload_file(file)


@router.get("")
def get_files(user=Depends(auth_service.require_auth)):
    return file_service.get_all_files()

@router.get("/models")
def get_models_files(user=Depends(auth_service.require_auth)):
    return file_service.get_models()

@router.get("/simulation-results")
def get_simulation_results(user=Depends(auth_service.require_auth)):
    return file_service.get_simulation_results()

@router.delete("/{filename}")
def delete_file(
    user=Depends(auth_service.require_auth),
    filename: str = None
):
    return file_service.delete_file(filename)

@router.get("/binance")
def get_binance_files(user=Depends(auth_service.require_auth)):
    """Zwraca listę plików z katalogu binance_historical_raw_data"""
    return file_service.get_binance_raw_files()


@router.get("/binance/{filename}")
def get_binance_file_preview(
    filename: str,
    user=Depends(auth_service.require_auth)
):
    """Zwraca podgląd (pierwsze 100 wierszy) i liczbę wszystkich wierszy"""
    return file_service.read_file_preview(filename)
