from fastapi import APIRouter, File, Request, UploadFile
from fastapi.params import Depends

router = APIRouter(prefix="/ml", tags=["ml"])

#endpointy ktore wywolaja odpowiednia metode z danego serwisu i przekaze wiadomosc do kolejki