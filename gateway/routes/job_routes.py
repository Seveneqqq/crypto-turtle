from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict, Optional
from services import job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/run")
async def run_job(request: Request):
    try:
        body: Dict[str, Any] = await request.json()
        action: Optional[str] = body.get("action")
        payload: Any = body.get("payload")
        host: Optional[str] = body.get("host")

        if not action:
            raise HTTPException(status_code=400, detail="Field 'action' is required")

        result = job_service.handle_action(host=host, action=action, payload=payload)

        
        print("[SSH] STDOUT:\n")
        print(result["stdout"])
        print("[SSH] STDERR:\n")
        print(result["stderr"])

        print("ACTION:", action)
        print("PAYLOAD:", payload)

        return {
            "status": "ok",
            "action": action,
            "payload": payload,
            "stdout": result["stdout"],  
            "stderr": result["stderr"]
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
