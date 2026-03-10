import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional


class OperationFileLoggerService:

    def __init__(self):

        project_root = os.getcwd()

        history_dir = os.path.join(project_root, "history")
        logs_dir = os.path.join(history_dir, "logs")

        os.makedirs(logs_dir, exist_ok=True)

        filename = f"operations_{datetime.utcnow().strftime('%Y%m%d')}.log"

        self.log_path = os.path.join(logs_dir, filename)

        self.logger = logging.Logger(f"OperationFileLogger_{id(self)}")

        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(self.log_path, encoding="utf-8")

        handler.setFormatter(logging.Formatter("%(message)s"))

        self.logger.addHandler(handler)

    def log(
        self,
        service_name: str,
        operation_name: str,
        status: str,
        request_payload: Optional[Dict[str, Any]] = None,
        response_payload: Optional[Dict[str, Any]] = None,
        request_params: Optional[Dict[str, Any]] = None,
        response_summary: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        file_path: Optional[str] = None,
        rabbitmq_msg_id: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ):

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": service_name,
            "operation_name": operation_name,
            "status": status,
            "request_payload": request_payload,
            "response_payload": response_payload,
            "request_params": request_params,
            "response_summary": response_summary,
            "filename": filename,
            "file_path": file_path,
            "rabbitmq_msg_id": rabbitmq_msg_id,
            "error_message": error_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
        }

        self.logger.info(json.dumps(payload, ensure_ascii=False, default=str))