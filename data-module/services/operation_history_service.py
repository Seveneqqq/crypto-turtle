import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


class OperationHistoryService:
    def __init__(self, db_path: Optional[str] = None):

        project_root = os.getcwd()
        history_dir = os.path.join(project_root, "history")
        os.makedirs(history_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(history_dir, "operations.db")

        self._lock = threading.Lock()

        self._initialize_database()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self):

        with self._lock:

            conn = self._get_connection()

            try:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS operations_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        service_name TEXT NOT NULL,
                        operation_name TEXT NOT NULL,
                        operation_group TEXT NOT NULL,
                        status TEXT NOT NULL,
                        request_payload TEXT,
                        response_payload TEXT,
                        request_params TEXT,
                        response_summary TEXT,
                        filename TEXT,
                        file_path TEXT,
                        rabbitmq_msg_id TEXT,
                        error_message TEXT,
                        started_at TEXT,
                        finished_at TEXT,
                        duration_ms INTEGER,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_operations_history_service_name
                    ON operations_history(service_name)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_operations_history_operation_name
                    ON operations_history(operation_name)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_operations_history_status
                    ON operations_history(status)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_operations_history_created_at
                    ON operations_history(created_at DESC)
                """)

                conn.commit()

            finally:
                conn.close()

    def save_operation(
        self,
        service_name: str,
        operation_name: str,
        operation_group: str,
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
    ) -> int:

        created_at = datetime.utcnow().isoformat()

        with self._lock:

            conn = self._get_connection()

            try:

                cursor = conn.execute("""
                    INSERT INTO operations_history (
                        service_name,
                        operation_name,
                        operation_group,
                        status,
                        request_payload,
                        response_payload,
                        request_params,
                        response_summary,
                        filename,
                        file_path,
                        rabbitmq_msg_id,
                        error_message,
                        started_at,
                        finished_at,
                        duration_ms,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    service_name,
                    operation_name,
                    operation_group,
                    status,
                    self._to_json(request_payload),
                    self._to_json(response_payload),
                    self._to_json(request_params),
                    self._to_json(response_summary),
                    filename,
                    file_path,
                    rabbitmq_msg_id,
                    error_message,
                    started_at,
                    finished_at,
                    duration_ms,
                    created_at,
                ))

                conn.commit()

                return int(cursor.lastrowid)

            finally:
                conn.close()

    def list_operations(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:

        query = """
            SELECT *
            FROM operations_history
            WHERE 1=1
        """

        params: List[Any] = []

        if service_name:
            query += " AND service_name=?"
            params.append(service_name)

        if operation_name:
            query += " AND operation_name=?"
            params.append(operation_name)

        if status:
            query += " AND status=?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_connection()

        try:

            rows = conn.execute(query, params).fetchall()

            return [self._row_to_dict(r) for r in rows]

        finally:
            conn.close()

    def count_operations(
        self,
        service_name: Optional[str] = None,
        operation_name: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:

        query = "SELECT COUNT(*) as total FROM operations_history WHERE 1=1"
        params = []

        if service_name:
            query += " AND service_name=?"
            params.append(service_name)

        if operation_name:
            query += " AND operation_name=?"
            params.append(operation_name)

        if status:
            query += " AND status=?"
            params.append(status)

        conn = self._get_connection()

        try:

            row = conn.execute(query, params).fetchone()

            return int(row["total"])

        finally:
            conn.close()

    def _row_to_dict(self, row: sqlite3.Row):

        return {
            "id": row["id"],
            "service_name": row["service_name"],
            "operation_name": row["operation_name"],
            "operation_group": row["operation_group"],
            "status": row["status"],
            "request_payload": self._from_json(row["request_payload"]),
            "response_payload": self._from_json(row["response_payload"]),
            "request_params": self._from_json(row["request_params"]),
            "response_summary": self._from_json(row["response_summary"]),
            "filename": row["filename"],
            "file_path": row["file_path"],
            "rabbitmq_msg_id": row["rabbitmq_msg_id"],
            "error_message": row["error_message"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "duration_ms": row["duration_ms"],
            "created_at": row["created_at"],
        }

    def _to_json(self, value):

        if value is None:
            return None

        return json.dumps(value, ensure_ascii=False, default=str)

    def _from_json(self, value):

        if not value:
            return None

        try:
            return json.loads(value)
        except Exception:
            return value