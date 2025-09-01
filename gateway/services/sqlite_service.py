import sqlite3
import os


class SQLiteService:
    def __init__(self, db_filename="rabbitmq_messages_db.db"):
        print("Initializing SQLiteService...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_dir = os.path.join(base_dir, "db")
        os.makedirs(db_dir, exist_ok=True)
    
        db_path = os.path.join(db_dir, db_filename)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rabbit_msg_id TEXT NOT NULL UNIQUE,
            state TEXT NOT NULL CHECK(state IN ('created','in progress','failed','success')),
            request TEXT NOT NULL,
            response TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT (datetime('now','utc')),
            edited_at TIMESTAMP
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_message(self, rabbit_msg_id: str, state: str, request: str, response: str = None):
        query = """
        INSERT INTO messages (rabbit_msg_id, state, request, response)
        VALUES (?, ?, ?, ?)
        """
        self.conn.execute(query, (rabbit_msg_id, state, request, response))
        self.conn.commit()

    def update_message(self, rabbit_msg_id: str, state: str, response: str = None):
        query = """
        UPDATE messages
        SET state = ?,
            response = ?,
            edited_at = datetime('now','utc')
        WHERE rabbit_msg_id = ?
        """
        self.conn.execute(query, (state, response, rabbit_msg_id))
        self.conn.commit()

    def get_all_messages(self):
        cursor = self.conn.execute("SELECT * FROM messages ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def filter_messages(self, **filters):
        """
        Filtrowanie po dowolnych polach:
        np. filter_messages(state="failed", rabbit_msg_id="123")
        """
        query = "SELECT * FROM messages"
        conditions = []
        values = []

        for field, value in filters.items():
            if value is not None:
                conditions.append(f"{field} = ?")
                values.append(value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        cursor = self.conn.execute(query, values)
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()
