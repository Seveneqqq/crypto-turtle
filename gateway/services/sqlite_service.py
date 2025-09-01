import sqlite3
import os
import uuid

class SQLiteService:
    def __init__(self, db_filename="rabbitmq_messages_db.db"):
        try:
            print("Initializing SQLiteService...")
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(base_dir, "db")
            os.makedirs(db_dir, exist_ok=True)
            print(f"Database directory: {db_dir}")

            db_path = os.path.join(db_dir, db_filename)
            print(f"Database path: {db_path}")
            
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            print("SQLite connection established successfully")
            
            self.create_tables()
            print("SQLiteService initialization completed")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize SQLiteService: {e}")
            import traceback
            traceback.print_exc()
            raise

    def create_tables(self):
        try:
            print("Creating tables...")
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
            print("Tables created/verified successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to create tables: {e}")
            import traceback
            traceback.print_exc()
            raise

    def insert_message(self, rabbit_msg_id: str, state: str, request: str, response: str = None):
        try:
            print(f"Inserting message: {rabbit_msg_id} with state: {state}")
            query = """
            INSERT INTO messages (rabbit_msg_id, state, request, response)
            VALUES (?, ?, ?, ?)
            """
            self.conn.execute(query, (rabbit_msg_id, state, request, response))
            self.conn.commit()
            print(f"Message {rabbit_msg_id} inserted successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to insert message {rabbit_msg_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def update_message(self, rabbit_msg_id: str, state: str, response: str = None):
        try:
            print(f"Updating message: {rabbit_msg_id} with state: {state}")
            query = """
            UPDATE messages
            SET state = ?,
                response = ?,
                edited_at = datetime('now','utc')
            WHERE rabbit_msg_id = ?
            """
            print(f"Executing query: {query}")
            print(f"Parameters: state={state}, response={response}, rabbit_msg_id={rabbit_msg_id}")
            
            cursor = self.conn.execute(query, (state, response, rabbit_msg_id))
            rows_affected = cursor.rowcount
            print(f"Rows affected: {rows_affected}")
            
            if rows_affected == 0:
                print(f"WARNING: No rows updated for rabbit_msg_id: {rabbit_msg_id}")
                cursor = self.conn.execute("SELECT 1 FROM messages WHERE rabbit_msg_id = ?", (rabbit_msg_id,))
                if cursor.fetchone() is None:
                    print(f"ERROR: Message with rabbit_msg_id {rabbit_msg_id} does not exist!")
                else:
                    print(f"Message exists but update failed for unknown reason")
            
            self.conn.commit()
            print(f"Message {rabbit_msg_id} updated successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to update message {rabbit_msg_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_all_messages(self):
        try:
            print("Fetching all messages...")
            cursor = self.conn.execute("SELECT * FROM messages ORDER BY created_at DESC")
            messages = [dict(row) for row in cursor.fetchall()]
            print(f"Retrieved {len(messages)} messages")
            return messages
            
        except Exception as e:
            print(f"ERROR: Failed to get all messages: {e}")
            import traceback
            traceback.print_exc()
            raise

    def filter_messages(self, **filters):
        try:
            print(f"Filtering messages with filters: {filters}")
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
            print(f"Executing query: {query}")
            print(f"Parameters: {values}")

            cursor = self.conn.execute(query, values)
            messages = [dict(row) for row in cursor.fetchall()]
            print(f"Retrieved {len(messages)} filtered messages")
            return messages
            
        except Exception as e:
            print(f"ERROR: Failed to filter messages: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_unique_rabbit_msg_id(self):
        try:
            print("Generating unique rabbit_msg_id...")
            attempts = 0
            while attempts < 10:  
                attempts += 1
                new_id = str(uuid.uuid4())
                cursor = self.conn.execute(
                    "SELECT 1 FROM messages WHERE rabbit_msg_id = ?", (new_id,)
                )
                if cursor.fetchone() is None:
                    print(f"Generated unique rabbit_msg_id: {new_id} (attempts: {attempts})")
                    return new_id
                print(f"Generated duplicate ID, retrying... (attempt {attempts})")
            
            print(f"ERROR: Failed to generate unique ID after {attempts} attempts")
            raise Exception("Could not generate unique rabbit_msg_id")
            
        except Exception as e:
            print(f"ERROR: Failed to generate unique rabbit_msg_id: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self):
        try:
            print("Closing SQLite connection...")
            self.conn.close()
            print("SQLite connection closed successfully")
            
        except Exception as e:
            print(f"ERROR: Failed to close SQLite connection: {e}")
            import traceback
            traceback.print_exc()