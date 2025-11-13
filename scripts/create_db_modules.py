# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# documents_db.py 내용
documents_db_code = """# -*- coding: utf-8 -*-
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class DocumentsDB:
    def __init__(self, db_path: str = 'data/documents.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_info (
                    file_hash TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    total_pages INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    total_chars INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS page_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    markdown_content TEXT,
                    token_count INTEGER,
                    is_empty BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_hash) REFERENCES file_info(file_hash) ON DELETE CASCADE,
                    UNIQUE(file_hash, page_number)
                )
            """)
            conn.commit()

    def insert_file_info(self, file_hash, file_name, total_pages, file_size, total_chars, total_tokens):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_info
                (file_hash, file_name, total_pages, file_size, total_chars, total_tokens, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_hash, file_name, total_pages, file_size, total_chars, total_tokens, datetime.now()))
            conn.commit()
            return True

    def insert_page_data(self, file_hash, page_number, markdown_content, token_count, is_empty=False):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO page_data
                (file_hash, page_number, markdown_content, token_count, is_empty)
                VALUES (?, ?, ?, ?, ?)
            """, (file_hash, page_number, markdown_content, token_count, is_empty))
            conn.commit()
            return True

    def get_document_stats(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM file_info')
            total_files = cursor.fetchone()[0]
            cursor.execute('SELECT SUM(total_pages) FROM file_info')
            total_pages = cursor.fetchone()[0] or 0
            cursor.execute('SELECT SUM(total_tokens) FROM file_info')
            total_tokens = cursor.fetchone()[0] or 0
            cursor.execute('SELECT SUM(file_size) FROM file_info')
            total_size = cursor.fetchone()[0] or 0
            return {
                'total_files': total_files,
                'total_pages': total_pages,
                'total_tokens': total_tokens,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
            }
"""

# embeddings_db.py 내용
embeddings_db_code = """# -*- coding: utf-8 -*-
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class EmbeddingsDB:
    def __init__(self, db_path: str = 'data/embeddings.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_meta (
                    embedding_hash TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    preprocessing_option TEXT,
                    embedding_model TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    vector_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_mapping (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_hash TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    start_page INTEGER,
                    end_page INTEGER,
                    chunk_text TEXT NOT NULL,
                    estimated_tokens INTEGER,
                    vector_index INTEGER NOT NULL,
                    FOREIGN KEY (embedding_hash) REFERENCES embedding_meta(embedding_hash) ON DELETE CASCADE
                )
            """)
            conn.commit()
"""

# chat_history_db.py 내용
chat_history_db_code = """# -*- coding: utf-8 -*-
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class ChatHistoryDB:
    def __init__(self, db_path: str = 'data/chat_history.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    retrieved_chunks TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def create_session(self, session_name=None):
        session_id = str(uuid.uuid4())
        if session_name is None:
            session_name = f"채팅 세션 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_sessions (session_id, session_name)
                VALUES (?, ?)
            """, (session_id, session_name))
            conn.commit()
            return session_id

    def add_message(self, session_id, role, content, retrieved_chunks=None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            chunks_json = None
            if retrieved_chunks:
                chunks_json = json.dumps(retrieved_chunks, ensure_ascii=False)
            cursor.execute("""
                INSERT INTO chat_messages
                (session_id, role, content, retrieved_chunks)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, chunks_json))
            cursor.execute("""
                UPDATE chat_sessions
                SET updated_at = ?
                WHERE session_id = ?
            """, (datetime.now(), session_id))
            conn.commit()
            return cursor.lastrowid

    def get_chat_stats(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM chat_sessions')
            total_sessions = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE is_active = 1')
            active_sessions = cursor.fetchone()[0]
            cursor.execute('SELECT COUNT(*) FROM chat_messages')
            total_messages = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE role = 'user'")
            user_messages = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chat_messages WHERE role = 'assistant'")
            assistant_messages = cursor.fetchone()[0]
            return {
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'total_messages': total_messages,
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
            }
"""

# 파일 생성
project_root = Path(__file__).parent.parent
db_dir = project_root / 'src' / 'db'

with open(db_dir / 'documents_db.py', 'w', encoding='utf-8') as f:
    f.write(documents_db_code)
print('documents_db.py created')

with open(db_dir / 'embeddings_db.py', 'w', encoding='utf-8') as f:
    f.write(embeddings_db_code)
print('embeddings_db.py created')

with open(db_dir / 'chat_history_db.py', 'w', encoding='utf-8') as f:
    f.write(chat_history_db_code)
print('chat_history_db.py created')

print('All DB modules created successfully!')
