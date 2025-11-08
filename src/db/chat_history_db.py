# -*- coding: utf-8 -*-
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
