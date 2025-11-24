# -*- coding: utf-8 -*-
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class ChatHistoryDB:
    """
    ChatHistoryDB는 SQLite를 사용하여 채팅 세션 및 메시지를 관리하는 데이터베이스를 제공합니다.
    주요 기능:
    - 채팅 세션 생성 및 관리
    - 채팅 메시지 추가 및 조회
    - 채팅 통계 정보 제공
    """

    def __init__(self, db_path: str = 'data/chat_history.db'):
        """
        데이터베이스 초기화 및 테이블 생성.
        :param db_path: 데이터베이스 파일 경로 (기본값: 'data/chat_history.db')
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        """
        SQLite 데이터베이스 연결 객체를 생성.
        :return: SQLite 연결 객체
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """
        필요한 데이터베이스 테이블(chat_sessions, chat_messages)을 생성.
        """
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

    def create_session(self, session_name: Optional[str] = None) -> str:
        """
        새로운 채팅 세션을 생성.
        :param session_name: 세션 이름 (기본값: 현재 시간 기반 자동 생성)
        :return: 생성된 세션 ID
        """
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

    def add_message(self, session_id: str, role: str, content: str, 
                    retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        특정 세션에 메시지를 추가.
        :param session_id: 메시지를 추가할 세션 ID
        :param role: 메시지의 역할 ('user' 또는 'assistant')
        :param content: 메시지 내용
        :param retrieved_chunks: 검색된 청크 데이터 (JSON 형식)
        :return: 추가된 메시지의 ID
        """
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

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        모든 세션 목록을 반환 (session_id, session_name, created_at, updated_at, is_active)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, session_name, created_at, updated_at, is_active
                FROM chat_sessions
                ORDER BY updated_at DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        특정 세션의 모든 메시지 목록을 반환
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, role, content, retrieved_chunks, timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, message_id ASC
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_chat_stats(self) -> Dict[str, int]:
        """
        채팅 데이터베이스의 통계 정보를 반환.
        :return: 총 세션 수, 활성 세션 수, 총 메시지 수, 사용자 메시지 수, 어시스턴트 메시지 수
        """
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
                'assistant_messages': assistant_messages
            }

    def delete_session(self, session_id: str) -> bool:
        """
        특정 세션과 관련된 모든 메시지를 삭제.
        :param session_id: 삭제할 세션 ID
        :return: 삭제 성공 여부
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 외래 키 제약으로 인해 메시지가 자동 삭제됨 (ON DELETE CASCADE)
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"세션 삭제 중 오류 발생: {e}")
            return False