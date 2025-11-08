# -*- coding: utf-8 -*-
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class DocumentsDB:
    """
    문서 데이터베이스 관리 클래스

    주요 기능:
    - 파일 정보 및 페이지 데이터를 SQLite 데이터베이스에 저장
    - 데이터베이스 테이블 생성 및 관리
    - 문서 통계 정보 제공

    사용 예:
    db = DocumentsDB("data/documents.db")
    db.insert_file_info(file_hash, file_name, total_pages, file_size, total_chars, total_tokens)
    stats = db.get_document_stats()
    """

    def __init__(self, db_path: str = 'data/documents.db'):
        """
        DocumentsDB 초기화

        Args:
            db_path (str): 데이터베이스 파일 경로 (기본값: 'data/documents.db')
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)  # 데이터베이스 경로 생성
        self._create_tables()  # 테이블 생성

    def _get_connection(self):
        """
        SQLite 데이터베이스 연결 생성

        Returns:
            sqlite3.Connection: 데이터베이스 연결 객체
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 결과를 딕셔너리 형태로 반환
        return conn

    def _create_tables(self):
        """
        데이터베이스 테이블 생성

        - file_info: 파일 정보 저장
        - page_data: 페이지별 데이터 저장
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 파일 정보 테이블 생성
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
            # 페이지 데이터 테이블 생성
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

    def insert_file_info(self, file_hash: str, file_name: str, total_pages: int, file_size: int, 
                         total_chars: int, total_tokens: int) -> bool:
        """
        파일 정보를 데이터베이스에 저장

        Args:
            file_hash (str): 파일의 해시값
            file_name (str): 파일 이름
            total_pages (int): 총 페이지 수
            file_size (int): 파일 크기 (바이트 단위)
            total_chars (int): 총 문자 수
            total_tokens (int): 총 토큰 수

        Returns:
            bool: 저장 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO file_info
                (file_hash, file_name, total_pages, file_size, total_chars, total_tokens, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_hash, file_name, total_pages, file_size, total_chars, total_tokens, datetime.now()))
            conn.commit()
            return True

    def insert_page_data(self, file_hash: str, page_number: int, markdown_content: str, 
                         token_count: int, is_empty: bool = False) -> bool:
        """
        페이지 데이터를 데이터베이스에 저장

        Args:
            file_hash (str): 파일의 해시값
            page_number (int): 페이지 번호
            markdown_content (str): 페이지의 Markdown 콘텐츠
            token_count (int): 페이지의 토큰 수
            is_empty (bool): 페이지가 비어 있는지 여부 (기본값: False)

        Returns:
            bool: 저장 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO page_data
                (file_hash, page_number, markdown_content, token_count, is_empty)
                VALUES (?, ?, ?, ?, ?)
            """, (file_hash, page_number, markdown_content, token_count, is_empty))
            conn.commit()
            return True

    def get_document_stats(self) -> Dict[str, Any]:
        """
        데이터베이스에 저장된 문서 통계 정보를 반환

        Returns:
            Dict[str, Any]: 문서 통계 정보
                - total_files: 총 파일 수
                - total_pages: 총 페이지 수
                - total_tokens: 총 토큰 수
                - total_size_bytes: 총 파일 크기 (바이트 단위)
                - total_size_mb: 총 파일 크기 (MB 단위)
        """
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