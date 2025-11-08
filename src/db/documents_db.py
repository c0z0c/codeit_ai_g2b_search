# -*- coding: utf-8 -*-
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
