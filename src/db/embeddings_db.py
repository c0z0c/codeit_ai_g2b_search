# -*- coding: utf-8 -*-
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
                    faiss_index_path TEXT,
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
