# -*- coding: utf-8 -*-
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class EmbeddingsDB:
    """
    EmbeddingsDB 클래스는 임베딩 메타데이터와 텍스트 청크 데이터를 저장 및 관리하기 위한 SQLite 데이터베이스를 다룹니다.
    
    주요 기능:
    - 데이터베이스 파일 경로를 설정하고, 필요한 경우 디렉토리를 생성합니다.
    - 임베딩 메타데이터와 텍스트 청크 데이터를 저장하기 위한 테이블을 생성합니다.
    - SQLite 연결을 관리합니다.
    
    Attributes:
        db_path (Path): 데이터베이스 파일 경로 (기본값: 'data/embeddings.db').
    """
    
    def __init__(self, db_path: str = 'data/embeddings.db'):
        """
        EmbeddingsDB 객체를 초기화합니다.
        
        Args:
            db_path (str): 데이터베이스 파일 경로. 기본값은 'data/embeddings.db'입니다.
        """
        self.db_path = Path(db_path)
        # 데이터베이스 디렉토리가 존재하지 않으면 생성
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 테이블 생성
        self._create_tables()

    def _get_connection(self):
        """
        SQLite 데이터베이스 연결을 생성합니다.
        
        Returns:
            sqlite3.Connection: SQLite 연결 객체.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 결과를 딕셔너리 형태로 반환
        return conn

    def _create_tables(self):
        """
        데이터베이스에 필요한 테이블을 생성합니다.
        - embedding_meta: 임베딩 메타데이터를 저장하는 테이블.
        - chunk_mapping: 텍스트 청크 데이터를 저장하는 테이블.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 임베딩 메타데이터 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_meta (
                    embedding_hash TEXT PRIMARY KEY,  -- 임베딩 해시 (고유 식별자)
                    file_hash TEXT NOT NULL,          -- 파일 해시
                    chunk_size INTEGER NOT NULL,      -- 청크 크기
                    chunk_overlap INTEGER NOT NULL,   -- 청크 중첩 크기
                    preprocessing_option TEXT,        -- 전처리 옵션
                    embedding_model TEXT NOT NULL,    -- 임베딩 모델 이름
                    total_chunks INTEGER NOT NULL,    -- 총 청크 개수
                    faiss_index_path TEXT,            -- FAISS 인덱스 경로
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 생성 시간
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- 수정 시간
                )
            """)
            # 텍스트 청크 매핑 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_mapping (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 청크 ID (자동 증가)
                    embedding_hash TEXT NOT NULL,                -- 임베딩 해시 (외래 키)
                    file_hash TEXT NOT NULL,                     -- 파일 해시
                    file_name TEXT NOT NULL,                     -- 파일 이름
                    start_page INTEGER,                          -- 시작 페이지
                    end_page INTEGER,                            -- 종료 페이지
                    chunk_text TEXT NOT NULL,                    -- 청크 텍스트
                    estimated_tokens INTEGER,                    -- 추정 토큰 수
                    vector_index INTEGER NOT NULL,               -- 벡터 인덱스
                    FOREIGN KEY (embedding_hash) REFERENCES embedding_meta(embedding_hash) ON DELETE CASCADE
                )
            """)
            conn.commit()