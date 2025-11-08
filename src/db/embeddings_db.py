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

    def insert_embedding_meta(
        self,
        embedding_hash: str,
        file_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        preprocessing_option: Dict,
        embedding_model: str,
        total_chunks: int,
        faiss_index_path: str
    ) -> bool:
        """
        임베딩 메타데이터를 데이터베이스에 저장합니다.

        Args:
            embedding_hash (str): 임베딩 해시값
            file_hash (str): 파일 해시값
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 중첩 크기
            preprocessing_option (Dict): 전처리 옵션
            embedding_model (str): 임베딩 모델명
            total_chunks (int): 총 청크 개수
            faiss_index_path (str): FAISS 인덱스 파일 경로

        Returns:
            bool: 저장 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embedding_meta
                (embedding_hash, file_hash, chunk_size, chunk_overlap, preprocessing_option,
                 embedding_model, total_chunks, faiss_index_path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_hash,
                file_hash,
                chunk_size,
                chunk_overlap,
                json.dumps(preprocessing_option),
                embedding_model,
                total_chunks,
                faiss_index_path,
                datetime.now()
            ))
            conn.commit()
            return True

    def insert_chunk_mapping(
        self,
        embedding_hash: str,
        file_hash: str,
        file_name: str,
        chunk_text: str,
        vector_index: int,
        estimated_tokens: int,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> bool:
        """
        청크 매핑 데이터를 데이터베이스에 저장합니다.

        Args:
            embedding_hash (str): 임베딩 해시값
            file_hash (str): 파일 해시값
            file_name (str): 파일 이름
            chunk_text (str): 청크 텍스트
            vector_index (int): 벡터 인덱스
            estimated_tokens (int): 추정 토큰 수
            start_page (Optional[int]): 시작 페이지
            end_page (Optional[int]): 종료 페이지

        Returns:
            bool: 저장 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chunk_mapping
                (embedding_hash, file_hash, file_name, start_page, end_page,
                 chunk_text, estimated_tokens, vector_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                embedding_hash,
                file_hash,
                file_name,
                start_page,
                end_page,
                chunk_text,
                estimated_tokens,
                vector_index
            ))
            conn.commit()
            return True

    def get_embedding_meta(self, embedding_hash: str) -> Optional[Dict[str, Any]]:
        """
        임베딩 메타데이터를 조회합니다.

        Args:
            embedding_hash (str): 임베딩 해시값

        Returns:
            Optional[Dict[str, Any]]: 메타데이터 딕셔너리 (없으면 None)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM embedding_meta WHERE embedding_hash = ?",
                (embedding_hash,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None

    def get_chunks(self, embedding_hash: str) -> List[Dict[str, Any]]:
        """
        특정 임베딩에 속한 모든 청크를 조회합니다.

        Args:
            embedding_hash (str): 임베딩 해시값

        Returns:
            List[Dict[str, Any]]: 청크 리스트
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunk_mapping WHERE embedding_hash = ? ORDER BY vector_index",
                (embedding_hash,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_embedding(self, embedding_hash: str) -> bool:
        """
        임베딩 메타데이터와 관련 청크를 삭제합니다.

        Args:
            embedding_hash (str): 임베딩 해시값

        Returns:
            bool: 삭제 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM embedding_meta WHERE embedding_hash = ?",
                (embedding_hash,)
            )
            conn.commit()
            return True