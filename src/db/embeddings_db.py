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
        - embedding_config: 통합 FAISS 설정을 저장하는 테이블.
        - chunk_mapping: 텍스트 청크 데이터를 저장하는 테이블.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 통합 임베딩 설정 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_size INTEGER NOT NULL,         -- 청킹 크기
                    chunk_overlap INTEGER NOT NULL,      -- 청크 오버랩
                    embedding_model TEXT NOT NULL,       -- 임베딩 모델명
                    vector_path TEXT NOT NULL,      -- 통합 FAISS 인덱스 경로
                    total_chunks INTEGER DEFAULT 0,      -- 총 청크 개수
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 텍스트 청크 매핑 테이블 생성 (통합 FAISS용)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_mapping (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 청크 ID (자동 증가)
                    file_hash TEXT NOT NULL,                     -- 원본 파일 해시
                    file_name TEXT NOT NULL,                     -- 파일명 (검색 결과에 포함)
                    start_page INTEGER,                          -- 시작 페이지 번호
                    end_page INTEGER,                            -- 종료 페이지 번호
                    chunk_text TEXT NOT NULL,                    -- 청크 텍스트
                    estimated_tokens INTEGER,                    -- 추정 토큰 수
                    vector_index INTEGER NOT NULL,               -- 통합 FAISS 벡터 인덱스
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(vector_index)                         -- 벡터 인덱스는 고유
                )
            """)
            conn.commit()

    def get_or_create_config(
        self,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        vector_path: str
    ) -> int:
        """
        통합 임베딩 설정을 조회하거나 생성합니다.

        Args:
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 중첩 크기
            embedding_model (str): 임베딩 모델명
            vector_path (str): FAISS 인덱스 파일 경로

        Returns:
            int: 설정 ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 기존 설정 조회
            cursor.execute("""
                SELECT id FROM embedding_config
                WHERE chunk_size = ? AND chunk_overlap = ? AND embedding_model = ?
                LIMIT 1
            """, (chunk_size, chunk_overlap, embedding_model))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            
            # 새 설정 생성
            cursor.execute("""
                INSERT INTO embedding_config
                (chunk_size, chunk_overlap, embedding_model, vector_path)
                VALUES (?, ?, ?, ?)
            """, (chunk_size, chunk_overlap, embedding_model, vector_path))
            conn.commit()
            return cursor.lastrowid or 0
    
    def update_total_chunks(self, config_id: int, total_chunks: int) -> bool:
        """
        통합 설정의 총 청크 수를 업데이트합니다.

        Args:
            config_id (int): 설정 ID
            total_chunks (int): 총 청크 개수

        Returns:
            bool: 업데이트 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE embedding_config
                SET total_chunks = ?, updated_at = ?
                WHERE id = ?
            """, (total_chunks, datetime.now(), config_id))
            conn.commit()
            return True

    def insert_chunk_mapping(
        self,
        file_hash: str,
        file_name: str,
        chunk_text: str,
        vector_index: int,
        estimated_tokens: int,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> bool:
        """
        청크 매핑 데이터를 통합 FAISS 인덱스에 저장합니다.

        Args:
            file_hash (str): 파일 해시값
            file_name (str): 파일명
            chunk_text (str): 청크 텍스트
            vector_index (int): 통합 FAISS 벡터 인덱스
            estimated_tokens (int): 추정 토큰 수
            start_page (Optional[int]): 시작 페이지 번호
            end_page (Optional[int]): 종료 페이지 번호

        Returns:
            bool: 저장 성공 여부
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chunk_mapping
                (file_hash, file_name, start_page, end_page,
                 chunk_text, estimated_tokens, vector_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
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

    def get_config(self, config_id: int = 1) -> Optional[Dict[str, Any]]:
        """
        통합 임베딩 설정을 조회합니다.

        Args:
            config_id (int): 설정 ID (기본값: 1)

        Returns:
            Optional[Dict[str, Any]]: 설정 딕셔너리 (없으면 None)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM embedding_config WHERE id = ?",
                (config_id,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None

    def get_chunk_by_vector_index(self, vector_index: int) -> Optional[Dict[str, Any]]:
        """
        벡터 인덱스로 청크 정보를 조회합니다.

        Args:
            vector_index (int): 통합 FAISS 벡터 인덱스

        Returns:
            Optional[Dict[str, Any]]: 청크 정보 (file_name, start_page, end_page 포함)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT chunk_id, file_hash, file_name, start_page, end_page, 
                          chunk_text, estimated_tokens, vector_index 
                   FROM chunk_mapping 
                   WHERE vector_index = ?""",
                (vector_index,)
            )
            result = cursor.fetchone()
            return dict(result) if result else None

    def get_chunks_by_file(self, file_hash: str) -> List[Dict[str, Any]]:
        """
        특정 파일에 속한 모든 청크를 조회합니다.

        Args:
            file_hash (str): 파일 해시값

        Returns:
            List[Dict[str, Any]]: 청크 리스트
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chunk_mapping WHERE file_hash = ? ORDER BY vector_index",
                (file_hash,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_total_chunks(self) -> int:
        """
        전체 청크 수를 조회합니다.

        Returns:
            int: 전체 청크 개수
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunk_mapping")
            result = cursor.fetchone()
            return result[0] if result else 0

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        임베딩 통계 정보를 조회합니다.

        Returns:
            Dict[str, Any]: 통계 정보
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 총 청크 수
            cursor.execute("SELECT COUNT(*) FROM chunk_mapping")
            total_chunks = cursor.fetchone()[0] or 0
            
            # 파일별 청크 수
            cursor.execute("SELECT COUNT(DISTINCT file_hash) FROM chunk_mapping")
            total_files = cursor.fetchone()[0] or 0
            
            # 총 토큰 수
            cursor.execute("SELECT SUM(estimated_tokens) FROM chunk_mapping")
            total_tokens = cursor.fetchone()[0] or 0
            
            return {
                'total_chunks': total_chunks,
                'total_files': total_files,
                'total_tokens': total_tokens
            }