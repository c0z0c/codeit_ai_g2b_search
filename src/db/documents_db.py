# -*- coding: utf-8 -*-
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from src.utils.logging_config import get_logger
logger = get_logger('[DOCDB]')

class DocumentsDB:
    """
    마크다운 문서 데이터베이스 관리 클래스

    주요 기능:
    - 마크다운 파일 정보를 SQLite 데이터베이스에 저장
    - 데이터베이스 테이블 생성 및 관리
    - 문서 통계 정보 제공

    사용 예:
        db = DocumentsDB("data/documents.db")
        db.insert_text_content(file_hash, file_name, total_pages, file_size, text_content)
        stats = db.get_document_stats()
    """

    def __init__(self, db_path: str = 'data/documents.db'):
        """
        DocumentsDB 초기화

        Args:
            db_path (str): 데이터베이스 파일 경로 (기본값: 'data/documents.db')
        """
        self.USER_VERSION = 1
        self.logger = get_logger('[DOCDB]')
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
        데이터베이스 테이블 및 인덱스 생성

        - TB_DOCUMENTS: 마크다운 파일 정보를 저장하는 테이블
            - file_hash: 파일의 고유 해시값 (기본 키, 중복 불가)
            - file_name: 파일 이름 (경로 포함, 고유 해야함)
            - total_pages: 원본 파일의 총 페이지 수
            - file_size: 원본 파일 크기 (바이트 단위)
            - text_content: 변환된 파일의 텍스트 콘텐츠 (텍스트, 마크다운, HTML 등)
            - created_at: 레코드 생성 시각 (기본값: 현재 시각)
            - updated_at: 레코드 수정 시각 (기본값: 현재 시각)
        
        - 인덱스:
            - idx_file_name: file_name 컬럼에 대한 인덱스
            - idx_created_at: created_at 컬럼에 대한 시간 기반 검색 최적화를 위한 인덱스
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # TB_DOCUMENTS 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS TB_DOCUMENTS (
                    file_hash TEXT PRIMARY KEY,  -- 파일 고유 해시값 (중복 불가)
                    file_name TEXT NOT NULL,    -- 파일 이름 (변환 전 이름, 고유 해야함)
                    total_pages INTEGER NOT NULL,  -- 원본 파일 총 페이지 수
                    file_size INTEGER NOT NULL,    -- 원본 파일 크기 (바이트 단위)
                    text_content TEXT,             -- 변환된 파일의 텍스트 콘텐츠
                    created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours')),  -- 생성 시각
                    updated_at TIMESTAMP DEFAULT (datetime('now', '+9 hours'))   -- 수정 시각
                )
            """)
            # 트리거를 사용해 updated_at 컬럼 자동 업데이트
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_updated_at
                AFTER UPDATE ON TB_DOCUMENTS
                FOR EACH ROW
                BEGIN
                    UPDATE TB_DOCUMENTS
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE file_hash = OLD.file_hash;
                END;
            """)            
            # 추후 프로그램 고도화시 마이 그레이션 에서 검토 한다.
            # # 인덱스 생성 (file_name)
            # cursor.execute("""
            #     CREATE INDEX IF NOT EXISTS idx_file_name
            #     ON TB_DOCUMENTS (file_name)
            # """)
            # # created_at 인덱스 생성
            # cursor.execute("""
            #     CREATE INDEX IF NOT EXISTS idx_created_at
            #     ON TB_DOCUMENTS (created_at)
            # """)
            
            # PRAGMA user_version을 사용한 DB 버전 관리
            cursor.execute("PRAGMA user_version")
            db_version = cursor.fetchone()[0]
            
            if db_version == 0:
                # 초기화: user_version이 0이면 현재 버전으로 설정
                cursor.execute(f"PRAGMA user_version = {self.USER_VERSION}")
                self.logger.info(f"데이터베이스 버전 초기화: {self.USER_VERSION}")
            elif db_version != self.USER_VERSION:
                self.logger.warning(f"데이터베이스 버전 불일치: {db_version} → {self.USER_VERSION}")
                # 마이그레이션 로직 추가 가능
                self._migrate_database(db_version, self.USER_VERSION)

            conn.commit()
            
    def _migrate_database(self, old_version: int, new_version: int):
        """
        데이터베이스 마이그레이션 로직
        추후 버전 업그레이드 시 필요한 마이그레이션 로직을 여기에 구현
        
        Args:
            old_version (int): 기존 데이터베이스 버전
            new_version (int): 새로운 데이터베이스 버전
        """
        #with self._get_connection() as conn:
            #cursor = conn.cursor()
            #self.logger.info(f"데이터베이스 마이그레이션 시작: {old_version} → {new_version}")
            
            # if old_version == 1 and 2 == new_version:
                
            #     # 버전 2로 마이그레이션
            #     cursor.execute("ALTER TABLE TB_DOCUMENTS ADD COLUMN new_column TEXT")
            #     self.logger.info("버전 2로 마이그레이션 완료")
            # 추가 마이그레이션 로직 작성 가능
            
            # 최종적으로 user_version 업데이트
            # cursor.execute(f"PRAGMA user_version = {new_version}")
            # conn.commit()
            #self.logger.info(f"데이터베이스 마이그레이션 완료: {new_version}")            
        raise NotImplementedError("마이그레이션 로직이 구현되지 않았습니다.")

    def insert_text_content(self, 
                            file_name: str = "",
                            file_hash: str = "", 
                            total_pages: Optional[int] = 0, 
                            file_size: Optional[int] = 0, 
                            text_content: Optional[str] = "") -> bool:
        """
        마크다운 파일 정보를 데이터베이스에 저장

        Args:
            file_hash (str): 파일의 해시값
            file_name (str): 파일 이름 (필수, 고유)
            total_pages (Optional[int]): 총 페이지 수 (기본값: None)
            file_size (Optional[int]): 파일 크기 (기본값: None)
            text_content (Optional[str]): 파일의 전체 텍스트 콘텐츠 (기본값: "")

        Returns:
            bool: 저장 성공 여부
        """
        if not file_name:
            raise ValueError("file_name은 필수 항목이며 None 또는 빈 값일 수 없습니다.")
        if text_content is None:
            text_content = ""  # 최소한 빈 문자열이어야 함

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO TB_DOCUMENTS
                (file_hash, file_name, total_pages, file_size, text_content, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_hash, file_name, total_pages, file_size, text_content, datetime.now()))
            conn.commit()
            return True
        
    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        파일 해시값으로 문서 정보 조회

        Args:
            file_hash (str): 조회할 파일의 해시값

        Returns:
            Optional[Dict[str, Any]]: 문서 정보 (없으면 None 반환)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM TB_DOCUMENTS WHERE file_hash = ?
            """, (file_hash,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_document_stats(self) -> Dict[str, Any]:
        """
        데이터베이스에 저장된 문서 통계 정보를 반환

        Returns:
            Dict[str, Any]: 문서 통계 정보
                - total_files: 총 파일 수
                - total_pages: 총 페이지 수
                - total_size_bytes: 총 파일 크기 (바이트 단위)
                - total_size_mb: 총 파일 크기 (MB 단위)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM TB_DOCUMENTS')
            total_files = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(total_pages) FROM TB_DOCUMENTS')
            total_pages = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(file_size) FROM TB_DOCUMENTS')
            total_size = cursor.fetchone()[0] or 0
            return {
                'total_files': total_files,
                'total_pages': total_pages,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
            }
            
    def search_documents(self, 
                        search_term: str, 
                        search_type: str = 'auto') -> List[Dict[str, Any]]:
        """
        파일명 또는 file_hash로 문서 검색
        
        Args:
            search_term (str): 검색어 (파일명 또는 file_hash)
            search_type (str): 검색 타입
                - 'auto': file_hash 형식(64자 hex)이면 hash 검색, 아니면 filename 검색
                - 'filename': file_name LIKE 검색
                - 'hash': file_hash 완전 일치 검색
        
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
        
        Example:
            >>> # file_hash로 검색
            >>> results = db.search_documents("abc123def456...")
            >>> 
            >>> # 파일명으로 검색
            >>> results = db.search_documents("공고문", search_type='filename')
            >>> 
            >>> # 자동 판별
            >>> results = db.search_documents("test.pdf", search_type='auto')
        """
        if not search_term:
            self.logger.warning("검색어가 비어있습니다.")
            return []
        
        # search_type 결정
        if search_type == 'auto':
            # 64자 hex 문자열이면 hash 검색
            if len(search_term) == 64 and all(c in '0123456789abcdefABCDEF' for c in search_term):
                search_type = 'hash'
            else:
                search_type = 'filename'
        
        # 검색 쿼리 실행
        if search_type == 'hash':
            query = "SELECT * FROM TB_DOCUMENTS WHERE file_hash = ?"
            params = (search_term,)
        elif search_type == 'filename':
            query = "SELECT * FROM TB_DOCUMENTS WHERE file_name LIKE ?"
            params = (f"%{search_term}%",)
        else:
            self.logger.error(f"지원하지 않는 search_type: {search_type}")
            return []
        
        results = self.execute_query(query, params)
        self.logger.debug(f"검색 완료: {len(results)}건 ({search_type} 모드, 검색어: {search_term})")
        return results
            
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        SELECT 쿼리를 실행하고 결과를 반환
        
        Args:
            query (str): 실행할 SQL 쿼리
            params (Optional[tuple]): 쿼리 파라미터 (기본값: None)
        
        Returns:
            List[Dict[str, Any]]: 쿼리 결과 (각 행이 딕셔너리)
        
        Example:
            >>> results = db.execute_query("SELECT * FROM TB_DOCUMENTS WHERE file_hash = ?", ("abc123",))
            >>> for row in results:
            >>>     print(row['file_name'])
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        
        except Exception as e:
            self.logger.error(f"쿼리 실행 중 오류: {e}")
            self.logger.debug(f"쿼리: {query}")
            if params:
                self.logger.debug(f"파라미터: {params}")
            return []
            
    def summary(self) -> None:
        """
        데이터베이스의 모든 테이블 요약 정보 출력.
        
        각 테이블의 컬럼명과 레코드 수를 로거로 출력합니다.
        """
        try:
            # 테이블 목록 조회
            tables = self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            
            if not tables:
                self.logger.info("데이터베이스에 테이블이 없습니다.")
                return
            
            self.logger.info("=" * 80)
            self.logger.info(f"데이터베이스: {self.db_path}")
            self.logger.info(f"총 테이블 수: {len(tables)}")
            self.logger.info("=" * 80)
            
            for table_info in tables:
                table_name = table_info['name']
                
                # 컬럼 정보
                columns = self.execute_query(f"PRAGMA table_info({table_name})")
                column_names = [col['name'] for col in columns]
                
                # 레코드 수
                count_result = self.execute_query(f"SELECT COUNT(*) as cnt FROM {table_name}")
                row_count = count_result[0]['cnt'] if count_result else 0
                
                self.logger.info(f"테이블: {table_name}")
                self.logger.info(f"  - 컬럼 수: {len(column_names)}")
                self.logger.info(f"  - 컬럼명: {', '.join(column_names)}")
                self.logger.info(f"  - 레코드 수: {row_count}")
                self.logger.info("-" * 80)
                
        except Exception as e:
            self.logger.error(f"summary() 실행 중 오류: {e}")
            
logger.info("DocumentsDB 모듈이 로드되었습니다.")
if __name__ == "__main__":
    db = DocumentsDB("data/documents.db")
    db.summary()