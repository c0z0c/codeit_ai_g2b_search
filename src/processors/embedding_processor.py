# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db.documents_db import DocumentsDB
from src.config import get_config
from src.utils.logging_config import get_logger
from src.vectorstore import VectorStoreManager

import importlib
from src.db import documents_db
importlib.reload(documents_db)
from src.db.documents_db import DocumentsDB


class EmbeddingProcessor:
    """
    EmbeddingProcessor 클래스는 문서를 청킹(chunking)하고, 벡터 임베딩을 생성하며,
    VectorStoreManager를 통해 FAISS 인덱스를 관리합니다.

    주요 기능:
    - 문서 청킹: 긴 텍스트를 작은 청크로 나눔
    - 벡터 임베딩 생성: VectorStoreManager를 통한 임베딩 생성 및 저장
    - 메타데이터 관리: Document.metadata에 모든 청크 정보 저장
    """

    def __init__(
        self,
        config=None
    ):
        """
        EmbeddingProcessor 초기화 메서드.

        Args:
            chunk_size (Optional[int]): 청크 크기 (기본값: config에서 로드)
            chunk_overlap (Optional[int]): 청크 간 중첩 크기 (기본값: config에서 로드)
            embedding_model (Optional[str]): 사용할 임베딩 모델 (기본값: config에서 로드)
            vector_path (Optional[str]): 통합 FAISS 인덱스 경로 (기본값: config에서 로드)
            config: 설정 객체 (기본값: get_config() 호출)
        """
        # Config 로드
        self.config = config or get_config()

        # 로거 초기화
        self.logger = get_logger('[EMBP]')

        # 파라미터 우선, 없으면 Config 사용
        self.chunking_mode = self.config.CHUNKING_MODE
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
        self.embedding_model = self.config.OPENAI_EMBEDDING_MODEL
        self.vector_path = self.config.VECTORSTORE_PATH

        # 데이터베이스 초기화
        self.docs_db = DocumentsDB(self.config.DOCUMENTS_DB_PATH)
        
        # VectorStoreManager 초기화
        try:
            self.vector_manager = VectorStoreManager(config=self.config)
        except ImportError as e:
            self.logger.error(f"VectorStoreManager 초기화 실패: {e}")
            self.vector_manager = None

        # LangChain 텍스트 스플리터 초기화
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.config.CHUNK_SEPARATORS
            )
            self.logger.info(
                f"EmbeddingProcessor 초기화 완료 "
                f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, "
                f"model={self.embedding_model}, faiss_path={self.vector_path})"
            )
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> bool:
        """
        문서를 청킹하고 통합 FAISS 인덱스에 임베딩을 추가합니다.

        Args:
            file_hash (str): 파일 해시값
            api_key (Optional[str]): OpenAI API 키 (선택사항)

        Returns:
            bool: 처리 성공 여부
        """
        # 필수 패키지 및 VectorStoreManager 확인
        if not LANGCHAIN_AVAILABLE or self.vector_manager is None:
            self.logger.error("필수 패키지가 설치되지 않았습니다.")
            return False

        self.logger.info(f"임베딩 처리 시작: {file_hash[:16]}...")

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 문서 내용 가져오기
        file_info = self.docs_db.get_file_info(file_hash)
        if not file_info:
            self.logger.error(f"파일 정보를 찾을 수 없습니다: {file_hash[:16]}...")
            return False

        pages = self.docs_db.get_page_data(file_hash)
        if not pages:
            self.logger.error(f"페이지 데이터를 찾을 수 없습니다: {file_hash[:16]}...")
            return False

        # 빈 페이지 제외하고 페이지별 텍스트 구성
        page_texts = []
        page_numbers = []
        for p in pages:
            if not p['is_empty']:
                page_texts.append(p['markdown_content'])
                page_numbers.append(p['page_number'])
        
        # 페이지별 텍스트를 마커와 함께 결합
        full_text_with_markers = ""
        page_start_positions = []  # 각 페이지 시작 위치 기록
        
        for page_num, page_text in zip(page_numbers, page_texts):
            page_start_positions.append((len(full_text_with_markers), page_num))
            full_text_with_markers += page_text + "\n\n"
        
        self.logger.debug(
            f"전체 텍스트 길이: {len(full_text_with_markers):,} 문자, "
            f"페이지 수: {len(page_numbers)}"
        )

        # 청킹
        chunks = self.text_splitter.split_text(full_text_with_markers)
        total_chunks = len(chunks)
        self.logger.info(f"청킹 완료: {total_chunks}개 청크 생성")
        
        # 각 청크의 페이지 범위 계산
        def get_page_range(chunk_text: str, chunk_start_pos: int) -> tuple:
            """청크의 시작/종료 페이지 번호를 계산합니다."""
            chunk_end_pos = chunk_start_pos + len(chunk_text)
            
            start_page = None
            end_page = None
            
            # 시작 페이지 찾기
            for pos, page_num in reversed(page_start_positions):
                if pos <= chunk_start_pos:
                    start_page = page_num
                    break
            
            # 종료 페이지 찾기
            for pos, page_num in reversed(page_start_positions):
                if pos < chunk_end_pos:
                    end_page = page_num
                    break
            
            return start_page, end_page
        
        # 청크별 페이지 정보 계산
        chunk_page_info = []
        current_pos = 0
        for chunk in chunks:
            start_page, end_page = get_page_range(chunk, current_pos)
            chunk_page_info.append({'start_page': start_page, 'end_page': end_page})
            # 다음 청크 시작 위치 계산 (overlap 고려)
            current_pos += len(chunk) - self.chunk_overlap
            if current_pos < 0:
                current_pos = 0
        
        self.logger.debug(f"페이지 범위 계산 완료: {len(chunk_page_info)}개 청크")

        # VectorStoreManager를 통해 벡터 추가
        try:
            # 메타데이터 생성 (페이지 정보 + 버전 정보 포함)
            from datetime import datetime
            metadatas = [
                {
                    'file_hash': file_hash,
                    'file_name': file_info['file_name'],
                    'start_page': chunk_page_info[i]['start_page'],
                    'end_page': chunk_page_info[i]['end_page'],
                    'chunk_type': 'paragraph',
                    'chunk_index': i,
                    'embedding_version': self.embedding_model,
                    'created_at': datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            # 벡터 추가 및 시작 인덱스 확인
            success, start_index = self.vector_manager.add_texts(chunks, metadatas)
            
            if not success:
                self.logger.error("벡터 추가 실패")
                return False
            
            # FAISS 인덱스 저장
            if not self.vector_manager.save():
                self.logger.error("FAISS 인덱스 저장 실패")
                return False
            
            self.logger.info(f"벡터 추가 및 저장 완료 (시작 인덱스: {start_index})")
        
        except Exception as e:
            self.logger.error(f"임베딩 처리 실패: {e}")
            return False

        self.logger.info(
            f"임베딩 처리 완료: {file_hash[:16]}... "
            f"(추가 {total_chunks}개 청크, 총 {self.vector_manager.get_vector_count()}개 청크)"
        )
        return True