# -*- coding: utf-8 -*-
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db import DocumentsDB, EmbeddingsDB
from src.config import get_config
from src.utils.logging_config import get_logger

class EmbeddingProcessor:
    """
    EmbeddingProcessor 클래스는 문서를 청킹(chunking)하고, 벡터 임베딩을 생성하며,
    생성된 임베딩 데이터를 저장 및 관리하는 역할을 수행합니다.

    주요 기능:
    - 문서 청킹: 긴 텍스트를 작은 청크로 나눔
    - 벡터 임베딩 생성: 청크 데이터를 벡터화
    - FAISS 인덱스 생성 및 저장: 임베딩 데이터를 효율적으로 검색 가능하도록 저장
    - 메타데이터 및 청크 매핑 저장: 임베딩과 원본 데이터 간의 매핑 관리
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        config=None
    ):
        """
        EmbeddingProcessor 초기화 메서드.

        Args:
            chunk_size (Optional[int]): 청크 크기 (기본값: config에서 로드)
            chunk_overlap (Optional[int]): 청크 간 중첩 크기 (기본값: config에서 로드)
            embedding_model (Optional[str]): 사용할 임베딩 모델 (기본값: config에서 로드)
            config: 설정 객체 (기본값: get_config() 호출)
        """
        # Config 로드
        self.config = config or get_config()

        # 로거 초기화
        self.logger = get_logger(__name__)

        # 파라미터 우선, 없으면 Config 사용
        self.chunk_size = chunk_size if chunk_size is not None else self.config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.config.CHUNK_OVERLAP
        self.embedding_model = embedding_model or self.config.OPENAI_EMBEDDING_MODEL

        # 데이터베이스 초기화
        self.docs_db = DocumentsDB()
        self.embeddings_db = EmbeddingsDB()

        # LangChain 관련 객체 초기화
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.config.CHUNK_SEPARATORS
            )
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self.logger.info(
                f"EmbeddingProcessor 초기화 완료 "
                f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap}, model={self.embedding_model})"
            )
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")

    def calculate_embedding_hash(self, file_hash: str, config: Dict) -> str:
        """
        임베딩 설정 해시를 계산합니다.

        Args:
            file_hash (str): 파일 해시값
            config (Dict): 임베딩 설정 정보

        Returns:
            str: 계산된 해시값
        """
        data = f"{file_hash}_{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> Optional[str]:
        """
        문서를 청킹하고 임베딩을 생성합니다.

        Args:
            file_hash (str): 파일 해시값
            api_key (Optional[str]): OpenAI API 키 (선택사항)

        Returns:
            Optional[str]: 생성된 임베딩 해시값 (실패 시 None 반환)
        """
        # 필수 패키지 확인
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            self.logger.error("필수 패키지가 설치되지 않았습니다.")
            return None

        self.logger.info(f"임베딩 처리 시작: {file_hash[:16]}...")

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 문서 내용 가져오기
        pages = self.docs_db.get_page_data(file_hash)
        if not pages:
            self.logger.error(f"문서를 찾을 수 없습니다: {file_hash[:16]}...")
            return None

        # 모든 페이지 콘텐츠 결합
        full_text = "\n\n".join([p['markdown_content'] for p in pages if not p['is_empty']])
        self.logger.debug(f"전체 텍스트 길이: {len(full_text):,} 문자")

        # 청킹
        chunks = self.text_splitter.split_text(full_text)
        total_chunks = len(chunks)
        self.logger.info(f"청킹 완료: {total_chunks}개 청크 생성")

        # 임베딩 생성
        try:
            self.logger.debug(f"임베딩 API 호출 중... (모델: {self.embedding_model})")
            embeddings = self.embeddings.embed_documents(chunks)
            embeddings_array = np.array(embeddings).astype('float32')
            self.logger.info(f"임베딩 벡터 생성 완료: shape={embeddings_array.shape}")
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            return None

        # FAISS 인덱스 생성
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # 임베딩 해시 계산
        config = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'model': self.embedding_model
        }
        embedding_hash = self.calculate_embedding_hash(file_hash, config)

        # FAISS 인덱스 저장
        faiss_path = self.config.get_vectorstore_path(embedding_hash)
        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, faiss_path)

        # 임베딩 메타데이터 저장
        self.embeddings_db.insert_embedding_meta(
            embedding_hash=embedding_hash,
            file_hash=file_hash,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            preprocessing_option={},
            embedding_model=self.embedding_model,
            total_chunks=total_chunks,
            faiss_index_path=faiss_path
        )

        # 청크 매핑 저장
        file_info = self.docs_db.get_file_info(file_hash)
        for idx, chunk_text in enumerate(chunks):
            self.embeddings_db.insert_chunk_mapping(
                embedding_hash=embedding_hash,
                file_hash=file_hash,
                file_name=file_info['file_name'] if file_info else 'unknown',
                chunk_text=chunk_text,
                vector_index=idx,
                estimated_tokens=len(chunk_text) // self.config.TOKEN_ESTIMATION_DIVISOR
            )

        self.logger.info(
            f"임베딩 처리 완료: {embedding_hash[:16]}... "
            f"(총 {total_chunks}개 청크, FAISS 인덱스: {faiss_path})"
        )
        return embedding_hash