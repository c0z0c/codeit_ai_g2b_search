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

class EmbeddingProcessor:
    """임베딩 처리 클래스 - 문서를 청킹하고 벡터 임베딩 생성"""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        config=None
    ):
        # Config 로드
        self.config = config or get_config()

        # 파라미터 우선, 없으면 Config 사용
        self.chunk_size = chunk_size if chunk_size is not None else self.config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else self.config.CHUNK_OVERLAP
        self.embedding_model = embedding_model or self.config.OPENAI_EMBEDDING_MODEL

        self.docs_db = DocumentsDB()
        self.embeddings_db = EmbeddingsDB()

        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.config.CHUNK_SEPARATORS
            )
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        else:
            print("LangChain이 설치되지 않았습니다.")

    def calculate_embedding_hash(self, file_hash: str, config: Dict) -> str:
        """임베딩 설정 해시 계산"""
        data = f"{file_hash}_{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> Optional[str]:
        """
        문서를 청킹하고 임베딩 생성

        Args:
            file_hash: 파일 해시값
            api_key: OpenAI API 키 (선택사항)

        Returns:
            임베딩 해시값 또는 None
        """
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            print("필수 패키지가 설치되지 않았습니다.")
            return None

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 문서 내용 가져오기
        pages = self.docs_db.get_page_data(file_hash)
        if not pages:
            print(f"문서를 찾을 수 없습니다: {file_hash[:8]}...")
            return None

        # 모든 페이지 콘텐츠 결합
        full_text = "\n\n".join([p['markdown_content'] for p in pages if not p['is_empty']])

        # 청킹
        chunks = self.text_splitter.split_text(full_text)
        total_chunks = len(chunks)
        print(f"청킹 완료: {total_chunks}개 청크 생성")

        # 임베딩 생성
        try:
            embeddings = self.embeddings.embed_documents(chunks)
            embeddings_array = np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
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

        print(f"임베딩 처리 완료: {embedding_hash[:8]}...")
        return embedding_hash
