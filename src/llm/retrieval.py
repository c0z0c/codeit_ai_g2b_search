# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db import EmbeddingsDB
from src.config import get_config
from src.utils.logging_config import get_logger

class Retrieval:
    """
    Retrieval 클래스

    쿼리에 대한 유사 청크를 검색하는 기능을 제공합니다. 
    OpenAI 임베딩 모델과 FAISS를 활용하여 유사도 기반 검색을 수행합니다.

    Attributes:
        config: 설정 객체 (Config)
        embedding_model: 사용할 임베딩 모델 이름
        embeddings_db: 임베딩 데이터베이스 객체
        embeddings: LangChain OpenAI 임베딩 객체 (선택적)
    """

    def __init__(self, embedding_model: Optional[str] = None, config=None):
        """
        Retrieval 클래스 초기화

        Args:
            embedding_model: 사용할 임베딩 모델 이름 (기본값: Config에서 로드)
            config: 설정 객체 (기본값: get_config() 호출)
        """
        # Config 로드
        self.config = config or get_config()

        # 로거 초기화
        self.logger = get_logger(__name__)

        # 파라미터 우선, 없으면 Config 사용
        self.embedding_model = embedding_model or self.config.OPENAI_EMBEDDING_MODEL

        # 임베딩 데이터베이스 초기화
        self.embeddings_db = EmbeddingsDB()

        # LangChain 임베딩 초기화
        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self.logger.info(f"Retrieval 초기화 완료 (embedding_model={self.embedding_model})")
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")

    def search(
        self,
        query: str,
        embedding_hash: str,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 유사 청크 검색

        Args:
            query: 검색 쿼리 문자열
            embedding_hash: 임베딩 해시값 (FAISS 인덱스 식별자)
            top_k: 상위 k개 검색 (기본값: Config의 TOP_K_SUMMARY)
            api_key: OpenAI API 키 (선택적)

        Returns:
            List[Dict[str, Any]]: 검색된 청크와 유사도 정보 리스트
        """
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            self.logger.error("필수 패키지가 설치되지 않았습니다.")
            return []

        # top_k 기본값 설정
        if top_k is None:
            top_k = self.config.TOP_K_SUMMARY

        self.logger.info(f"검색 시작: query='{query[:50]}...', top_k={top_k}")

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 임베딩 메타데이터 가져오기
        meta = self.embeddings_db.get_embedding_meta(embedding_hash)
        if not meta or not meta.get('faiss_index_path'):
            self.logger.error(f"임베딩을 찾을 수 없습니다: {embedding_hash[:16]}...")
            return []

        # FAISS 인덱스 로드
        try:
            index = faiss.read_index(meta['faiss_index_path'])
            self.logger.debug(f"FAISS 인덱스 로드 완료: {meta['faiss_index_path']}")
        except Exception as e:
            self.logger.error(f"FAISS 인덱스 로드 실패: {e}")
            return []

        # 쿼리 임베딩 생성
        try:
            self.logger.debug(f"쿼리 임베딩 생성 중... (모델: {self.embedding_model})")
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
        except Exception as e:
            self.logger.error(f"쿼리 임베딩 생성 실패: {e}")
            return []

        # 유사도 검색
        distances, indices = index.search(query_vector, top_k)
        self.logger.debug(f"FAISS 검색 완료: {len(distances[0])}개 결과")

        # 결과 조회
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.embeddings_db.get_chunk_by_vector_index(embedding_hash, int(idx))
            if chunk:
                similarity = 1 / (1 + float(dist))
                chunk['similarity'] = similarity
                chunk['distance'] = float(dist)
                results.append(chunk)
                self.logger.debug(
                    f"  - 청크 {idx}: distance={dist:.4f}, similarity={similarity:.4f}, "
                    f"file={chunk.get('file_name', 'unknown')}"
                )

        self.logger.info(f"검색 완료: {len(results)}개 청크 반환")
        return results