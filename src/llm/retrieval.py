# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.vectorstore.vector_store_manager import VectorStoreManager
from src.config import get_config
from src.utils.logging_config import get_logger

class Retrieval:
    """
    Retrieval 클래스

    쿼리에 대한 유사 청크를 검색하는 기능을 제공합니다. 
    VectorStoreManager를 통해 통합 FAISS 인덱스에서 유사도 기반 검색을 수행합니다.

    Attributes:
        config: 설정 객체 (Config)
        embedding_model: 사용할 임베딩 모델 이름
        vector_manager: VectorStoreManager 인스턴스
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
        self.logger = get_logger('[RT]')

        # 파라미터 우선, 없으면 Config 사용
        self.embedding_model = embedding_model or self.config.OPENAI_EMBEDDING_MODEL

        # VectorStoreManager 초기화
        self.vector_manager = VectorStoreManager(config=self.config)
        
        # FAISS 인덱스 로드
        load_success = self.vector_manager.load()
        if not load_success:
            self.logger.warning("FAISS 인덱스 로드 실패")

        # LangChain 임베딩 초기화
        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            self.logger.info(
                f"Retrieval 초기화 완료 "
                f"(model={self.embedding_model}, vectors={self.vector_manager.get_vector_count()})"
            )
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        쿼리에 대한 유사 청크 검색

        Args:
            query: 검색 쿼리 문자열
            top_k: 상위 k개 검색 (기본값: Config의 TOP_K_SUMMARY)
            filter_metadata: 메타데이터 필터 (선택적, 예: {'file_hash': 'abc123'})
            api_key: OpenAI API 키 (선택적)

        Returns:
            List[Dict[str, Any]]: 검색된 청크와 유사도 정보 리스트
            각 항목 구조:
            {
                'text': str,              # 청크 텍스트
                'file_hash': str,          # 파일 해시
                'file_name': str,          # 파일명
                'page_number': int,        # 페이지 번호
                'chunk_index': int,        # 청크 인덱스
                'chunk_hash': str,         # 청크 해시
                'distance': float,         # L2 거리 (작을수록 유사)
                'created_at': str          # 생성 시각
            }
        """
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("LangChain이 설치되지 않았습니다.")
            return []

        # top_k 기본값 설정
        if top_k is None:
            top_k = self.config.TOP_K_SUMMARY

        self.logger.info(f"검색 시작: query='{query[:50]}...', top_k={top_k}")

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # VectorStoreManager를 통한 검색
        try:
            search_results = self.vector_manager.search(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            self.logger.debug(f"FAISS 검색 완료: {len(search_results)}개 결과")
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            return []

        # 결과 변환: List[Tuple[Document, float]] -> List[Dict[str, Any]]
        results = []
        for doc, distance in search_results:
            result = {
                'text': doc.page_content,
                'file_hash': doc.metadata.get('file_hash', ''),
                'file_name': doc.metadata.get('file_name', 'unknown'),
                'page_number': doc.metadata.get('page_number', 0),
                'chunk_index': doc.metadata.get('chunk_index', 0),
                'chunk_hash': doc.metadata.get('chunk_hash', ''),
                'distance': float(distance),
                'created_at': doc.metadata.get('created_at', '')
            }
            results.append(result)
            
            self.logger.debug(
                f"  - 청크 {result['chunk_index']}: distance={distance:.4f}, "
                f"file={result['file_name']}, page={result['page_number']}"
            )

        self.logger.info(f"검색 완료: {len(results)}개 청크 반환")
        return results