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

class Retrieval:
    \"\"\"검색 클래스 - 쿼리에 대한 유사 청크 검색\"\"\"

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings_db = EmbeddingsDB()
        self.embedding_model = embedding_model

        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            print("LangChain이 설치되지 않았습니다.")

    def search(
        self,
        query: str,
        embedding_hash: str,
        top_k: int = 5,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        \"\"\"
        쿼리에 대한 유사 청크 검색

        Args:
            query: 검색 쿼리
            embedding_hash: 임베딩 해시값
            top_k: 상위 k개 검색
            api_key: OpenAI API 키

        Returns:
            검색 결과 리스트
        \"\"\"
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            print("필수 패키지가 설치되지 않았습니다.")
            return []

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 임베딩 메타데이터 가져오기
        meta = self.embeddings_db.get_embedding_meta(embedding_hash)
        if not meta or not meta.get('faiss_index_path'):
            print(f"임베딩을 찾을 수 없습니다: {embedding_hash[:8]}...")
            return []

        # FAISS 인덱스 로드
        try:
            index = faiss.read_index(meta['faiss_index_path'])
        except Exception as e:
            print(f"FAISS 인덱스 로드 실패: {e}")
            return []

        # 쿼리 임베딩 생성
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
        except Exception as e:
            print(f"쿼리 임베딩 생성 실패: {e}")
            return []

        # 유사도 검색
        distances, indices = index.search(query_vector, top_k)

        # 결과 조회
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.embeddings_db.get_chunk_by_vector_index(embedding_hash, int(idx))
            if chunk:
                chunk['similarity'] = 1 / (1 + float(dist))
                chunk['distance'] = float(dist)
                results.append(chunk)

        return results
