# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9 이상에서 사용 가능

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain.docstore.document import Document
    
    # LangChain FAISS 객체 재구성
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.utils import DistanceStrategy

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.config import get_config, Config
from src.utils.logging_config import get_logger


class VectorStoreManager:
    """
    VectorStoreManager 클래스
    
    FAISS 벡터 인덱스의 생성, 로드, 저장, 검색을 관리합니다.
    LangChain FAISS 통합을 통해 일관된 인터페이스를 제공합니다.
    
    주요 기능:
    - FAISS 인덱스 생성/로드/저장
    - 벡터 추가 (단일/배치)
    - 유사도 검색
    - 메타데이터 관리 (Document.metadata)
    
    Attributes:
        vector_path: FAISS 인덱스 파일 경로
        embedding_model: 임베딩 모델명
        embeddings: LangChain OpenAIEmbeddings 객체
        vectorstore: LangChain FAISS 벡터스토어
    """
    
    def __init__(
        self,
        config=None
    ):
        """
        VectorStoreManager 초기화
        
        Args:
            vector_path: FAISS 인덱스 파일 경로
            embedding_model: 임베딩 모델명
            embeddings_db: 임베딩 데이터베이스 객체 (선택)
        """
        self.logger = get_logger('[VC]')
        
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            self.logger.error("필수 패키지(LangChain, FAISS)가 설치되지 않았습니다.")
            raise ImportError("LangChain 및 FAISS 설치 필요")
        
        self.config = config or get_config()
        
        self.vector_path = Path(self.config.VECTORSTORE_PATH)
        self.embedding_model = self.config.OPENAI_EMBEDDING_MODEL
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vectorstore: Optional[FAISS] = None
        
        
        self.logger.info(
            f"VectorStoreManager 초기화 완료 "
            f"(model={self.embedding_model}, path={self.vector_path})"
        )
        
    def _create_dummy_index(self) -> bool:
        """
        더미 문서로 빈 FAISS 인덱스를 생성합니다.
        
        Returns:
            bool: 생성 성공 여부
        """
        try:
            dummy_doc = Document(
                page_content="[초기화용 더미 문서]",
                metadata={
                    'file_name': '__init__',
                    'file_hash': '',
                    'start_page': 0,
                    'end_page': 0,
                    'chunk_type': 'dummy',
                    'chunk_index': 0,
                    "embedding_version": self.config.OPENAI_EMBEDDING_MODEL,
                    "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
                    
                }
            )
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            
            # 디렉토리 생성 및 저장
            self.vector_path.parent.mkdir(parents=True, exist_ok=True)
            index_dir = str(self.vector_path.parent)
            index_name = self.vector_path.stem
            self.vectorstore.save_local(index_dir, index_name)
            
            self.logger.info(f"더미 FAISS 인덱스 생성 완료: {self.vector_path}")
            return True
        except Exception as e:
            self.logger.error(f"더미 인덱스 생성 실패: {e}")
            return False
    
    def load(self) -> bool:
        """
        기존 FAISS 인덱스를 로드합니다.
        파일이 없으면 더미 인덱스를 자동 생성합니다.
        
        Returns:
            bool: 로드 성공 여부
        """
        index_dir = str(self.vector_path.parent)
        index_name = self.vector_path.stem
        faiss_file = self.vector_path.parent / f"{index_name}.faiss"
        
        # .faiss 파일 존재 여부 체크
        if not faiss_file.exists():
            self.logger.warning(
                f"FAISS 인덱스 미존재: {faiss_file} — 더미 인덱스 생성"
            )
            return self._create_dummy_index()
        
        # 기존 인덱스 로드
        self.vectorstore = FAISS.load_local(
            folder_path=index_dir,
            embeddings=self.embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=True
        )
        
        vector_count = self.vectorstore.index.ntotal
        self.logger.info(
            f"FAISS 인덱스 로드 완료: {self.vector_path} "
            f"(벡터 수: {vector_count})"
        )
        return True
        
    
    def create_from_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        문서 리스트로부터 새로운 FAISS 인덱스를 생성합니다.
        
        Args:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트 (선택)
        
        Returns:
            bool: 생성 성공 여부
        """
        try:
            # LangChain Document 객체 생성
            documents = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas or [{}] * len(texts))
            ]
            
            # FAISS.from_documents() 사용
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            self.logger.info(f"FAISS 인덱스 생성 완료 (벡터 수: {len(texts)})")
            return True
        
        except Exception as e:
            self.logger.error(f"FAISS 인덱스 생성 실패: {e}")
            return False
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, int]:
        """
        기존 FAISS 인덱스에 텍스트를 추가합니다.
        
        Args:
            texts: 추가할 텍스트 리스트
            metadatas: 메타데이터 리스트 (선택)
        
        Returns:
            Tuple[bool, int]: (성공 여부, 추가 전 벡터 인덱스 시작 위치)
        """
        if self.vectorstore is None:
            # 인덱스가 없으면 새로 생성
            if not self.load():
                self.logger.info("기존 인덱스 없음. 새 인덱스 생성")
                success = self.create_from_documents(texts, metadatas)
                return success, 0
        
        try:
            # 현재 벡터 수 저장
            start_index = self.vectorstore.index.ntotal
            
            # LangChain add_texts() 사용
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas or [{}] * len(texts)
            )
            
            self.logger.info(
                f"벡터 추가 완료: {len(texts)}개 "
                f"(시작 인덱스: {start_index}, 총 {self.vectorstore.index.ntotal}개)"
            )
            return True, start_index
        
        except Exception as e:
            self.logger.error(f"벡터 추가 실패: {e}")
            return False, 0
    
    def save(self) -> bool:
        """
        FAISS 인덱스를 로컬 파일로 저장합니다.
        
        Returns:
            bool: 저장 성공 여부
        """
        if self.vectorstore is None:
            self.logger.error("저장할 벡터스토어가 없습니다.")
            return False
        
        try:
            # 디렉토리 생성
            self.vector_path.parent.mkdir(parents=True, exist_ok=True)
            
            # LangChain save_local() 사용
            index_dir = str(self.vector_path.parent)
            index_name = self.vector_path.stem
            
            self.vectorstore.save_local(
                folder_path=index_dir,
                index_name=index_name
            )
            
            self.logger.info(f"FAISS 인덱스 저장 완료: {self.vector_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"FAISS 인덱스 저장 실패: {e}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        유사도 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
            filter_metadata: 메타데이터 필터 (선택)
        
        Returns:
            List[Tuple[Document, float]]: (문서, 유사도 점수) 리스트
        """
        if self.vectorstore is None:
            if not self.load():
                self.logger.error("로드할 벡터스토어가 없습니다.")
                return []
        
        try:
            # LangChain similarity_search_with_score() 사용
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_metadata
            )
            
            self.logger.info(f"검색 완료: {len(results)}개 결과 반환")
            return results
        
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            return []
    
    def get_vector_count(self) -> int:
        """
        현재 저장된 벡터 수를 반환합니다.
        
        Returns:
            int: 벡터 개수
        """
        if self.vectorstore is None:
            if not self.load():
                return 0
        
        return self.vectorstore.index.ntotal if self.vectorstore else 0
    
    def remove_by_file_hash(self, file_hash: str) -> bool:
        """
        특정 파일 해시에 해당하는 모든 벡터를 삭제합니다.
        
        Args:
            file_hash: 삭제할 파일의 해시값
        
        Returns:
            bool: 삭제 성공 여부
        """
        return self.remove_by_metadata({'file_hash': file_hash})
    
    def rebuild_without_file(self, file_hash: str) -> bool:
        """
        특정 파일을 제외하고 FAISS 인덱스를 재구성합니다.
        
        Args:
            file_hash: 제외할 파일 해시
        
        Returns:
            bool: 재구성 성공 여부
        """
        if self.vectorstore is None:
            if not self.load():
                return False
        
        # 필터링
        texts, metadatas = [], []
        for idx in range(self.vectorstore.index.ntotal):
            doc_id = self.vectorstore.index_to_docstore_id.get(idx)
            if doc_id is None:
                continue
            
            doc = self.vectorstore.docstore.search(doc_id)
            if doc and doc.metadata.get('file_hash') != file_hash:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
        
        # 재구성
        if not texts:
            return self._create_dummy_index()
        
        success = self.create_from_documents(texts, metadatas)
        self.logger.info(f"파일 제거 후 재구성: {len(texts)}개 벡터")
        return success
    
    def remove_by_metadata(
        self, 
        filter_metadata: Dict[str, Any]
    ) -> bool:
        """
        메타데이터 필터 조건에 맞는 벡터를 삭제합니다.
        
        Args:
            filter_metadata: 삭제 조건 (예: {'file_hash': 'abc123'})
        
        Returns:
            bool: 삭제 성공 여부
        """
        if self.vectorstore is None:
            if not self.load():
                self.logger.error("로드할 벡터스토어가 없습니다.")
                return False
        
        try:
            # 기존 벡터 추출
            index = self.vectorstore.index
            docstore = self.vectorstore.docstore
            index_to_id = self.vectorstore.index_to_docstore_id
            
            # 필터링: 삭제 조건에 맞지 않는 벡터만 추출
            keep_indices = []
            keep_docs = []
            
            for idx in range(index.ntotal):
                doc_id = index_to_id.get(idx)
                if doc_id is None:
                    continue
                
                doc = docstore.search(doc_id)
                if doc is None:
                    continue
                
                # 메타데이터 필터 체크
                match = all(
                    doc.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
                
                if not match:
                    keep_indices.append(idx)
                    keep_docs.append(doc)
            
            removed_count = index.ntotal - len(keep_indices)
            
            if removed_count == 0:
                self.logger.info("삭제 대상 벡터 없음")
                return True
            
            # 인덱스 재구성
            self._rebuild_index(keep_indices, keep_docs)
            
            self.logger.info(
                f"벡터 삭제 완료: {removed_count}개 제거 "
                f"({len(keep_docs)}개 남음)"
            )
            return True
        
        except Exception as e:
            self.logger.error(f"벡터 삭제 실패: {e}")
            return False
    
    def _rebuild_index(
        self, 
        keep_indices: List[int], 
        keep_docs: List[Document]
    ) -> None:
        """
        필터링된 벡터로 FAISS 인덱스를 재구성합니다.
        
        Args:
            keep_indices: 유지할 벡터의 인덱스 리스트
            keep_docs: 유지할 Document 리스트
        """
        if not keep_docs:
            # 빈 인덱스 생성
            self._create_dummy_index()
            return
        
        # 기존 벡터 추출
        old_index = self.vectorstore.index
        vectors = np.array([
            old_index.reconstruct(int(idx)) 
            for idx in keep_indices
        ])
        
        # 새 인덱스 생성
        dimension = vectors.shape[1]
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(vectors)
        
        
        docstore = InMemoryDocstore({
            str(i): doc for i, doc in enumerate(keep_docs)
        })
        index_to_id = {i: str(i) for i in range(len(keep_docs))}
        
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=new_index,
            docstore=docstore,
            index_to_docstore_id=index_to_id,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

    def summary(self) -> None:
        """
        벡터스토어의 현재 상태를 테이블 형식으로 출력합니다.
        
        출력 정보:
        - 기본 정보: 벡터 수, 임베딩 모델, 차원, 파일 경로, 청크 개수, 토탈 사이즈
        - 메타데이터 통계: file_hash별 벡터 수 분포
        - 인덱스 정보: 차원, 인덱스 타입
        """
        if self.vectorstore is None:
            if not self.load():
                self.logger.warning("로드할 벡터스토어가 없습니다.")
                return
        
        # 1. 기본 정보
        vector_count = self.vectorstore.index.ntotal
        dimension = self.vectorstore.index.d
        index_type = type(self.vectorstore.index).__name__
        
        # 파일 사이즈 계산
        faiss_file = self.vector_path.parent / f"{self.vector_path.stem}.faiss"
        pkl_file = self.vector_path.parent / f"{self.vector_path.stem}.pkl"
        
        total_size_bytes = 0
        if faiss_file.exists():
            total_size_bytes += faiss_file.stat().st_size
        if pkl_file.exists():
            total_size_bytes += pkl_file.stat().st_size
        
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        self.logger.info("=" * 80)
        self.logger.info("VectorStore Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"벡터 수 (Vector Count)       : {vector_count:,}")
        self.logger.info(f"청크 개수 (Chunk Count)       : {vector_count:,}")
        self.logger.info(f"차원 (Dimension)              : {dimension}")
        self.logger.info(f"인덱스 타입 (Index Type)      : {index_type}")
        self.logger.info(f"임베딩 모델 (Embedding Model) : {self.embedding_model}")
        self.logger.info(f"파일 경로 (File Path)         : {self.vector_path}")
        self.logger.info(f"토탈 사이즈 (Total Size)      : {total_size_mb:.2f} MB")
        
        if vector_count == 0:
            self.logger.info("=" * 80)
            return
        
        # 2. 메타데이터 통계
        docstore = self.vectorstore.docstore
        index_to_id = self.vectorstore.index_to_docstore_id
        
        file_hash_counts: Dict[str, int] = {}
        file_name_map: Dict[str, str] = {}
        
        for idx in range(vector_count):
            doc_id = index_to_id.get(idx)
            if doc_id is None:
                continue
            
            doc = docstore.search(doc_id)
            if doc is None:
                continue
            
            file_hash = doc.metadata.get('file_hash', 'unknown')
            file_name = doc.metadata.get('file_name', 'unknown')
            
            file_hash_counts[file_hash] = file_hash_counts.get(file_hash, 0) + 1
            if file_hash not in file_name_map:
                file_name_map[file_hash] = file_name
        
        # 3. 테이블 출력 (file_hash별)
        self.logger.info("-" * 80)
        self.logger.info(f"파일별 벡터 분포 (총 {len(file_hash_counts)}개 파일)")
        self.logger.info("-" * 80)
        
        # 벡터 수 기준 내림차순 정렬
        sorted_items = sorted(
            file_hash_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        self.logger.info(f"{'File Name':<40} {'Hash':<12} {'Count':>10}")
        self.logger.info("-" * 80)
        
        for file_hash, count in sorted_items:
            file_name = file_name_map.get(file_hash, 'unknown')
            # 파일명이 길면 자르기
            display_name = (file_name[:37] + '...') if len(file_name) > 40 else file_name
            display_hash = file_hash[:8] if file_hash != 'unknown' else file_hash
            
            self.logger.info(f"{display_name:<40} {display_hash:<12} {count:>10,}")
        
        self.logger.info("=" * 80)