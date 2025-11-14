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

        # 결과 변환: List[Dict[str, Any]]
        results = []
        for doc, distance in search_results:
            # 메타데이터 전체 복사
            result = dict(doc.metadata)
            
            # 필수 필드 추가/덮어쓰기
            result.update({
                'text': doc.page_content,
                'distance': float(distance)
            })
            
            results.append(result)
            
            self.logger.debug(
                f"  - 청크 {result.get('chunk_index', 0)}: distance={distance:.4f}, "
                f"file={result.get('file_name', 'unknown')}, "
                f"page={result.get('start_page', 0)}-{result.get('end_page', 0)}"
            )

        self.logger.info(f"검색 완료: {len(results)}개 청크 반환")
        return results
    
    def search_page(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        page_window: int = 1,
        sort_by: str = "score"
    ) -> Dict[str, Any]:
        """
        쿼리에 대한 페이지 단위 검색 (±page_window 범위 포함)

        Args:
            query: 검색 쿼리 문자열
            top_k: 상위 k개 검색 (기본값: Config의 TOP_K_SUMMARY)
            filter_metadata: 메타데이터 필터 (선택적)
            api_key: OpenAI API 키 (선택적)
            page_window: 페이지 확장 범위 (기본값: 1)
            sort_by: 정렬 기준 ("score" 또는 "page", 기본값: "score")

        Returns:
            Dict[str, Any]:
            {
                'best_page': Dict[str, Any],
                'page_scores': Dict[Tuple[str, int], float],
                'file_names': List[str],
                'pages': List[Dict[str, Any]]
            }
        """
        # sort_by 검증
        if sort_by not in ["score", "page"]:
            self.logger.warning(f"잘못된 sort_by 값: {sort_by}, 기본값 'score' 사용")
            sort_by = "score"
        
        initial_results = self.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
            api_key=api_key
        )
        
        if not initial_results:
            self.logger.warning("초기 검색 결과 없음")
            return {
                'best_page': {
                    'file_name': '',
                    'file_hash': '',
                    'page_number': -1,
                    'score': float('inf')
                },
                'page_scores': {},
                'file_names': [],
                'pages': []
            }
        
        chunk_dict: Dict[tuple, Dict[str, Any]] = {}
        
        for result in initial_results:
            file_hash = result.get('file_hash')
            start_page = result.get('start_page')
            end_page = result.get('end_page')
            chunk_index = result.get('chunk_index')
            
            if not all([file_hash, start_page is not None, chunk_index is not None]):
                continue
            
            key = (file_hash, chunk_index)
            chunk_dict[key] = result.copy()
            chunk_dict[key]['score'] = result['distance']
            
            page_start = max(1, start_page - page_window)
            page_end = end_page + page_window
            
            extended_chunks = self.vector_manager.get_by_metadata(
                file_hash=file_hash,
                start_page=page_start,
                end_page=page_end
            )
            
            for doc, _ in extended_chunks:
                ext_chunk_index = doc.metadata.get('chunk_index')
                ext_key = (file_hash, ext_chunk_index)
                
                if ext_key in chunk_dict:
                    continue
                
                ext_start_page = doc.metadata.get('start_page', start_page)
                page_diff = abs(ext_start_page - start_page)
                
                base_distance = result['distance']
                penalty = 0.1 * page_diff
                
                chunk_data = dict(doc.metadata)
                chunk_data['text'] = doc.page_content
                chunk_data['distance'] = base_distance
                chunk_data['score'] = base_distance + penalty
                chunk_dict[ext_key] = chunk_data
        
        from collections import defaultdict
        page_groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        file_names_set = set()
        
        for chunk in chunk_dict.values():
            file_hash = chunk.get('file_hash')
            start_page = chunk.get('start_page')
            file_name = chunk.get('file_name')
            
            if file_hash and start_page is not None:
                page_groups[(file_hash, start_page)].append(chunk)
                if file_name:
                    file_names_set.add(file_name)
        
        pages: List[Dict[str, Any]] = []
        
        for (file_hash, page_num), chunks in page_groups.items():
            chunks_sorted = sorted(chunks, key=lambda x: x.get('chunk_index', 0))
            merged_text = '\n'.join(c['text'] for c in chunks_sorted)
            min_score = min(c['score'] for c in chunks_sorted)
            
            pages.append({
                'file_hash': file_hash,
                'file_name': chunks_sorted[0].get('file_name', ''),
                'page_number': page_num,
                'text': merged_text,
                'score': min_score,
                'chunk_count': len(chunks_sorted)
            })
        
        # sort_by 옵션에 따른 정렬
        if sort_by == "score":
            pages.sort(key=lambda x: x['score'])
        else:  # sort_by == "page"
            pages.sort(key=lambda x: (x['file_name'], x['page_number']))
        
        # page_scores를 pages와 동일 순서로 재구성
        page_scores_sorted = {
            (p['file_name'], p['page_number']): p['score'] 
            for p in pages
        }
        
        # best_page: score 기준 최저 (정렬 방식 무관)
        best_page_data = min(pages, key=lambda x: x['score']) if pages else None
        best_page = {
            'file_name': best_page_data['file_name'],
            'file_hash': best_page_data['file_hash'],
            'page_number': best_page_data['page_number'],
            'score': best_page_data['score']
        } if best_page_data else {
            'file_name': '',
            'file_hash': '',
            'page_number': -1,
            'score': float('inf')
        }
        
        self.logger.info(
            f"페이지 검색 완료 (sort_by={sort_by}): "
            f"best_page={best_page['file_name']}:p{best_page['page_number']} "
            f"(score={best_page['score']:.4f}), total_pages={len(pages)}, files={len(file_names_set)}"
        )
        
        return {
            'best_page': best_page,
            'page_scores': page_scores_sorted,
            'file_names': sorted(file_names_set),
            'pages': pages
        }