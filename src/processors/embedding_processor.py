# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import hashlib
import json
import importlib

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db.documents_db import DocumentsDB
from src.config import get_config
from src.utils.logging_config import get_logger
from src.vectorstore import VectorStoreManager

from src.db import documents_db
importlib.reload(documents_db)
from src.db.documents_db import DocumentsDB

from src import vectorstore
importlib.reload(vectorstore)
from src.vectorstore import VectorStoreManager

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
        #try:
        self.vector_manager = VectorStoreManager(config=self.config)
        self.vector_manager.load()
        # except ImportError as e:
        #     self.logger.error(f"VectorStoreManager 초기화 실패: {e}")
        #     self.vector_manager = None
        #     raise ValueError("VectorStoreManager 초기화 실패") from e
            
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
            self.text_splitter = None  # 이 부분도 실행되지 않음
            self.logger.error("LangChain이 설치되지 않았습니다.")
            raise ValueError("LangChain이 설치되지 않았습니다.")

        self.sync_with_docs_db(self.config.OPENAI_API_KEY)

    def clean_markdown_text(self, text: str) -> str:
        """
        통합 마크다운 전처리: config 옵션에 따라 선택적 처리
        
        config.MARKDOWN_PROTECT_BLOCKS: 보호할 블록 타입 ['code', 'math', 'inline_math', 'mermaid']
        config.MARKDOWN_REMOVE_ELEMENTS: 제거할 요소 ['html', 'images', 'links', 'emphasis', 'headers']
        config.MARKDOWN_MAX_LINES: 블록 타입별 최대 라인 수 {'code': 100, 'math': 50}
        
        Args:
            text (str): 원본 마크다운 텍스트
            
        Returns:
            str: 정제된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 특수 블록 보호 (코드, 수식 등)
        protected_blocks = {}
        block_counter = 0
        
        # 코드 블록 보호 (최우선)
        if 'code' in self.config.MARKDOWN_PROTECT_BLOCKS:
            # 4백틱 코드 블록 (중첩 방지)
            pattern_4 = r'````[\s\S]*?````'
            for match in re.finditer(pattern_4, text):
                placeholder = f"XPROTECTEDXCODE4X{block_counter}X"
                protected_blocks[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                block_counter += 1
            
            # 3백틱 코드 블록
            pattern_3 = r'```[\s\S]*?```'
            for match in re.finditer(pattern_3, text):
                placeholder = f"XPROTECTEDXCODE3X{block_counter}X"
                protected_blocks[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                block_counter += 1
        
        # 수식 블록 보호
        if 'math' in self.config.MARKDOWN_PROTECT_BLOCKS:
            pattern_math = r'\$\$[\s\S]*?\$\$'
            for match in re.finditer(pattern_math, text):
                placeholder = f"XPROTECTEDXMATHX{block_counter}X"
                protected_blocks[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                block_counter += 1
        
        # 인라인 수식 보호
        if 'inline_math' in self.config.MARKDOWN_PROTECT_BLOCKS:
            pattern_inline = r'(?<!\$)\$(?!\$)[^\$\n]+?\$(?!\$)'
            for match in re.finditer(pattern_inline, text):
                placeholder = f"XPROTECTEDXINLINEX{block_counter}X"
                protected_blocks[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                block_counter += 1
        
        # 2. 페이지 마커 보호 (항상 보호) - 순차 교체로 중복 방지
        protected_markers = {}
        marker_counter = 0
        
        # ERROR_PAGE_MARKER 보호
        error_marker = self.config.ERROR_PAGE_MARKER
        for match in re.finditer(re.escape(error_marker), text):
            placeholder = f"XPROTECTEDXMARKERX{marker_counter}X"
            protected_markers[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder, 1)
            marker_counter += 1
        
        # EMPTY_PAGE_MARKER 보호
        empty_marker = self.config.EMPTY_PAGE_MARKER
        for match in re.finditer(re.escape(empty_marker), text):
            placeholder = f"XPROTECTEDXMARKERX{marker_counter}X"
            protected_markers[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder, 1)
            marker_counter += 1
        
        # 페이지 번호 마커 보호
        page_marker_pattern = r'--- 페이지 \d+ ---'
        for match in re.finditer(page_marker_pattern, text):
            placeholder = f"XPROTECTEDXMARKERX{marker_counter}X"
            protected_markers[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder, 1)
            marker_counter += 1
        
        # 3. 탈출문자 처리 (보호 블록 외부만)
        text = re.sub(r'\\([*_\[\]()#+-])', r'\1', text)
        
        # 3. 요소 제거 (config 옵션에 따라)
        if 'html' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        if 'images' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        if 'links' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        if 'emphasis' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^\*]+)\*', r'\1', text)
            text = re.sub(r'~~([^~]+)~~', r'\1', text)
            text = re.sub(r'__([^_]+)__', r'\1', text)
            text = re.sub(r'_([^_]+)_', r'\1', text)
            text = re.sub(r'_([^_]+)_', r'\1', text)
        
        if 'headers' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        if 'blockquotes' in self.config.MARKDOWN_REMOVE_ELEMENTS:
            text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
            
        # if 'lists' in self.config.MARKDOWN_REMOVE_ELEMENTS:
        #     text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # 순서 없는 리스트
        #     # text = re.sub(r'^\s*\d+[.)]\s+', '', text, flags=re.MULTILINE)  # 순서 있는 리스트            
                    
        # 4. 공백 정리
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # 각 라인 시작 공백 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text) 
        
        # 4-1. 빈 테이블 행 제거 (파이프만 있고 내용 없는 행)
        # empty_row_count = len(re.findall(r'^\|[\|\s]*\|$', text, flags=re.MULTILINE))
        text = re.sub(r'^\|[\|\s]*\|$', '', text, flags=re.MULTILINE)
        # if empty_row_count > 0:
        #     self.logger.debug(f"빈 테이블 행 제거: {empty_row_count}개")
        
        # 4-2. 목차 구분선 축약 (·····················... → ···)
        # dot_count_before = len(re.findall(r'·{4,}', text))
        text = re.sub(r'·{4,}', '···', text)
        # if dot_count_before > 0:
        #     self.logger.debug(f"목차 구분선 축약: {dot_count_before}개 패턴 변환")
        
        # 5. 복원 (역순: 나중에 보호한 것부터 복원하여 중첩 방지)
        # 먼저 블록 복원 (코드, 수식)
        for placeholder in sorted(protected_blocks.keys(), reverse=True):
            text = text.replace(placeholder, protected_blocks[placeholder])
        
        # 나중에 마커 복원 (페이지 구분자)
        for placeholder in sorted(protected_markers.keys(), reverse=True):
            text = text.replace(placeholder, protected_markers[placeholder])
        
        return text.strip()

    def clean_page_text(self, page_text: str) -> str:
        """
        페이지별 텍스트 정제: 페이지 마커 제거
        
        - ERROR_PAGE_MARKER, EMPTY_PAGE_MARKER 제거
        - 페이지 번호 마커는 유지하지 않음
        
        Args:
            page_text (str): 개별 페이지 텍스트
            
        Returns:
            str: 정제된 페이지 텍스트 (마커 제거됨)
        """
        if not page_text or not isinstance(page_text, str):
            return ""
        
        text = page_text
        
        # ERROR_PAGE_MARKER 제거
        text = text.replace(self.config.ERROR_PAGE_MARKER, "")
        
        # EMPTY_PAGE_MARKER 제거
        text = text.replace(self.config.EMPTY_PAGE_MARKER, "")
        
        # 페이지 번호 마커 제거 (--- 페이지 N --- 형식)
        # config.PAGE_MARKER_FORMAT을 기반으로 정규식 패턴 생성
        page_marker_pattern = r'---\s*페이지\s+\d+\s*---'
        text = re.sub(page_marker_pattern, '', text)
        
        # 공백 정리
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text

    def _dump_chunk(self, chunk_text: str, metadata: Dict[str, Any], safe_name: str) -> None:
        """
        청크를 YAML 프론트매터와 함께 마크다운 파일로 저장
        
        Args:
            chunk_text (str): 청크 텍스트
            metadata (Dict[str, Any]): 청크 메타데이터
            safe_name (str): 안전한 파일명 (확장자 제외)
        """
        if not self.config.MARKER_DUMP_ENABLED:
            return
        
        try:
            dump_dir = Path(self.config.MARKER_DUMP_PATH) / 'chunks'
            dump_dir.mkdir(parents=True, exist_ok=True)
            
            chunk_index = metadata.get('chunk_index', 0)
            dump_file = dump_dir / f"{safe_name}_chunk_{chunk_index:04d}.md"
            
            # YAML 프론트매터 생성
            yaml_lines = [
                "---",
                f"file_hash: {metadata.get('file_hash', '')}",
                f"file_name: {metadata.get('file_name', '')}",
                f"start_page: {metadata.get('start_page', 0)}",
                f"end_page: {metadata.get('end_page', 0)}",
                f"chunk_type: {metadata.get('chunk_type', '')}",
                f"chunk_index: {chunk_index}",
                f"config_chunk_size: {metadata.get('config_chunk_size', 0)}",
                f"config_chunk_overlap: {metadata.get('config_chunk_overlap', 0)}",
                f"embedding_version: {metadata.get('embedding_version', '')}",
                f"created_at: {metadata.get('created_at', '')}",
                "---",
                ""
            ]
            
            with open(dump_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yaml_lines))
                f.write(chunk_text)
            
            self.logger.debug(f"청크 {chunk_index} 저장: {dump_file}")
        except Exception as e:
            self.logger.warning(f"청크 {metadata.get('chunk_index', 0)} 저장 실패: {e}")

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> bool:
        """
        문서를 페이지 단위로 청킹하고 통합 FAISS 인덱스에 임베딩을 추가합니다.

        메타데이터는 다음과 같습니다:
            - file_hash (str): 처리 중인 파일의 해시 값
            - file_name (str): 처리 중인 파일의 이름
            - start_page (int): 청크가 시작된 페이지 번호
            - end_page (int): 청크가 끝난 페이지 번호
            - chunk_type (str): 청크 유형 ('split', 'merged', 'single')
            - chunk_index (int): 청크의 인덱스 번호
            - embedding_config_hash (str): 파일 및 설정을 통합한 해시 값
            - chunk_hash (str): 청크 내용 기반의 해시 값
            - config_chunk_size (int): 청크 크기 설정 값
            - config_chunk_overlap (int): 청크 간 중첩 크기 설정 값
            - config_chunking_mode (str): 청킹 모드 설정 값
            - config_chunk_separators (str): 청킹 시 사용된 구분자 설정 값 (JSON 형식)
            - config_markdown_max_lines (str): 마크다운 블록별 최대 라인 수 설정 값 (JSON 형식)
            - embedding_version (str): 사용된 임베딩 모델 버전
            - created_at (str): 청크 생성 시각 (ISO 형식)

        Args:
            file_hash (str): 처리할 파일의 해시 값
            api_key (Optional[str]): OpenAI API 키 (선택)

        Returns:
            bool: 처리 성공 여부
        """
        if not LANGCHAIN_AVAILABLE or self.vector_manager is None:
            self.logger.error("필수 패키지가 설치되지 않았습니다.")
            return False
        
        self.logger.info(f"임베딩 처리 시작: {file_hash[:16]}...")
        
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key
        
        # 문서 정보 가져오기
        docs = self.docs_db.search_documents(file_hash)
        if not docs or len(docs) == 0:
            self.logger.error(f"문서를 찾을 수 없습니다: {file_hash[:16]}...")
            return False
        
        doc_info = docs[0]
        file_name = doc_info.get('file_name', 'unknown')
        
        # embedding_config_hash 계산 (사전 검사용)
        embedding_config_hash = self.vector_manager.calculate_embedding_config_hash(file_hash)
        
        # chunk_map에서 기존 벡터 확인 (파일 단위 스킵 검사)
        if hasattr(self.vector_manager, 'chunk_map') and self.vector_manager.chunk_map:
            existing_vectors = [
                (chunk_idx, faiss_idx, chunk_hash, config_hash)
                for (fh, chunk_idx), (faiss_idx, chunk_hash, config_hash) 
                in self.vector_manager.chunk_map.items()
                if fh == file_hash
            ]
            
            if existing_vectors:
                # 첫 번째 벡터의 config_hash 추출 (모든 청크가 동일 config_hash 가정)
                old_config_hash = existing_vectors[0][3]
                
                if old_config_hash == embedding_config_hash:
                    # 동일 설정 → 스킵
                    self.logger.info(
                        f"임베딩 스킵: 동일 설정 "
                        f"(file_hash={file_hash[:16]}..., existing_count={len(existing_vectors)}, "
                        f"config_hash={embedding_config_hash[:8]}...)"
                    )
                    return True
                else:
                    # 설정 변경 → 기존 벡터 삭제
                    self.logger.warning(
                        f"설정 변경 감지: 기존 벡터 삭제 후 재임베딩 "
                        f"(file_hash={file_hash[:16]}..., "
                        f"old_hash={old_config_hash[:8]}... → new_hash={embedding_config_hash[:8]}...)"
                    )
                    if not self.vector_manager.remove_by_file_hash(file_hash):
                        self.logger.error(f"기존 벡터 삭제 실패: {file_hash[:16]}...")
                        return False
            else:
                self.logger.debug(f"chunk_map에 file_hash 없음: 신규 임베딩 진행")
        else:
            self.logger.debug("chunk_map 비어있음: 신규 임베딩 진행")
        
        text_content = doc_info.get('text_content', '')
        text_content = self.clean_markdown_text(text_content)
        
        # MARKER_DUMP_ENABLED: 전체 문서 clean 결과 저장
        if self.config.MARKER_DUMP_ENABLED:
            import os
            dump_dir = Path(self.config.MARKER_DUMP_PATH) / 'clean'
            dump_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명에서 확장자 제거 및 안전한 이름으로 변환
            safe_name = Path(file_name).stem.replace('/', '_').replace('\\', '_')
            dump_file = dump_dir / f"{safe_name}_clean.md"
            
            with open(dump_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            self.logger.debug(f"전체 문서 clean 저장: {dump_file}")
        
        if not text_content:
            self.logger.warning(f"text_content가 비어있습니다: {file_hash[:16]}...")
            return False
        
        # text_content에서 페이지 단위로 분리
        page_pattern = r'--- 페이지 (\d+) ---'
        page_splits = re.split(page_pattern, text_content)
        
        page_data = []
        for i in range(1, len(page_splits), 2):
            page_num = int(page_splits[i])
            page_text = page_splits[i+1] if i+1 < len(page_splits) else ""
            
            # 빈페이지/오류페이지 스킵
            if '--- [빈페이지] ---' in page_text or '--- [오류페이지] ---' in page_text:
                continue
            
            cleaned_text = self.clean_page_text(page_text)
            
            # MARKER_DUMP_ENABLED: 페이지별 clean 결과 저장
            if self.config.MARKER_DUMP_ENABLED and cleaned_text.strip():
                dump_dir = Path(self.config.MARKER_DUMP_PATH) / 'pages'
                dump_dir.mkdir(parents=True, exist_ok=True)
                safe_name = Path(file_name).stem.replace('/', '_').replace('\\', '_')
                dump_file = dump_dir / f"{safe_name}_clean_page_{page_num:04d}.md"
                
                with open(dump_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                self.logger.debug(f"페이지 {page_num} clean 저장: {dump_file}")
            
            if cleaned_text.strip():
                page_data.append({
                    'page_number': page_num,
                    'text': cleaned_text,
                    'length': len(cleaned_text)
                })
        
        if not page_data:
            self.logger.warning(f"전처리 후 유효한 페이지 없음: {file_hash[:16]}...")
            return False
        
        self.logger.debug(
            f"전처리 완료: {len(page_data)}개 페이지 "
            f"(총 {sum(p['length'] for p in page_data):,} 문자)"
        )
        
        # 페이지 단위 청킹
        chunks = []
        metadatas = []
        
        # safe_name 미리 생성 (청크 덤프용)
        safe_name = Path(file_name).stem.replace('/', '_').replace('\\', '_')
        
        buffer_pages = []
        buffer_text = ""
        
        for i, page in enumerate(page_data):
            buffer_pages.append(page['page_number'])
            buffer_text += page['text'] + "\n\n"
            
            # 버퍼가 CHUNK_SIZE 이상이거나 마지막 페이지인 경우
            if len(buffer_text) >= self.chunk_size or i == len(page_data) - 1:
                
                # CHUNK_SIZE 초과 시 분할
                if len(buffer_text) > self.chunk_size:
                    sub_chunks = self.text_splitter.split_text(buffer_text)
                    
                    # 각 sub_chunk에 페이지 범위 할당
                    for sub_chunk in sub_chunks:
                        chunks.append(sub_chunk)
                        metadata = {
                            'file_hash': file_hash,
                            'file_name': file_name,
                            'start_page': buffer_pages[0],
                            'end_page': buffer_pages[-1],
                            'chunk_type': 'split',
                            'chunk_index': len(chunks) - 1,
                            
                            # embedding_config_hash (파일+config 통합 hash)
                            'embedding_config_hash': embedding_config_hash,
                            
                            # chunk_hash (내용 기반 hash)
                            'chunk_hash': hashlib.sha256(sub_chunk.encode('utf-8')).hexdigest(),
                            
                            # 재현성 정보 (디버깅용)
                            'config_chunk_size': self.chunk_size,
                            'config_chunk_overlap': self.chunk_overlap,
                            'config_chunking_mode': self.chunking_mode,
                            'config_chunk_separators': json.dumps(self.config.CHUNK_SEPARATORS),
                            'config_markdown_max_lines': json.dumps(self.config.MARKDOWN_MAX_LINES),
                            
                            'embedding_version': self.embedding_model,
                            'created_at': datetime.now().isoformat()
                        }
                        metadatas.append(metadata)
                        
                        # 청크 덤프
                        self._dump_chunk(sub_chunk, metadata, safe_name)
                else:
                    # 단일 청크로 추가
                    chunk_text = buffer_text.strip()
                    chunks.append(chunk_text)
                    metadata = {
                        'file_hash': file_hash,
                        'file_name': file_name,
                        'start_page': buffer_pages[0],
                        'end_page': buffer_pages[-1],
                        'chunk_type': 'merged' if len(buffer_pages) > 1 else 'single',
                        'chunk_index': len(chunks) - 1,
                        
                        # embedding_config_hash (파일+config 통합 hash)
                        'embedding_config_hash': embedding_config_hash,
                        
                        # chunk_hash (내용 기반 hash)
                        'chunk_hash': hashlib.sha256(chunk_text.encode('utf-8')).hexdigest(),
                        
                        # 재현성 정보 (디버깅용)
                        'config_chunk_size': self.chunk_size,
                        'config_chunk_overlap': self.chunk_overlap,
                        'config_chunking_mode': self.chunking_mode,
                        'config_chunk_separators': json.dumps(self.config.CHUNK_SEPARATORS),
                        'config_markdown_max_lines': json.dumps(self.config.MARKDOWN_MAX_LINES),
                        
                        'embedding_version': self.embedding_model,
                        'created_at': datetime.now().isoformat()
                    }
                    metadatas.append(metadata)
                    
                    # 청크 덤프
                    self._dump_chunk(buffer_text.strip(), metadata, safe_name)
                
                # 버퍼 초기화
                buffer_pages = []
                buffer_text = ""
        
        total_chunks = len(chunks)
        self.logger.info(
            f"페이지 단위 청킹 완료: {total_chunks}개 청크 "
            f"(단일: {sum(1 for m in metadatas if m['chunk_type']=='single')}, "
            f"병합: {sum(1 for m in metadatas if m['chunk_type']=='merged')}, "
            f"분할: {sum(1 for m in metadatas if m['chunk_type']=='split')})"
        )
        
        # VectorStoreManager를 통해 벡터 추가
        success = self.vector_manager.add_texts(chunks, metadatas)
        
        if not success:
            self.logger.error("벡터 추가 실패")
            return False
        
        if not self.vector_manager.save():
            self.logger.error("FAISS 인덱스 저장 실패")
            return False
        
        self.logger.info(
            f"임베딩 처리 완료: {file_hash[:16]}... "
            f"(추가 {total_chunks}개 청크, 총 {self.vector_manager.get_vector_count()}개)"
        )
        self.vector_manager.save()
        return True
    
    def sync_with_docs_db(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        docs_db와 vector_manager 간 데이터 동기화
        
        동기화 로직:
        1. docs_db에만 존재하는 파일 → 임베딩 벡터 추가 (process_document 호출)
        2. vector_manager에만 존재하는 파일 → 임베딩 벡터 삭제 (remove_by_file_hash 호출)
        3. 양쪽 모두 존재하지만 config 변경 → 재임베딩 (process_document 호출)
        
        Args:
            api_key (Optional[str]): OpenAI API 키 (선택)
        
        Returns:
            Dict[str, Any]: 동기화 결과 통계
                - total_docs: 총 문서 수 (docs_db)
                - total_vectors: 총 벡터 파일 수 (vector_manager)
                - added: 추가된 파일 수
                - removed: 삭제된 파일 수
                - updated: 갱신된 파일 수
                - skipped: 스킵된 파일 수
                - failed: 실패한 파일 수
                - details: 상세 정보 {'added': [], 'removed': [], 'updated': [], 'failed': []}
        
        Example:
            >>> proc = EmbeddingProcessor()
            >>> result = proc.sync_with_docs_db()
            >>> print(f"추가: {result['added']}, 삭제: {result['removed']}, 갱신: {result['updated']}")
        """
        if not LANGCHAIN_AVAILABLE or self.vector_manager is None:
            self.logger.error("필수 패키지가 설치되지 않았습니다.")
            return {
                'total_docs': 0, 'total_vectors': 0,
                'added': 0, 'removed': 0, 'updated': 0, 'skipped': 0, 'failed': 0,
                'details': {'added': [], 'removed': [], 'updated': [], 'failed': []}
            }
        
        self.logger.info("=" * 80)
        self.logger.info("docs_db와 vector_manager 동기화 시작")
        self.logger.info("=" * 80)
        
        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key
        
        # 1. docs_db의 모든 파일 해시 조회
        all_docs = self.docs_db.get_documents_all()
        docs_file_hashes = {doc['file_hash']: doc['file_name'] for doc in all_docs}
        
        self.logger.info(f"docs_db 파일 수: {len(docs_file_hashes)}")
        
        # 2. vector_manager의 모든 파일 해시 조회 (chunk_map 활용)
        vector_file_hashes = set()
        if hasattr(self.vector_manager, 'chunk_map') and self.vector_manager.chunk_map:
            vector_file_hashes = {
                file_hash 
                for (file_hash, chunk_idx) in self.vector_manager.chunk_map.keys()
            }
        
        self.logger.info(f"vector_manager 파일 수: {len(vector_file_hashes)}")
        
        # 3. 차집합 계산
        only_in_docs = set(docs_file_hashes.keys()) - vector_file_hashes
        only_in_vectors = vector_file_hashes - set(docs_file_hashes.keys())
        both = set(docs_file_hashes.keys()) & vector_file_hashes
        
        self.logger.info(f"docs_db에만 존재: {len(only_in_docs)}개")
        self.logger.info(f"vector_manager에만 존재: {len(only_in_vectors)}개")
        self.logger.info(f"양쪽 모두 존재: {len(both)}개")
        self.logger.info("-" * 80)
        
        # 결과 통계
        result = {
            'total_docs': len(docs_file_hashes),
            'total_vectors': len(vector_file_hashes),
            'added': 0,
            'removed': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'details': {
                'added': [],
                'removed': [],
                'updated': [],
                'failed': []
            }
        }
        
        # 4. docs_db에만 존재하는 파일 → 임베딩 추가
        if only_in_docs:
            self.logger.info(f"[추가 작업] {len(only_in_docs)}개 파일 임베딩 추가 시작")
            for file_hash in only_in_docs:
                file_name = docs_file_hashes[file_hash]
                try:
                    self.logger.info(f"  추가: {file_name} ({file_hash[:16]}...)")
                    success = self.process_document(file_hash, api_key)
                    if success:
                        result['added'] += 1
                        result['details']['added'].append({
                            'file_hash': file_hash,
                            'file_name': file_name
                        })
                    else:
                        self.logger.error(f"  갱신 실패: {file_name}")
                        result['failed'] += 1
                        result['details']['failed'].append({
                            'file_hash': file_hash,
                            'file_name': file_name,
                            'reason': 'process_document 실패'
                        })
                except Exception as e:
                    self.logger.error(f"  추가 실패: {file_name} - {e}")
                    result['failed'] += 1
                    result['details']['failed'].append({
                        'file_hash': file_hash,
                        'file_name': file_name,
                        'reason': str(e)
                    })
            self.logger.info(f"[추가 완료] 성공: {result['added']}, 실패: {len([f for f in result['details']['failed'] if f['file_hash'] in only_in_docs])}")
            self.logger.info("-" * 80)
        
        # 5. vector_manager에만 존재하는 파일 → 임베딩 삭제
        if only_in_vectors:
            self.logger.info(f"[삭제 작업] {len(only_in_vectors)}개 파일 임베딩 삭제 시작")
            for file_hash in only_in_vectors:
                file_name = docs_file_hashes.get(file_hash)
                if file_name is None:
                    self.logger.warning(f"Orphaned embedding detected: {file_hash} not in documents DB")
                    continue  # 또는 삭제 로직 추가                
                try:
                    self.logger.info(f"  삭제: {file_hash[:16]}...")
                    success = self.vector_manager.remove_by_file_hash(file_hash)
                    if success:
                        result['removed'] += 1
                        result['details']['removed'].append({
                            'file_hash': file_hash
                        })
                    else:
                        self.logger.error(f"  갱신 실패: {file_name}")
                        result['failed'] += 1
                        result['details']['failed'].append({
                            'file_hash': file_hash,
                            'file_name': 'unknown',
                            'reason': 'remove_by_file_hash 실패'
                        })
                except Exception as e:
                    self.logger.error(f"  삭제 실패: {file_hash[:16]}... - {e}")
                    result['failed'] += 1
                    result['details']['failed'].append({
                        'file_hash': file_hash,
                        'file_name': 'unknown',
                        'reason': str(e)
                    })
            
            # 삭제 후 저장
            if result['removed'] > 0:
                self.vector_manager.save()
            
            self.logger.info(f"[삭제 완료] 성공: {result['removed']}, 실패: {len([f for f in result['details']['failed'] if f['file_hash'] in only_in_vectors])}")
            self.logger.info("-" * 80)
        
        # 6. 양쪽 모두 존재하지만 config 변경된 파일 → 재임베딩
        if both:
            self.logger.info(f"[갱신 체크] {len(both)}개 파일 config 변경 확인 시작")
            for file_hash in both:
                file_name = docs_file_hashes[file_hash]
                try:
                    # 현재 config 기반 hash 계산
                    current_config_hash = self.vector_manager.calculate_embedding_config_hash(file_hash)
                    
                    # chunk_map에서 기존 config_hash 조회
                    existing_vectors = [
                        (chunk_idx, faiss_idx, chunk_hash, config_hash)
                        for (fh, chunk_idx), (faiss_idx, chunk_hash, config_hash) 
                        in self.vector_manager.chunk_map.items()
                        if fh == file_hash
                    ]
                    
                    if not existing_vectors:
                        self.logger.warning(f"  chunk_map에 없음 (스킵): {file_name}")
                        result['skipped'] += 1
                        continue
                    
                    old_config_hash = existing_vectors[0][3]
                    
                    if old_config_hash != current_config_hash:
                        # config 변경 감지 → 재임베딩
                        self.logger.info(
                            f"  갱신: {file_name} ({file_hash[:16]}...) "
                            f"[{old_config_hash[:8]}... → {current_config_hash[:8]}...]"
                        )
                        success = self.process_document(file_hash, api_key)
                        if success:
                            result['updated'] += 1
                            result['details']['updated'].append({
                                'file_hash': file_hash,
                                'file_name': file_name,
                                'old_config_hash': old_config_hash,
                                'new_config_hash': current_config_hash
                            })
                        else:
                            self.logger.error(f"  갱신 실패: {file_name}")
                            result['failed'] += 1
                            result['details']['failed'].append({
                                'file_hash': file_hash,
                                'file_name': file_name,
                                'reason': 'process_document 실패 (재임베딩)'
                            })
                    else:
                        # config 동일 → 스킵
                        result['skipped'] += 1
                
                except Exception as e:
                    self.logger.error(f"  갱신 체크 실패: {file_name} - {e}")
                    result['failed'] += 1
                    result['details']['failed'].append({
                        'file_hash': file_hash,
                        'file_name': file_name,
                        'reason': str(e)
                    })
            
            self.logger.info(
                f"[갱신 완료] 갱신: {result['updated']}, 스킵: {result['skipped']}, "
                f"실패: {len([f for f in result['details']['failed'] if f['file_hash'] in both])}"
            )
            self.logger.info("-" * 80)
        
        # 최종 결과 출력
        self.logger.info("=" * 80)
        self.logger.info("동기화 완료")
        self.logger.info(f"  추가: {result['added']}개")
        self.logger.info(f"  삭제: {result['removed']}개")
        self.logger.info(f"  갱신: {result['updated']}개")
        self.logger.info(f"  스킵: {result['skipped']}개")
        self.logger.info(f"  실패: {result['failed']}개")
        self.logger.info("=" * 80)
        
        return result
