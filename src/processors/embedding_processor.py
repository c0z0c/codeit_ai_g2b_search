# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db.documents_db import DocumentsDB
from src.config import get_config
from src.utils.logging_config import get_logger
from src.vectorstore import VectorStoreManager

import re
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
        
        # 5. 복원 (역순: 나중에 보호한 것부터 복원하여 중첩 방지)
        # 먼저 블록 복원 (코드, 수식)
        for placeholder in sorted(protected_blocks.keys(), reverse=True):
            text = text.replace(placeholder, protected_blocks[placeholder])
        
        # 나중에 마커 복원 (페이지 구분자)
        for placeholder in sorted(protected_markers.keys(), reverse=True):
            text = text.replace(placeholder, protected_markers[placeholder])
        
        return text.strip()

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> bool:
        """
        문서를 페이지 단위로 청킹하고 통합 FAISS 인덱스에 임베딩을 추가합니다.
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
        text_content = doc_info.get('text_content', '')
        
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
            
            cleaned_text = self.clean_markdown_text(page_text)
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
                        metadatas.append({
                            'file_hash': file_hash,
                            'file_name': file_name,
                            'start_page': buffer_pages[0],
                            'end_page': buffer_pages[-1],
                            'chunk_type': 'split',
                            'chunk_index': len(chunks) - 1,
                            'embedding_version': self.embedding_model,
                            'created_at': datetime.now().isoformat()
                        })
                else:
                    # 단일 청크로 추가
                    chunks.append(buffer_text.strip())
                    metadatas.append({
                        'file_hash': file_hash,
                        'file_name': file_name,
                        'start_page': buffer_pages[0],
                        'end_page': buffer_pages[-1],
                        'chunk_type': 'merged' if len(buffer_pages) > 1 else 'single',
                        'chunk_index': len(chunks) - 1,
                        'embedding_version': self.embedding_model,
                        'created_at': datetime.now().isoformat()
                    })
                
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
        return True
