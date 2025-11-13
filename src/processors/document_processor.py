# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import tiktoken

try:
    import pymupdf
    import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from src.config import get_config
from src.utils.logging_config import get_logger
from tqdm.notebook import tqdm

import importlib
from src.db import documents_db
importlib.reload(documents_db)
from src.db.documents_db import DocumentsDB

class DocumentProcessor:
    """
    문서 처리 클래스

    주요 기능:
    - PDF 파일을 Markdown 형식으로 변환
    - 변환된 데이터를 데이터베이스에 저장
    - 파일 해시 계산

    사용 예:
        processor = DocumentProcessor()
        file_hash = processor.process_pdf("example.pdf")
    """

    def __init__(self, db_path: Optional[str] = None, config=None, progress_callback=None):
        """
        DocumentProcessor 초기화

        Args:
            db_path (Optional[str]): 데이터베이스 파일 경로 (기본값: config에서 로드)
            config: 설정 객체 (기본값: get_config() 호출)
            progress_callback: 진행 상황 콜백 함수
        """
        # 설정 객체 로드 (기본값: get_config 함수 호출)
        self.config = config or get_config()
        # 로깅 객체 초기화
        self.logger = get_logger('[DOCP]')
        
        # 데이터베이스 경로 설정 (기본값: config에서 로드)
        if db_path is None:
            db_path = self.config.DOCUMENTS_DB_PATH

        # 데이터베이스 객체 초기화
        self.docs_db = DocumentsDB(db_path)
        # 토크나이저 초기화 (OpenAI 모델 기반)
        self.tokenizer = tiktoken.encoding_for_model(self.config.OPENAI_TOKENIZER_MODEL)
        # 진행 상황 콜백 함수 설정
        self.progress_callback = progress_callback

        # 초기화 완료 메시지 로깅
        self.logger.info(f"DocumentProcessor 초기화 완료 (DB: {db_path})")

    def clean_markdown_text(self, text: str) -> str:
        """
        Markdown 텍스트 전처리

        Args:
            text: 원본 텍스트

        Returns:
            str: 전처리된 텍스트
        """
        import re

        # 공백 및 탭을 단일 공백으로 변환
        text = re.sub(r'[ \t]+', ' ', text)
        # 연속된 세 줄 이상의 개행을 두 줄로 축소
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 각 줄의 앞뒤 공백 제거
        lines = [line.strip() for line in text.split('\n')]
        # 줄 단위로 다시 합침
        text = '\n'.join(lines)

        return text.strip()

    def calculate_file_hash(self, file_path: str) -> str:
        """
        파일의 SHA-256 해시를 계산합니다.

        Args:
            file_path (str): 해시를 계산할 파일 경로

        Returns:
            str: 파일의 SHA-256 해시값
        """
        # 파일을 바이너리 모드로 열고, 내용을 읽어 SHA-256 해시 계산
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def markdown_with_progress(self, pdf_path: str) -> Tuple[List[Dict], int]:
        """
        PDF를 페이지별로 Markdown 변환 (진행 상황 표시)
        
        Args:
            pdf_path (str): PDF 파일 경로
        
        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
        """
        # 파일 이름 추출
        file_name = Path(pdf_path).name
        total_pages = 0
        
        # PDF 파일 열기
        with pymupdf.open(pdf_path) as doc:
            total_pages = len(doc)  # 총 페이지 수 계산

        pages_data = []
        # 진행 상황 표시를 위한 tqdm 초기화
        with tqdm(total=total_pages, desc="PDF to Markdown", unit="page") as pbar:
            for page_num in range(total_pages):
                try:
                    # 페이지를 Markdown 형식으로 변환
                    markdown = pymupdf4llm.to_markdown(
                        doc=pdf_path,
                        pages=[page_num]
                    )
                    
                    # 변환된 Markdown 텍스트 전처리
                    markdown = self.clean_markdown_text(markdown)
                    
                    # 빈 페이지 처리
                    if not markdown.strip():
                        status = 'empty'
                        markdown = self.config.EMPTY_PAGE_MARKER
                        pbar_msg = f"빈 페이지: {page_num + 1}"
                    else:
                        status = 'processing'
                        pbar_msg = f"페이지 {page_num + 1} len={len(markdown)}"
                    
                    # 페이지 데이터 저장
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': markdown
                    })
                    
                    # 진행 상황 콜백 호출
                    if self.progress_callback:
                        self.progress_callback({
                            'file_name': file_name,
                            'current_page': page_num + 1,
                            'total_pages': total_pages,
                            'page_content_length': len(markdown),
                            'status': status,
                            'error': ""
                        })

                except Exception as e:
                    # 예외 발생 시 경고 메시지 로깅 및 실패 처리
                    pbar_msg = f"페이지 {page_num + 1} 실패: {e}"
                    self.logger.warning(pbar_msg)
                    
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': self.config.ERROR_PAGE_MARKER
                    })
                    
                    if self.progress_callback:
                        self.progress_callback({
                            'file_name': file_name,
                            'current_page': page_num + 1,
                            'total_pages': total_pages,
                            'page_content_length': 0,
                            'status': 'failed',
                            'error': str(e)
                        })
                finally:
                    # 진행 상황 업데이트
                    pbar.set_postfix_str(pbar_msg)
                    pbar.update(1)
        
        return pages_data, total_pages

    def process_pdf(self, pdf_path: str, pdf_name: Optional[str] = None) -> Optional[str]:
        """
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            pdf_path (str): 처리할 PDF 파일 경로
            pdf_name (Optional[str]): 처리할 PDF 파일 이름 (기본값: None)
                데이터베이스에 저장할 때 사용됩니다, 고유 해야 합니다. 만약 동일한 이름이라면 경로를 추가하세요.

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        # PyMuPDF 설치 여부 확인
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF/pymupdf4llm 미설치: pip install pymupdf pymupdf4llm")
            return None

        # PDF 파일 경로 확인
        pdf_file = Path(pdf_path)
        if not pdf_file.exists() or not pdf_file.is_file():
            self.logger.error(f"파일 없음: {pdf_path}")
            return None
        
        # 파일 이름 설정
        if pdf_name is None:
            pdf_name = pdf_file.name

        self.logger.info(f"PDF 처리 시작: {pdf_file.name}")

        # 파일 크기 및 해시 계산
        file_size = pdf_file.stat().st_size
        file_hash = self.calculate_file_hash(pdf_path)
        self.logger.debug(f"파일 해시: {file_hash[:16]}...")

        # PDF를 Markdown으로 변환
        pages_data, total_pages = self.markdown_with_progress(pdf_path)
        
        # 변환된 페이지 데이터를 하나의 텍스트로 합침
        all_content = []
        for page_data in pages_data:
            page_num = page_data['page_num']
            content = page_data['content']
            
            is_empty = (content == self.config.EMPTY_PAGE_MARKER)
            
            if not is_empty:
                page_content = f"{self.config.PAGE_MARKER_FORMAT.format(page_num=page_num)}\n\n{content}"
            else:
                page_content = content
            
            all_content.append(page_content)
            
        text_content = '\n'.join(all_content)
        text_content = self.clean_markdown_text(text_content)
        
        # 데이터베이스에 저장
        self.docs_db.insert_text_content(
            file_name=pdf_name,
            file_hash=file_hash,
            total_pages=total_pages,
            file_size=file_size,
            text_content=text_content
        )

        self.logger.info(
            f"PDF 처리 완료: {pdf_file.name} "
            f"({len(pages_data)} 페이지, {file_size / 1024:.1f}KB), 변환문자수={len(text_content):,}"
        )
        return file_hash