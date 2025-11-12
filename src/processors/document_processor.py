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

from src.db import DocumentsDB
from src.config import get_config
from src.utils.logging_config import get_logger
from tqdm.notebook import tqdm

class DocumentProcessor:
    """
    문서 처리 클래스

    주요 기능:
    - PDF 파일을 Markdown 형식으로 변환
    - 변환된 데이터를 데이터베이스에 저장
    - 파일 해시 계산 및 토큰 수 계산

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
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        
        if db_path is None:
            db_path = self.config.DOCUMENTS_DB

        self.docs_db = DocumentsDB(db_path)
        self.tokenizer = tiktoken.encoding_for_model(self.config.OPENAI_TOKENIZER_MODEL)
        self.progress_callback = progress_callback

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

        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = [line.strip() for line in text.split('\n')]
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
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def count_tokens(self, text: str) -> int:
        """
        주어진 텍스트의 토큰 수를 계산합니다.

        Args:
            text (str): 토큰 수를 계산할 텍스트

        Returns:
            int: 텍스트의 토큰 수
        """
        return len(self.tokenizer.encode(text))

    def markdown_with_progress(self, pdf_path: str) -> List[Dict]:
        """
        PDF를 페이지별로 Markdown 변환 (진행 상황 표시)
        
        Args:
            pdf_path (str): PDF 파일 경로
        
        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
        """
        file_name = Path(pdf_path).name
        
        with pymupdf.open(pdf_path) as doc:
            total_pages = len(doc)

        pages_data = []
        with tqdm(total=total_pages, desc="PDF to Markdown", unit="page") as pbar:
            for page_num in range(total_pages):
                try:
                    markdown = pymupdf4llm.to_markdown(
                        doc=pdf_path,
                        pages=[page_num]
                    )
                    
                    markdown = self.clean_markdown_text(markdown)
                    
                    if not markdown.strip():
                        status = 'empty'
                        markdown = self.config.EMPTY_PAGE_MARKER
                        pbar_msg = f"빈 페이지: {page_num + 1}"
                    else:
                        status = 'processing'
                        pbar_msg = f"페이지 {page_num + 1} len={len(markdown)}"
                    
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': markdown
                    })
                    
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
                    pbar_msg = f"페이지 {page_num + 1} 실패: {e}"
                    self.logger.warning(pbar_msg)
                    
                    pages_data.append({
                        'page_num': page_num + 1,
                        'content': "[변환 실패]"
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
                    pbar.set_postfix_str(pbar_msg)
                    pbar.update(1)
        
        return pages_data

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            pdf_path (str): 처리할 PDF 파일 경로

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF/pymupdf4llm 미설치: pip install pymupdf pymupdf4llm")
            return None

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            self.logger.error(f"파일 없음: {pdf_path}")
            return None

        self.logger.info(f"PDF 처리 시작: {pdf_file.name}")

        file_hash = self.calculate_file_hash(pdf_path)
        self.logger.debug(f"파일 해시: {file_hash[:16]}...")

        pages_data = self.markdown_with_progress(pdf_path)
        
        all_content = []
        for page_data in pages_data:
            page_num = page_data['page_num']
            content = page_data['content']
            
            is_empty = content == self.config.EMPTY_PAGE_MARKER
            
            if not is_empty:
                page_content = f"{self.config.PAGE_MARKER_FORMAT.format(page_num=page_num)}\n\n{content}"
            else:
                page_content = content
            
            all_content.append(page_content)
            
            token_count = self.count_tokens(page_content)
            self.docs_db.insert_page_data(
                file_hash=file_hash,
                page_number=page_num,
                markdown_content=page_content,
                token_count=token_count,
                is_empty=is_empty
            )

        full_content = "\n\n".join(all_content)
        full_content = self.clean_markdown_text(full_content)
        total_tokens = self.count_tokens(full_content)

        self.docs_db.insert_file_info(
            file_hash=file_hash,
            file_name=pdf_file.name,
            total_pages=len(pages_data),
            file_size=pdf_file.stat().st_size,
            total_chars=len(full_content),
            total_tokens=total_tokens
        )

        self.logger.info(
            f"PDF 처리 완료: {pdf_file.name} "
            f"({len(pages_data)} 페이지, {total_tokens:,} 토큰, {pdf_file.stat().st_size / 1024:.1f}KB)"
        )
        return file_hash