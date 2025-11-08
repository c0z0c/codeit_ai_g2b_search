# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import tiktoken

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from src.db import DocumentsDB
from src.config import get_config
from src.utils.logging_config import get_logger

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

    def __init__(self, db_path: Optional[str] = None, config=None):
        """
        DocumentProcessor 초기화

        Args:
            db_path (Optional[str]): 데이터베이스 파일 경로 (기본값: config에서 로드)
            config: 설정 객체 (기본값: get_config() 호출)
        """
        # Config 로드
        self.config = config or get_config()

        # 로거 초기화
        self.logger = get_logger(__name__)

        # DB 경로 설정 (파라미터 우선, 없으면 Config 사용)
        if db_path is None:
            db_path = self.config.DOCUMENTS_DB

        self.docs_db = DocumentsDB(db_path)
        self.tokenizer = tiktoken.encoding_for_model(self.config.OPENAI_TOKENIZER_MODEL)

        self.logger.info(f"DocumentProcessor 초기화 완료 (DB: {db_path})")

    def calculate_file_hash(self, file_path: Path) -> str:
        """
        파일의 SHA-256 해시를 계산합니다.

        Args:
            file_path (Path): 해시를 계산할 파일 경로

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

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        """
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            pdf_path (str): 처리할 PDF 파일 경로

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF가 설치되지 않았습니다. pip install pymupdf")
            return None

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            self.logger.error(f"파일을 찾을 수 없습니다: {pdf_path}")
            return None

        self.logger.info(f"PDF 처리 시작: {pdf_file.name}")

        # 파일 해시 계산
        file_hash = self.calculate_file_hash(pdf_file)
        self.logger.debug(f"파일 해시 계산 완료: {file_hash[:16]}...")

        # PDF 열기 및 텍스트 추출
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        all_content = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            # 빈 페이지 처리
            is_empty = len(text.strip()) < self.config.EMPTY_PAGE_THRESHOLD
            if is_empty:
                page_content = self.config.EMPTY_PAGE_MARKER
            else:
                page_content = f"{self.config.PAGE_MARKER_FORMAT.format(page_num=page_num + 1)}\n\n{text}"

            all_content.append(page_content)

            # 페이지 데이터 저장
            token_count = self.count_tokens(page_content)
            self.docs_db.insert_page_data(
                file_hash=file_hash,
                page_number=page_num + 1,
                markdown_content=page_content,
                token_count=token_count,
                is_empty=is_empty
            )

        # 전체 콘텐츠 결합
        full_content = "\n\n".join(all_content)
        total_tokens = self.count_tokens(full_content)

        # 파일 정보 저장
        self.docs_db.insert_file_info(
            file_hash=file_hash,
            file_name=pdf_file.name,
            total_pages=total_pages,
            file_size=pdf_file.stat().st_size,
            total_chars=len(full_content),
            total_tokens=total_tokens
        )

        doc.close()
        self.logger.info(
            f"PDF 처리 완료: {pdf_file.name} "
            f"({total_pages} 페이지, {total_tokens:,} 토큰, {pdf_file.stat().st_size / 1024:.1f}KB)"
        )
        return file_hash