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

class DocumentProcessor:
    \"\"\"문서 처리 클래스 - PDF를 Markdown으로 변환하고 DB에 저장\"\"\"

    def __init__(self, db_path: str = 'data/documents.db'):
        self.docs_db = DocumentsDB(db_path)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def calculate_file_hash(self, file_path: Path) -> str:
        \"\"\"파일 SHA-256 해시 계산\"\"\"
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def count_tokens(self, text: str) -> int:
        \"\"\"텍스트 토큰 수 계산\"\"\"
        return len(self.tokenizer.encode(text))

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        \"\"\"
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            파일 해시값 또는 None
        \"\"\"
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF가 설치되지 않았습니다. pip install pymupdf")
            return None

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"파일을 찾을 수 없습니다: {pdf_path}")
            return None

        # 파일 해시 계산
        file_hash = self.calculate_file_hash(pdf_file)

        # PDF 열기 및 텍스트 추출
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        all_content = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            # 빈 페이지 처리
            is_empty = len(text.strip()) < 10
            if is_empty:
                page_content = "--- [빈페이지] ---"
            else:
                page_content = f"--- 페이지 {page_num + 1} ---\n\n{text}"

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
        print(f"PDF 처리 완료: {pdf_file.name} ({total_pages} 페이지, {total_tokens} 토큰)")
        return file_hash
