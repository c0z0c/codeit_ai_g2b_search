# -*- coding: utf-8 -*-
import hashlib
import importlib
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple
import tiktoken
from tqdm import tqdm

# PDF 처리 라이브러리
try:
    import pymupdf
    import pymupdf4llm
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# HWP 처리 라이브러리
try:
    from helper_hwp import hwp_to_markdown
    HELPER_HWP_AVAILABLE = True
except ImportError:
    HELPER_HWP_AVAILABLE = False

from src.config import get_config
from src.utils.logging_config import get_logger
from src.db import documents_db
importlib.reload(documents_db)
from src.db.documents_db import DocumentsDB


class DocumentProcessor:
    """
    문서 처리 클래스

    주요 기능:
    - HWP, PDF 파일을 Markdown 형식으로 변환
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

        # 페이지 마커 덤프 디렉토리 생성
        if self.config.MARKER_DUMP_ENABLED:
            self.marker_dump_path = Path(self.config.MARKER_DUMP_PATH)
            self.marker_dump_path.mkdir(parents=True, exist_ok=True)
        else:
            self.marker_dump_path = None

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
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def split_text_into_chunks(self, text: str, n: int = 5) -> List[str]:
        """
        텍스트를 n개의 조각으로 분할합니다.

        Args:
            text (str): 처리할 텍스트
            n (int): 텍스트를 나누는 조각 수 (기본값: 5)

        Returns:
            chunks (List[str]): 조각으로 분할된 텍스트
        """
        length = len(text)
        # 몫과 나머지 계산
        k, m = divmod(length, n)

        chunks = []
        start = 0
        for i in range(n):
            # 나머지(m)만큼 앞쪽 청크들에 1씩 더 배분하여 균등하게 나눔
            chunk_size = k + 1 if i < m else k
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end

        return chunks

    def _create_missing_dump_files(self) -> None:
        """
        DB에 있는 문서 중 덤프 파일이 없는 문서의 마커 덤프 파일 생성
        """
        if not self.config.MARKER_DUMP_ENABLED:
            self.logger.debug("마커 덤프 비활성화됨")
            return

        all_docs = self.docs_db.get_documents_all()

        if not all_docs:
            self.logger.debug("DB에 문서 없음")
            return

        created_count = 0
        for doc in all_docs:
            file_name = doc.get('file_name')
            text_content = doc.get('text_content')

            if not file_name or not text_content:
                continue

            if self._save_marker_dump_file(file_name, text_content):
                created_count += 1

        if created_count > 0:
            self.logger.info(f"기존 문서 덤프 파일 생성: {created_count}개")

    def _save_marker_dump_file(self, file_name: str, text_content: str) -> bool:
        """
        단일 문서의 마커 덤프 파일 저장

        Args:
            file_name: 문서 파일명
            text_content: 마크다운 텍스트 내용

        Returns:
            bool: 파일 생성 여부 (이미 존재하면 False)
        """
        if not self.config.MARKER_DUMP_ENABLED or not self.marker_dump_path:
            return False

        # 확장자 처리: .md면 그대로, 아니면 .md로 변경
        file_path = Path(file_name)
        dump_file_name = file_path.name if file_path.suffix.lower() == '.md' else file_path.stem + '.md'

        dump_file_path = self.marker_dump_path / dump_file_name

        if dump_file_path.exists():
            return False

        dump_file_path.write_text(text_content, encoding='utf-8')
        return True

    # [New] 공통 처리 로직 (중복 제거)
    def _process_common_workflow(
        self,
        doc_path: str,
        doc_name: Optional[str],
        converter_func: Callable[[str], Tuple[List[Dict], int]],
        doc_type_log: str
    ) -> Optional[str]:
        """
        문서 처리의 공통 워크플로우를 수행합니다. (PDF/HWP 공통)

        1. 파일 존재 확인 및 이름 설정
        2. 해시 계산 및 DB 중복 확인
        3. 문서 변환 (converter_func 실행)
        4. 텍스트 병합 및 전처리
        5. 청킹 및 DB 저장
        6. 덤프 파일 생성

        Args:
            doc_path: 파일 경로
            doc_name: 파일 이름 (Optional)
            converter_func: 실제 변환을 수행하는 함수 (doc_path -> (pages_data, total_pages))
            doc_type_log: 로그용 문서 타입 문자열 (예: "PDF", "HWP")
        """
        # 파일 경로 확인
        doc_file = Path(doc_path)
        if not doc_file.exists():
            self.logger.error(f"파일을 찾을 수 없습니다: {doc_path}")
            return None

        # 파일 이름 설정
        if doc_name is None:
            doc_name = doc_file.name

        self.logger.info(f"{doc_type_log} 처리 시작: {doc_file.name}")

        # 파일 크기 및 해시 계산
        file_size = doc_file.stat().st_size
        file_hash = self.calculate_file_hash(doc_path)

        # 중복 검사 (이미 처리된 파일인지)
        existing_docs = self.docs_db.search_documents(file_hash, search_type='hash')
        if existing_docs:
            self.logger.info(f"이미 처리된 파일 (skip): {doc_name}")
            return file_hash

        # [변환 실행] 해당 파일 타입에 맞는 converter 함수를 호출하여 데이터를 받아옴
        pages_data, total_pages = converter_func(doc_path)

        # 변환된 전체 텍스트 합치기 (Page Marker 포함)
        all_content = []
        for page_data in pages_data:
            content = page_data['content']
            if content != self.config.EMPTY_PAGE_MARKER:
                page_content = f"{self.config.PAGE_MARKER_FORMAT.format(page_num=page_data['page_num'])}\n\n{content}"
                all_content.append(page_content)
            else:
                all_content.append(content)

        text_content = '\n'.join(all_content)
        text_content = self.clean_markdown_text(text_content)

        # 텍스트를 5개로 분할하여 저장
        chunks = self.split_text_into_chunks(text_content, n=5)
        self.logger.info(f"텍스트 분할 저장 (총 5개 조각) - {doc_name}")

        # 데이터베이스에 저장
        for idx, chunk in enumerate(chunks):
            self.docs_db.insert_text_content(
                file_name=doc_name,
                file_hash=file_hash,
                total_pages=total_pages,
                file_size=file_size,
                text_content=chunk,
                chunk_index=idx  # chunk_index 전달 (0, 1, 2, 3, 4)
            )

        # 덤프 파일 저장
        if self._save_marker_dump_file(doc_name, text_content):
            self.logger.info("마커 덤프 생성 완료")

        self.logger.info(f"{doc_type_log} 처리 및 저장 완료: {doc_name}")
        return file_hash

    # [Converter Methods] - 각 파일 포맷별 변환 로직 (형태 유지)
    def markdown_with_progress_pdf(self, doc_path: str) -> Tuple[List[Dict], int]:
        """
        PDF를 페이지별로 Markdown 변환 (진행 상황 표시)

        Args:
            doc_path (str): PDF 파일 경로

        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
            int: 총 페이지 수
        """
        # 파일 이름 추출
        doc_name = Path(doc_path).name
        total_pages = 0

        # PDF 파일 열기
        with pymupdf.open(doc_path) as doc:
            total_pages = len(doc)  # 총 페이지 수 계산

        pages_data = []
        # 진행 상황 표시를 위한 tqdm 초기화
        with tqdm(total=total_pages, desc="PDF to Markdown", unit="page") as pbar:
            for page_num in range(total_pages):
                try:
                    # 페이지를 Markdown 형식으로 변환
                    markdown = pymupdf4llm.to_markdown(
                        doc=doc_path,
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
                            'file_name': doc_name,
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
                            'file_name': doc_name,
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

    def markdown_with_progress_hwp(self, doc_path: str) -> Tuple[List[Dict], int]:
        """
        HWP를 Markdown 변환 (진행 상황 표시)
        markdown을 읽으면서 80줄마다 페이지 데이터 저장

        Args:
            doc_path (str): HWP 파일 경로

        Returns:
            List[Dict]: [{'page_num': int, 'content': str}, ...]
            int: 총 페이지 수
        """
        # 파일 이름 추출
        doc_name = Path(doc_path).name
        pages_data = []

        # 한 페이지당 줄 수 설정
        LINES_PER_PAGE = 80
        total_pages = 0

        try:
            # 1. HWP를 전체 Markdown 텍스트로 변환
            full_markdown = hwp_to_markdown(doc_path)
            full_markdown = self.clean_markdown_text(full_markdown)

            # 2. 줄 단위로 분리
            lines = full_markdown.splitlines()
            total_lines = len(lines)

            # 3. 총 페이지 수 계산
            if total_lines == 0:
                total_pages = 1  # 빈 파일이라도 1페이지로 처리
            else:
                total_pages = (total_lines + LINES_PER_PAGE - 1) // LINES_PER_PAGE

            # 4. 80줄 단위로 순회하며 처리
            with tqdm(total=total_pages, desc="HWP to Markdown 80줄마다 페이지 분할", unit="page") as pbar:
                for i in range(0, total_lines, LINES_PER_PAGE):
                    page_num = (i // LINES_PER_PAGE) + 1

                    try:
                        chunk_lines = lines[i: i + LINES_PER_PAGE]
                        page_content = "\n".join(chunk_lines)

                        # 빈 페이지 처리 로직
                        if not page_content.strip():
                            status = 'empty'
                            page_content = self.config.EMPTY_PAGE_MARKER
                            pbar_msg = f"빈 페이지: {page_num}"
                        else:
                            status = 'processing'
                            pbar_msg = f"페이지 {page_num} len={len(page_content)}"

                        # 페이지 데이터 저장
                        pages_data.append({
                            'page_num': page_num,
                            'content': page_content
                        })

                        # 진행 상황 콜백 호출
                        if self.progress_callback:
                            self.progress_callback({
                                'file_name': doc_name,
                                'current_page': page_num,
                                'total_pages': total_pages,
                                'page_content_length': len(page_content),
                                'status': status,
                                'error': ""
                            })

                    except Exception as e:
                        pbar_msg = f"페이지 {page_num} 처리 실패: {e}"
                        self.logger.warning(pbar_msg)

                        pages_data.append({
                            'page_num': page_num,
                            'content': self.config.ERROR_PAGE_MARKER
                        })

                        if self.progress_callback:
                            self.progress_callback({
                                'file_name': doc_name,
                                'current_page': page_num,
                                'total_pages': total_pages,
                                'page_content_length': 0,
                                'status': 'failed',
                                'error': str(e)
                            })
                    finally:
                        pbar.set_postfix_str(pbar_msg)
                        pbar.update(1)

        except Exception as e:
            self.logger.warning(f"HWP 파일 변환 실패: {e}")
            pages_data.append({
                'page_num': 1,
                'content': self.config.ERROR_PAGE_MARKER
            })
            if self.progress_callback:
                self.progress_callback({
                    'file_name': doc_name,
                    'current_page': 1,
                    'total_pages': 1,
                    'page_content_length': 0,
                    'status': 'failed',
                    'error': str(e)
                })
            return pages_data, 1

        return pages_data, total_pages

    # [공통 인터페이스] - process_doc, process_pdf, process_hwp
    def process_pdf(self, doc_path: str, doc_name: Optional[str] = None) -> Optional[str]:
        """
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            doc_path (str): 처리할 PDF 파일 경로
            doc_name (Optional[str]): 처리할 PDF 파일 이름 (기본값: None)
                데이터베이스에 저장할 때 사용됩니다, 고유 해야 합니다. 만약 동일한 이름이라면 경로를 추가하세요.

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        # PyMuPDF 설치 여부 확인
        if not PYMUPDF_AVAILABLE:
            self.logger.error("PyMuPDF 미설치로 PDF 처리 불가")
            return None

        # 공통 워크플로우 실행 (PDF 변환 함수 전달)
        return self._process_common_workflow(
            doc_path=doc_path,
            doc_name=doc_name,
            converter_func=self.markdown_with_progress_pdf,
            doc_type_log="PDF"
        )

    def process_hwp(self, doc_path: str, doc_name: Optional[str] = None) -> Optional[str]:
        """
        HWP 파일을 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            doc_path (str): 처리할 HWP 파일 경로
            doc_name (Optional[str]): 처리할 HWP 파일 이름 (기본값: None)
                데이터베이스에 저장할 때 사용됩니다, 고유 해야 합니다. 만약 동일한 이름이라면 경로를 추가하세요.

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        # helper_hwp 설치 여부 확인
        if not HELPER_HWP_AVAILABLE:
            self.logger.error("helper_hwp 미설치로 HWP 처리 불가")
            return None

        # 공통 워크플로우 실행 (HWP 변환 함수 전달)
        return self._process_common_workflow(
            doc_path=doc_path,
            doc_name=doc_name,
            converter_func=self.markdown_with_progress_hwp,
            doc_type_log="HWP"
        )

    def process_doc(self, doc_path: str, doc_name: Optional[str] = None) -> Optional[str]:
        """
        HWP, PDF 파일을 분기 처리하여 Markdown으로 변환하고 DB에 저장합니다.

        Args:
            doc_path (str): 처리할 문서 파일 경로
            doc_name (Optional[str]): 처리할 문서 파일 이름 (기본값: None)
                데이터베이스에 저장할 때 사용됩니다, 고유 해야 합니다. 만약 동일한 이름이라면 경로를 추가하세요.

        Returns:
            Optional[str]: 처리된 파일의 해시값 (실패 시 None 반환)
        """
        file_ext = Path(doc_path).suffix.lower()

        if file_ext == '.hwp':
            return self.process_hwp(doc_path, doc_name)
        elif file_ext == '.pdf':
            return self.process_pdf(doc_path, doc_name)
        else:
            self.logger.error(f"지원하지 않는 파일 형식: {file_ext}")
            return None
