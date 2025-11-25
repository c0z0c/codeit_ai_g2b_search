# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import hashlib
import importlib
import os
from pathlib import Path
import re
from venv import logger
import requests
import tempfile
import tiktoken
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple
import urllib.parse

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

    # HWP, PDF 공통 처리 로직 (중복 제거)
    def _process_common_workflow(
        self,
        doc_path: str,
        doc_name: Optional[str],
        converter_func: Callable[[str], Tuple[List[Dict], int]],
        doc_type_log: str
    ) -> Tuple[Optional[str], bool]:
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
            return None, False

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
            return file_hash, False

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
        return file_hash, True

    # 각 파일 포맷별 변환 로직 (형태 유지)
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
    def process_pdf(self, doc_path: str, doc_name: Optional[str] = None) -> Tuple[Optional[str], bool]:
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
            return None, False

        # 공통 워크플로우 실행 (PDF 변환 함수 전달)
        return self._process_common_workflow(
            doc_path=doc_path,
            doc_name=doc_name,
            converter_func=self.markdown_with_progress_pdf,
            doc_type_log="PDF"
        )

    def process_hwp(self, doc_path: str, doc_name: Optional[str] = None) -> Tuple[Optional[str], bool]:
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

    def process_doc(self, doc_path: str, doc_name: Optional[str] = None) -> Tuple[Optional[str], bool]:
        """
        외부 실행 함수
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
            return None, False

    def get_data_go_kr(
        self,
        api_url: str,
        timeout: int = 10,
        service_key: Optional[str] = None
    ) -> Dict:
        """
        공공데이터포털 API 공통 호출 함수 (Low-level).
        상위 함수로부터 전달받은 service_key를 우선 사용합니다.
        """
        # 1. 서비스 키 로드 우선순위: 인자값 > 환경변수
        if not service_key:
            service_key = os.getenv('DATA_GO_KR_SERVICE_KEY')

        if not service_key:
            self.logger.error("API 인증 키(service_key)가 누락되었습니다.")
            raise ValueError("API Service Key is missing provided via args or env var.")

        # 2. serviceKey 인코딩 (이미 인코딩된 키인지 확인이 필요할 수 있으나, 일반적으로 raw 키를 받아 인코딩함)
        try:
            # 키에 %, / 같은 특수문자가 있을 수 있으므로 safe 처리 주의
            encoded_service_key = urllib.parse.quote(service_key, safe='')
        except Exception as e:
            raise ValueError(f"Service Key 인코딩 실패: {e}")

        full_url = f"{api_url}&serviceKey={encoded_service_key}"
        self.logger.debug(f"API 요청 URL 생성 완료: {full_url[:100]}...")

        try:
            response = requests.get(full_url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            self.logger.error(f"API HTTP 요청 실패: {e}")
            raise

        # 3. 응답 구조 및 결과 코드 검증
        if 'response' not in data:
            raise ValueError("API 응답 구조 오류: 'response' 키 없음")

        header = data.get('response', {}).get('header', {})
        result_code = header.get('resultCode')
        result_msg = header.get('resultMsg')

        if result_code != '00':
            self.logger.error(f"API 논리 오류: Code={result_code}, Msg={result_msg}")
            if result_code == '03':
                raise ValueError("API 서비스 키 인증 실패 (Code 03). 올바른 키인지 확인하세요.")
            raise ValueError(f"API 오류 응답: [{result_code}] {result_msg}")

        return data

    def get_bid_file_info(
        self,
        start_date: str,
        end_date: str,
        service_key: str,  # 필수 인자로 변경
        bid_ntce_no: Optional[str] = '',
        page_no: int = 1,
        inqry_div: Optional[int] = 1,
        num_of_rows_per_page: int = 100
    ) -> List[Dict]:
        """
        날짜 범위 및 페이지네이션을 처리하며 입찰 공고 정보를 조회합니다.
        전달받은 service_key를 get_data_go_kr로 넘겨줍니다.

        Args:
            start_date: 조회 시작 날짜 (YYYYMMDD or YYYYMMDDHHMM)
            end_date: 조회 종료 날짜 (YYYYMMDD or YYYYMMDDHHMM)
            service_key: 공공데이터포털 API 서비스 키 (필수)
            bid_ntce_no: 입찰 공고 번호 (기본값: 빈 문자열)
            page_no: 조회할 페이지 번호 (기본값: 1)
            inqry_div: 조회 구분 (기본값: 1)
            num_of_rows_per_page: 페이지당 조회 건수 (기본값: 100)

        Returns:
            List[Dict]: 조회된 입찰 공고 정보 항목 리스트

        Raises:
            ValueError: 날짜 형식 오류.
        """
        # 날짜 포맷팅 및 검증
        date_pattern = re.compile(r'^\d{8}(\d{4})?$')
        if not date_pattern.match(start_date) or not date_pattern.match(end_date):
            raise ValueError("날짜 형식 오류: YYYYMMDD 또는 YYYYMMDDHHMM 형식이어야 합니다.")

        # YYYYMMDD 형식일 때 시간 추가
        start_date += '0000' if len(start_date) == 8 else start_date
        end_date += '2359' if len(end_date) == 8 else end_date

        # 날짜 변환
        start_dt = datetime.strptime(start_date, "%Y%m%d%H%M")
        end_dt = datetime.strptime(end_date, "%Y%m%d%H%M")

        # 결과 병합
        merged_results = []
        current_start = start_dt

        # 날짜 루프 (월 단위 분할)
        while current_start <= end_dt:
            current_end = (current_start + timedelta(days=31)).replace(day=1) - timedelta(seconds=1)
            if current_end > end_dt:
                current_end = end_dt

            # 날짜를 문자열로 변환
            current_start_str = current_start.strftime("%Y%m%d%H%M")
            current_end_str = current_end.strftime("%Y%m%d%H%M")

            self.logger.debug(f"API 호출: {current_start_str} ~ {current_end_str}")

            api_url = (
                f"https://apis.data.go.kr/1230000/ad/BidPublicInfoService/"
                f"getBidPblancListPPIFnlRfpIssAtchFileInfo"
                f"?pageNo={page_no}"
                f"&numOfRows={num_of_rows_per_page}"
                f"&inqryDiv={inqry_div}"
                f"&type=json"
                f"&bidNtceNo={bid_ntce_no}"
                f"&inqryBgnDt={current_start_str}"
                f"&inqryEndDt={current_end_str}"
            )

            # API 호출
            try:
                data = self.get_data_go_kr(
                                api_url=api_url,
                                service_key=service_key,
                                timeout=10
                )
                items = data.get('response', {}).get('body', {}).get('items', {})
                if items:
                    merged_results.extend(items)
                    logger.info(f"{current_start_str} ~ {current_end_str}: {len(items)}건 조회")
                else:
                    logger.info(f"{current_start_str} ~ {current_end_str}: 조회된 문서 없음")
            except Exception as e:
                logger.error(f"API 호출 실패: {e}")
                raise

            current_start = current_end + timedelta(minutes=1)

        return merged_results

    def extract_file_url(self, items: List[Dict]) -> List[str]:
        """
        API 응답 항목 리스트에서 첨부 파일 URL 추출.

        Args:
            items: API 응답 항목 딕셔너리 리스트.

        Returns:
            List[str]: 유효한 첨부 파일 URL 리스트.
        """
        file_urls = []
        for item in items:
            file_url = item.get('atchFileUrl')
            if file_url and isinstance(file_url, str) and file_url.strip():
                file_urls.append(file_url.strip())

        self.logger.info(f"전체 {len(items)}건 중 유효한 파일 URL {len(file_urls)}개 추출.")
        return list(set(file_urls))  # 중복 제거

    def download_file(self, file_urls: List[str], save_dir: Optional[str] = None) -> List[Path]:
        """
        외부 실행 함수
        파일들을 다운로드하여 지정된 디렉토리에 저장.

        Args:
            file_urls: 다운로드할 파일들의 URL 리스트.
            save_dir: 파일을 저장할 디렉토리 경로.
                      지정하지 않으면 **tempfile.mkdtemp()**를 사용하여 임시 디렉토리에 저장.
        Returns:
            List[Path]: 저장된 파일들의 전체 경로 리스트.

        Raises:
            requests.HTTPError: 다운로드 실패 시.
        """
        if not file_urls:
            self.logger.info("다운로드할 파일 URL이 없습니다.")
            return []

        # 디렉토리 설정 및 생성
        if not save_dir:
            save_dir = Path(tempfile.mkdtemp())
            temp_dir_created = True
            self.logger.info(f"임시 디렉토리 생성: {save_dir.resolve()}")

        save_dir_path = Path(save_dir)

        if not temp_dir_created:
            save_dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"파일 저장 디렉토리: {save_dir.resolve()}")

        saved_file_paths: List[Path] = []

        for i, file_url in enumerate(file_urls):
            self.logger.info(f"[{i+1}/{len(file_urls)}] 다운로드 시작: {file_url}")

            try:
                # 1. 파일 다운로드 요청
                # 일부 API는 헤더만 가져오기 위해 stream=True로 설정하고,
                # 실제 파일 다운로드는 별도로 진행할 수 있지만, 여기서는 바로 다운로드 스트림을 엽니다.
                response = requests.get(file_url, stream=True, timeout=30)  # 다운로드 타임아웃 30초
                response.raise_for_status()  # HTTP 오류가 발생하면 예외 발생

                # 2. 파일 이름 추출
                file_name_raw = None
                if 'Content-Disposition' in response.headers:
                    # 'filename=' 또는 'filename*=' 패턴으로 파일 이름을 추출
                    fname = re.findall(r'filename=["\']?([^"\']*)["\']?', response.headers['Content-Disposition'])
                    if fname:
                        file_name_raw = fname[0]

                # Content-Disposition 헤더가 없거나 추출 실패 시 URL 경로의 기본 이름 사용
                if not file_name_raw:
                    # '?' 이후의 쿼리 문자열 제거
                    file_name_raw = os.path.basename(urllib.parse.urlparse(file_url).path)

                # URL 디코딩 및 파일 이름 정리
                # 파일 이름은 먼저 URL 디코딩을 시도하고, 세미콜론과 같은 유효하지 않은 문자를 정리합니다.
                file_name = urllib.parse.unquote(file_name_raw).replace(';', '').strip()
                if not file_name:
                    # 파일 이름을 얻지 못하면 임시 이름 부여
                    file_name = f"downloaded_file_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}.bin"
                    self.logger.warning(f"파일 이름 추출 실패. 임시 이름 사용: {file_name}")

                # 3. 최종 저장 경로 설정
                final_file_path = save_dir / file_name
                file_path_str = str(final_file_path)

                # 4. Long Path Prefix 적용 (Windows)
                final_path_for_open = file_path_str
                # Windows 환경이고, 경로 길이가 260자를 초과할 경우
                if len(file_path_str) > 260 and os.name == 'nt':
                    if not file_path_str.startswith('\\\\?\\'):
                        # Long Path Prefix (\?\) 추가: 절대 경로로 변환하여 적용
                        final_path_for_open = '\\\\?\\' + os.path.abspath(file_path_str)
                        self.logger.debug(f"Long Path Prefix 적용: {final_path_for_open}")

                # 5. 파일 저장
                with open(final_path_for_open, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                self.logger.info(f"다운로드 완료: {final_file_path.name}")
                saved_file_paths.append(final_file_path)

            except requests.HTTPError as http_err:
                self.logger.error(f"HTTP 오류 발생 (URL: {file_url}): {http_err}")
                # 다음 파일 다운로드를 위해 현재 파일을 건너뛰고 계속 진행합니다.
            except Exception as e:
                self.logger.error(f"다운로드 중 알 수 없는 오류 발생 (URL: {file_url}): {e}")

        self.logger.info(f"총 {len(file_urls)}개 파일 중 {len(saved_file_paths)}개 파일 저장 완료.")
        return saved_file_paths

    def cleanup_files(self, saved_file_paths: List[Path]):
        """
        다운로드된 임시 파일들을 삭제합니다.

        Args:
            saved_file_paths: 삭제할 파일의 Path 객체 리스트
        """
        if not saved_file_paths:
            return

        self.logger.info(f"임시 파일 {len(saved_file_paths)}개 삭제 시작...")

        for path in saved_file_paths:
            try:
                if path.exists() and path.is_file():
                    path.unlink()  # 임시 파일 삭제
                    self.logger.debug(f"임시 파일 삭제 성공: {path.name}")
                else:
                    self.logger.warning(f"삭제할 파일이 없거나 파일이 아님: {path.name}")
            except Exception as e:
                # 임시 파일 삭제 실패가 전체 프로세스를 중단시키지 않도록 로그만 남김
                self.logger.error(f"임시 파일 삭제 실패 ({path.name}): {e}")

        self.logger.info("임시 파일 정리 완료.")

    def process_date(self, data_key: str, start_date: str, end_date: str) -> bool:
        """
        [메인 실행 함수]
        1. data_key(인증키) 검증
        2. 날짜별 공고 조회 (API 호출)
        3. 첨부파일 URL 추출 및 다운로드
        4. 파일 확장자별 마크다운 및 DB 저장
        5. DB 저장 후 임시 파일 정리

        Args:
            data_key: 공공데이터포털 디코딩된 Service Key (필수 권한 요소)
            start_date: 검색 시작일 (YYYYMMDD)
            end_date: 검색 종료일 (YYYYMMDD)
        """
        # 1. 권한 요소(data_key) 검증
        if not data_key or not isinstance(data_key, str):
            self.logger.error("유효하지 않은 data_key이므로 검색 권한이 없습니다.")
            return False

        self.logger.info(f"프로세스 시작: {start_date} ~ {end_date}")

        try:
            # 2. 입찰 공고 조회 (Service Key 전달)
            # data_key를 service_key 파라미터로 명시적으로 전달하여 API 권한을 획득합니다.
            bid_items = self.get_bid_file_info(
                start_date=start_date,
                end_date=end_date,
                service_key=data_key
            )

            if not bid_items:
                self.logger.info("해당 기간에 조회된 입찰 공고가 없습니다.")
                return True

            # 3. URL 추출
            file_urls = self.extract_file_url(bid_items)
            self.logger.info(f"총 {len(file_urls)}개의 첨부파일 URL 발견.")

            # 4. 파일 다운로드
            if file_urls:
                downloaded_files = self.download_file(file_urls)
                # 5. 다운로드 파일 후처리
                for fpath in downloaded_files:
                    ext = fpath.suffix.lower()
                    if ext == '.pdf':
                        self.process_pdf(str(fpath))
                    elif ext == '.hwp':
                        self.process_hwp(str(fpath))
                    else:
                        self.logger.error(f"현재 처리할 수 없는 파일 형식: {ext}")
            self.cleanup_files(downloaded_files)
            self.logger.info("해당 기간의 문서를 처리하여 DB에 저장 완료 및 임시 파일 제거.")
            return True

        except ValueError as ve:
            self.logger.error(f"검증 오류 발생: {ve}")
            return False
        except Exception as e:
            self.logger.critical(f"시스템 치명적 오류 발생: {e}", exc_info=True)
            return False
