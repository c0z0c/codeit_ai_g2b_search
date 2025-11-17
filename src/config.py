# -*- coding: utf-8 -*-
"""
Configuration Management Module

중앙 집중식 설정 관리를 위한 Config 클래스
config/config.json 파일을 통해 설정 값을 관리하며,
환경 변수를 통한 오버라이드 지원
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

@dataclass
class Config:
    """
    애플리케이션 중앙 집중식 설정 관리 클래스 (Singleton)

    config/config.json 파일에서 설정을 로드하며, 환경 변수를 통한 오버라이드를 지원합니다.
    모든 RAG 파이프라인 관련 설정(OpenAI API, 청킹, 임베딩, 검색, 로깅 등)을 관리합니다.

    사용 예:
        >>> config = Config.get_instance()
        >>> print(config.OPENAI_MODEL)
        'gpt-4o-mini'
        >>> config.CHUNK_SIZE = 800
        >>> config.save_to_json()

    Attributes:
        version (str): 설정 파일 버전
        OPENAI_API_KEY (Optional[str]): OpenAI API 키 (환경 변수 우선)
        OPENAI_MODEL (str): 사용할 LLM 모델명
        OPENAI_TEMPERATURE (float): 생성 온도 (0.0~2.0, 낮을수록 결정적)
        OPENAI_EMBEDDING_MODEL (str): 임베딩 모델명
        OPENAI_TOKENIZER_MODEL (str): 토큰 카운팅용 모델명
        
        CHUNK_SIZE (int): 텍스트 청크 크기 (토큰 단위)
        CHUNK_OVERLAP (int): 청크 간 중첩 크기 (토큰 단위)
        CHUNK_SEPARATORS (List[str]): 청크 분할 구분자 우선순위
        
        SUMMARY_RATIO (float): 요약 비율 (0.0~1.0)
        SUMMARY_OVERLAP_RATIO (float): 요약 청크 중첩 비율
        SUMMARY_MIN_LENGTH (int): 최소 요약 길이 (토큰)
        
        SIMILARITY_THRESHOLD (float): 유사도 검색 임계값 (0.0~1.0)
        TOP_K_SUMMARY (int): 요약 단계에서 반환할 최대 청크 수
        TOP_K_FINAL (int): 최종 검색 결과 수
        SCORE_GAP_THRESHOLD (float): 상위 결과 간 점수 차이 임계값
        
        EMBEDDING_BATCH_SIZE (int): 임베딩 배치 처리 크기
        EMBEDDING_DIMENSION (int): 임베딩 벡터 차원 (text-embedding-3-small: 1536)
        
        DATA_PATH (str): 데이터 디렉토리 경로
        DOCUMENTS_DB_PATH (str): 문서 메타데이터 DB 경로
        EMBEDDINGS_DB_PATH (str): 임베딩 메타데이터 DB 경로
        CHAT_HISTORY_DB_PATH (str): 채팅 이력 DB 경로
        VECTORSTORE_PATH (str): FAISS 벡터 스토어 디렉토리
        CONFIG_PATH (str): 설정 파일 경로
        
        EMPTY_PAGE_THRESHOLD (int): 빈 페이지 판별 임계값 (글자 수)
        EMPTY_PAGE_MARKER (str): 빈 페이지 마커 문자열
        PAGE_MARKER_FORMAT (str): 페이지 번호 마커 포맷
        TOKEN_ESTIMATION_DIVISOR (int): 토큰 수 추정 나눗수 (문자수/4)
        
        MARKDOWN_PROTECT_BLOCKS (List[str]): 보호할 블록 타입 리스트 ['code', 'math', 'inline_math', 'mermaid']
        MARKDOWN_REMOVE_ELEMENTS (List[str]): 제거할 요소 리스트 ['html', 'images', 'links', 'emphasis', 'headers']
        MARKDOWN_MAX_LINES (Dict[str, int]): 블록 타입별 최대 라인 수 {'code': 100, 'math': 50}
        
        HASH_ALGORITHM (str): 해시 알고리즘 (sha256, md5 등)
        
        LOG_LEVEL (str): 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_DIR (str): 로그 파일 디렉토리
        LOG_FILE_MAX_BYTES (int): 로그 파일 최대 크기 (바이트)
        LOG_FILE_BACKUP_COUNT (int): 로그 파일 백업 개수
        
        RAG_PROMPT_TEMPLATE (str): RAG 프롬프트 템플릿
        NO_CONTEXT_MESSAGE (str): 검색 결과 없을 때 메시지
        CONTEXT_FORMAT (str): 컨텍스트 포맷 문자열
    """

    # ==================== 버전 정보 ====================
    version: str = "1.0.0"  # 설정 스키마 버전

    # ==================== OpenAI API 설정 ====================
    OPENAI_API_KEY: Optional[str] = None  # API 키 (환경 변수에서 자동 로드)
    OPENAI_MODEL: str = "gpt-5-mini"  # 답변 생성용 LLM 모델
    OPENAI_TEMPERATURE: float = 0.0  # 생성 온도 (0.0=결정적, 2.0=창의적)
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # 임베딩 모델
    OPENAI_TOKENIZER_MODEL: str = "gpt-4"  # 토큰 카운팅용 모델명

    # ==================== 청킹(Chunking) 설정 ====================
    CHUNKING_MODE: str = "token"  # 청킹 모드 ('token' 또는 'character')
    CHUNK_SIZE: int = 1500  # 청크 크기 (토큰 단위, 권장: 500-1000)
    CHUNK_OVERLAP: int = 300  # 청크 중첩 크기 (문맥 유지용)
    CHUNK_SEPARATORS: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""
    ])  # 분할 구분자 우선순위 (단락 > 문장 > 단어)

    # ==================== 요약 설정 ====================
    SUMMARY_RATIO: float = 0.2  # 요약 비율 (원문의 20%)
    SUMMARY_OVERLAP_RATIO: float = 0.1  # 요약 청크 중첩 비율
    SUMMARY_MIN_LENGTH: int = 100  # 최소 요약 길이 (토큰)

    # ==================== 검색 설정 ====================
    SIMILARITY_THRESHOLD: float = 0.75  # 유사도 임계값 (0.0~1.0, 높을수록 엄격)
    TOP_K_SUMMARY: int = 5  # 1단계: 요약 기반 검색 결과 수
    TOP_K_FINAL: int = 2  # 2단계: 최종 청크 검색 결과 수
    SCORE_GAP_THRESHOLD: float = 0.15  # 상위 결과 간 점수 차이 임계값

    # ==================== 임베딩 설정 ====================
    EMBEDDING_BATCH_SIZE: int = 100  # 임베딩 API 배치 크기 (비용/속도 최적화)
    EMBEDDING_DIMENSION: int = 1536  # 임베딩 벡터 차원 (모델 종속)

    # ==================== 경로 설정 ====================
    DATA_PATH: str = "data"  # 데이터 루트 디렉토리
    DOCUMENTS_DB_PATH: str = "data/documents.db"  # 문서 메타데이터 SQLite DB
    EMBEDDINGS_DB_PATH: str = "data/embeddings.db"  # 임베딩 메타데이터 SQLite DB
    CHAT_HISTORY_DB_PATH: str = "data/chat_history.db"  # 채팅 이력 SQLite DB
    VECTORSTORE_PATH: str = "data/vectorstore"  # FAISS 인덱스 저장 디렉토리
    CONFIG_PATH: str = "config/config.json"  # 설정 파일 경로

    # ==================== 문서 처리 설정 ====================
    MARKER_DUMP_ENABLED: bool = True  # 페이지 마커 덤프 활성화 여부
    MARKER_DUMP_PATH: str = "data/markers"  # 페이지 마커 덤프 디렉토리
    EMPTY_PAGE_THRESHOLD: int = 10  # 빈 페이지 판별 기준 (10자 이하)
    ERROR_PAGE_MARKER: str = "--- [오류페이지] ---"  # 오류 페이지 마커 문자열 (변환실패)
    EMPTY_PAGE_MARKER: str = "--- [빈페이지] ---"  # 빈 페이지 마커 문자열
    PAGE_MARKER_FORMAT: str = "--- 페이지 {page_num} ---"  # 페이지 구분자 포맷
    TOKEN_ESTIMATION_DIVISOR: int = 4  # 토큰 수 추정용 나눗수 (문자수/4 ≈ 토큰수)

    # ==================== 마크다운 전처리 설정 ====================
    MARKDOWN_PROTECT_BLOCKS: List[str] = field(default_factory=lambda: [
        'code', 'math', 'inline_math', 'mermaid'
    ])  # 보호할 블록 타입 리스트 (빈 리스트 = 보호 비활성화)
    
    MARKDOWN_REMOVE_ELEMENTS: List[str] = field(default_factory=lambda: [
        'html', 'images', 'links', 'emphasis', 'headers', 'blockquotes', 'lists'
    ])  # 제거할 요소 리스트 (빈 리스트 = 제거 비활성화)
        
    MARKDOWN_MAX_LINES: Dict[str, int] = field(default_factory=lambda: {
        'code': 100,
        'math': 50
    })  # 블록 타입별 최대 라인 수

    # ==================== 해시 설정 ====================
    HASH_ALGORITHM: str = "sha256"  # 문서/임베딩 해시 알고리즘

    # ==================== 로깅 설정 ====================
    LOG_LEVEL: str = "DEBUG"  # 로깅 레벨 (DEBUG < INFO < WARNING < ERROR < CRITICAL)
    LOG_DIR: str = "logs"  # 로그 파일 저장 디렉토리
    LOG_FILE_NAME: str = "rag_system.log"  # 통합 로그 파일명
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  # 로그 파일 최대 크기 (10MB)
    LOG_FILE_BACKUP_COUNT: int = 5  # 로그 파일 로테이션 백업 개수

    # ==================== 프롬프트 템플릿 ====================
    RAG_PROMPT_TEMPLATE: str = """다음 문서를 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {question}

답변:"""  # RAG 시스템 프롬프트 템플릿 ({context}, {query} 플레이스홀더)

    NO_CONTEXT_MESSAGE: str = "관련 문서를 찾을 수 없습니다."  # 검색 실패 시 메시지
    CONTEXT_FORMAT: str = "[문서 {index}: {file_name}]\n{chunk_text}"  # 컨텍스트 포맷

    # 싱글톤 인스턴스 (클래스 변수)
    _instance: Optional['Config'] = None

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'Config':
        """
        싱글톤 인스턴스 반환

        Args:
            config_path: config.json 파일 경로 (선택사항)

        Returns:
            Config 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls.load_from_json(config_path)
        return cls._instance

    @classmethod
    def load_from_json(cls, config_path: Optional[str] = None) -> 'Config':
        """
        config.json 파일에서 설정 로드
        
        JSON에만 존재하는 키는 동적 속성으로 추가
        기존 config.json에 없는 신규 옵션은 dataclass 기본값 자동 적용

        Args:
            config_path: config.json 파일 경로

        Returns:
            Config 인스턴스
        """
        if config_path is None:
            config_path = "config/config.json"

        config_file = Path(config_path)

        # dataclass 기본값 딕셔너리 생성
        default_instance = cls()
        default_dict = asdict(default_instance)
        
        # JSON 파일이 존재하면 로드
        config_data = {}
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"설정 파일 로드 완료: {config_path}")
        
        # 신규 옵션 자동 병합 (기본값 + JSON)
        merged_data = {**default_dict, **config_data}
        
        # 신규 옵션 로그 출력 (dataclass에 추가된 필드)
        new_keys = set(default_dict.keys()) - set(config_data.keys())
        if new_keys and config_file.exists():
            print(f"신규 옵션 자동 적용: {', '.join(sorted(new_keys))}")
        
        # JSON에만 존재하는 키 (동적 속성으로 추가될 항목)
        json_only_keys = set(config_data.keys()) - set(default_dict.keys())
        if json_only_keys:
            print(f"JSON 전용 옵션 동적 추가: {', '.join(sorted(json_only_keys))}")

        # 환경 변수에서 API 키 로드 (최우선)
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            merged_data['OPENAI_API_KEY'] = api_key

        # dataclass 필드만으로 인스턴스 생성
        dataclass_fields = {k: v for k, v in merged_data.items() if k in default_dict}
        instance = cls(**dataclass_fields)
        
        # JSON 전용 키를 동적 속성으로 추가
        for key in json_only_keys:
            setattr(instance, key, config_data[key])

        # 검증
        instance.validate()

        return instance

    def save_to_json(self, config_path: Optional[str] = None) -> bool:
        """
        현재 설정을 config.json 파일로 저장
        
        dataclass 필드 + 동적 속성 모두 저장
        JSON 직렬화 불가능한 타입이 있으면 ValueError 발생

        Args:
            config_path: config.json 파일 경로

        Returns:
            저장 성공 여부
            
        Raises:
            ValueError: JSON 직렬화 불가능한 타입이 포함된 경우
        """
        if config_path is None:
            config_path = self.CONFIG_PATH

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # dataclass 필드 변환
        config_dict = asdict(self)
        
        # 동적 속성 추가 (dataclass 필드가 아닌 것들)
        dataclass_field_names = {f.name for f in self.__dataclass_fields__.values()}
        for key in dir(self):
            if not key.startswith('_') and key not in dataclass_field_names:
                attr = getattr(self, key)
                if not callable(attr):
                    config_dict[key] = attr
        
        # 제외할 키 제거
        config_dict.pop('_instance', None)
        config_dict.pop('OPENAI_API_KEY', None)  # 보안: API 키 제외

        # JSON 직렬화 가능 여부 검증
        for key, value in config_dict.items():
            if value is None:
                continue
            
            if not isinstance(value, (str, int, float, bool, list, dict)):
                raise ValueError(
                    f"Config.{key}의 타입 '{type(value).__name__}'은 JSON 직렬화 불가능합니다. "
                    f"허용 타입: str, int, float, bool, list, dict"
                )

        # 파일 저장
        with open(config_file, "w", encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"설정 파일 저장 완료: {config_path} (필드 수: {len(config_dict)})")
        return True

    def validate(self) -> bool:
        """
        설정 값 검증

        Returns:
            검증 성공 여부
        """
        errors = []

        # 청킹 설정 검증
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE는 양수여야 합니다.")

        if self.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP은 0 이상이어야 합니다.")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")

        # 온도 설정 검증
        if not 0.0 <= self.OPENAI_TEMPERATURE <= 2.0:
            errors.append("OPENAI_TEMPERATURE는 0.0~2.0 사이여야 합니다.")

        # 유사도 임계값 검증
        if not 0.0 <= self.SIMILARITY_THRESHOLD <= 1.0:
            errors.append("SIMILARITY_THRESHOLD는 0.0~1.0 사이여야 합니다.")

        # Top-K 검증
        if self.TOP_K_SUMMARY <= 0:
            errors.append("TOP_K_SUMMARY는 양수여야 합니다.")

        if self.TOP_K_FINAL <= 0:
            errors.append("TOP_K_FINAL은 양수여야 합니다.")

        # 임베딩 배치 크기 검증
        if self.EMBEDDING_BATCH_SIZE <= 0:
            errors.append("EMBEDDING_BATCH_SIZE는 양수여야 합니다.")
        
        # 마크다운 전처리 옵션 검증
        valid_protect_blocks = {'code', 'math', 'inline_math', 'mermaid'}
        invalid_protect = set(self.MARKDOWN_PROTECT_BLOCKS) - valid_protect_blocks
        if invalid_protect:
            errors.append(f"MARKDOWN_PROTECT_BLOCKS에 잘못된 값: {invalid_protect}. 허용: {valid_protect_blocks}")
        
        valid_remove_elements = {'html', 'images', 'links', 'emphasis', 'headers', 'blockquotes', 'lists'}
        invalid_remove = set(self.MARKDOWN_REMOVE_ELEMENTS) - valid_remove_elements
        if invalid_remove:
            errors.append(f"MARKDOWN_REMOVE_ELEMENTS에 잘못된 값: {invalid_remove}. 허용: {valid_remove_elements}")
        
        valid_max_lines_keys = {'code', 'math'}
        invalid_max_lines = set(self.MARKDOWN_MAX_LINES.keys()) - valid_max_lines_keys
        if invalid_max_lines:
            errors.append(f"MARKDOWN_MAX_LINES에 잘못된 키: {invalid_max_lines}. 허용: {valid_max_lines_keys}")

        if errors:
            print("설정 검증 실패:")
            for error in errors:
                print(f"  - {error}")
            return False

        print("설정 검증 통과")
        return True

    def get_db_path(self, db_type: str) -> str:
        """
        데이터베이스 경로 반환

        Args:
            db_type: 'documents', 'embeddings', 'chat_history'

        Returns:
            DB 파일 경로
        """
        db_paths = {
            'documents': self.DOCUMENTS_DB_PATH,
            'embeddings': self.EMBEDDINGS_DB_PATH,
            'chat_history': self.CHAT_HISTORY_DB_PATH
        }
        return db_paths.get(db_type, self.DOCUMENTS_DB_PATH)

    def get_vectorstore_path(self, embedding_hash: str) -> str:
        """
        벡터 스토어 파일 경로 반환

        Args:
            embedding_hash: 임베딩 해시값

        Returns:
            FAISS 인덱스 파일 경로
        """
        return f"{self.VECTORSTORE_PATH}/{embedding_hash}.faiss"
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        config_dict = asdict(self)
        config_dict.pop('_instance', None)
        return config_dict

    def __repr__(self) -> str:
        """문자열 표현"""
        return f"Config(version={self.version}, model={self.OPENAI_MODEL})"


# 전역 설정 인스턴스 (싱글톤)
def get_config(config_path: Optional[str] = None) -> Config:
    """
    전역 Config 인스턴스 반환

    Args:
        config_path: config.json 파일 경로 (선택사항)

    Returns:
        Config 싱글톤 인스턴스
    """
    return Config.get_instance(config_path)

