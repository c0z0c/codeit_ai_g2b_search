---
layout: default
title: "RAG 시스템 인터페이스 문서 - Config 인터페이스 문서"
description: "RAG 시스템 인터페이스 문서 - Config 인터페이스 문서"
date: 2025-11-14
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# Config 인터페이스 문서

## 파일 정보
- **경로**: `src/config.py`
- **목적**: 중앙 집중식 설정 관리를 위한 Config 클래스 제공

## 클래스: Config

### 개요
애플리케이션의 모든 설정을 관리하는 싱글톤 패턴 기반 클래스입니다. `config/config.json` 파일에서 설정을 로드하며, 환경 변수를 통한 오버라이드를 지원합니다.

### 주요 설정 카테고리

#### 1. OpenAI API 설정
- `OPENAI_API_KEY`: OpenAI API 키 (환경 변수에서 자동 로드)
- `OPENAI_MODEL`: 답변 생성용 LLM 모델 (기본값: "gpt-5-mini")
- `OPENAI_TEMPERATURE`: 생성 온도 (0.0~2.0, 기본값: 0.0)
- `OPENAI_EMBEDDING_MODEL`: 임베딩 모델 (기본값: "text-embedding-3-small")
- `OPENAI_TOKENIZER_MODEL`: 토큰 카운팅용 모델명 (기본값: "gpt-4")

#### 2. 청킹(Chunking) 설정
- `CHUNKING_MODE`: 청킹 모드 ("token" 또는 "character", 기본값: "token")
- `CHUNK_SIZE`: 청크 크기 (토큰 단위, 기본값: 1500)
- `CHUNK_OVERLAP`: 청크 중첩 크기 (기본값: 300)
- `CHUNK_SEPARATORS`: 분할 구분자 우선순위 리스트

#### 3. 검색 설정
- `SIMILARITY_THRESHOLD`: 유사도 임계값 (0.0~1.0, 기본값: 0.75)
- `TOP_K_SUMMARY`: 1단계 요약 기반 검색 결과 수 (기본값: 5)
- `TOP_K_FINAL`: 2단계 최종 청크 검색 결과 수 (기본값: 2)
- `SCORE_GAP_THRESHOLD`: 상위 결과 간 점수 차이 임계값 (기본값: 0.15)

#### 4. 임베딩 설정
- `EMBEDDING_BATCH_SIZE`: 임베딩 API 배치 크기 (기본값: 100)
- `EMBEDDING_DIMENSION`: 임베딩 벡터 차원 (기본값: 1536)

#### 5. 경로 설정
- `DATA_PATH`: 데이터 루트 디렉토리 (기본값: "data")
- `DOCUMENTS_DB_PATH`: 문서 메타데이터 SQLite DB (기본값: "data/documents.db")
- `EMBEDDINGS_DB_PATH`: 임베딩 메타데이터 SQLite DB (기본값: "data/embeddings.db")
- `CHAT_HISTORY_DB_PATH`: 채팅 이력 SQLite DB (기본값: "data/chat_history.db")
- `VECTORSTORE_PATH`: FAISS 인덱스 저장 디렉토리 (기본값: "data/vectorstore")
- `CONFIG_PATH`: 설정 파일 경로 (기본값: "config/config.json")

#### 6. 문서 처리 설정
- `MARKER_DUMP_ENABLED`: 페이지 마커 덤프 활성화 여부 (기본값: True)
- `MARKER_DUMP_PATH`: 페이지 마커 덤프 디렉토리 (기본값: "data/markers")
- `EMPTY_PAGE_THRESHOLD`: 빈 페이지 판별 기준 (기본값: 10자)
- `ERROR_PAGE_MARKER`: 오류 페이지 마커 문자열
- `EMPTY_PAGE_MARKER`: 빈 페이지 마커 문자열
- `PAGE_MARKER_FORMAT`: 페이지 구분자 포맷

#### 7. 마크다운 전처리 설정
- `MARKDOWN_PROTECT_BLOCKS`: 보호할 블록 타입 리스트 (기본값: ['code', 'math', 'inline_math', 'mermaid'])
- `MARKDOWN_REMOVE_ELEMENTS`: 제거할 요소 리스트 (기본값: ['html', 'images', 'links', 'emphasis', 'headers', 'blockquotes', 'lists'])
- `MARKDOWN_MAX_LINES`: 블록 타입별 최대 라인 수 (기본값: {'code': 100, 'math': 50})

#### 8. 로깅 설정
- `LOG_LEVEL`: 로깅 레벨 (기본값: "DEBUG")
- `LOG_DIR`: 로그 파일 저장 디렉토리 (기본값: "logs")
- `LOG_FILE_NAME`: 통합 로그 파일명 (기본값: "rag_system.log")
- `LOG_FILE_MAX_BYTES`: 로그 파일 최대 크기 (기본값: 10MB)
- `LOG_FILE_BACKUP_COUNT`: 로그 파일 로테이션 백업 개수 (기본값: 5)

#### 9. 프롬프트 템플릿
- `RAG_PROMPT_TEMPLATE`: RAG 시스템 프롬프트 템플릿
- `NO_CONTEXT_MESSAGE`: 검색 실패 시 메시지
- `CONTEXT_FORMAT`: 컨텍스트 포맷 문자열

### 메서드

#### get_instance(config_path: Optional[str] = None) -> Config
싱글톤 인스턴스를 반환합니다.

**Parameters:**
- `config_path` (Optional[str]): config.json 파일 경로

**Returns:**
- `Config`: 싱글톤 인스턴스

**사용 예:**
```python
config = Config.get_instance()
print(config.OPENAI_MODEL)
```

#### load_from_json(config_path: Optional[str] = None) -> Config
config.json 파일에서 설정을 로드합니다.

**Parameters:**
- `config_path` (Optional[str]): config.json 파일 경로 (기본값: "config/config.json")

**Returns:**
- `Config`: Config 인스턴스

**특징:**
- JSON에만 존재하는 키는 동적 속성으로 추가
- 기존 config.json에 없는 신규 옵션은 dataclass 기본값 자동 적용

#### save_to_json(config_path: Optional[str] = None) -> bool
현재 설정을 config.json 파일로 저장합니다.

**Parameters:**
- `config_path` (Optional[str]): config.json 파일 경로

**Returns:**
- `bool`: 저장 성공 여부

**Raises:**
- `ValueError`: JSON 직렬화 불가능한 타입이 포함된 경우

**사용 예:**
```python
config = Config.get_instance()
config.CHUNK_SIZE = 800
config.save_to_json()
```

#### validate() -> bool
설정 값 검증을 수행합니다.

**Returns:**
- `bool`: 검증 성공 여부

**검증 항목:**
- 청킹 설정 (CHUNK_SIZE, CHUNK_OVERLAP)
- 온도 설정 (OPENAI_TEMPERATURE)
- 유사도 임계값 (SIMILARITY_THRESHOLD)
- Top-K 설정 (TOP_K_SUMMARY, TOP_K_FINAL)
- 임베딩 배치 크기 (EMBEDDING_BATCH_SIZE)
- 마크다운 전처리 옵션 유효성

#### get_db_path(db_type: str) -> str
데이터베이스 경로를 반환합니다.

**Parameters:**
- `db_type` (str): 'documents', 'embeddings', 'chat_history'

**Returns:**
- `str`: DB 파일 경로

#### get_vectorstore_path(embedding_hash: str) -> str
벡터 스토어 파일 경로를 반환합니다.

**Parameters:**
- `embedding_hash` (str): 임베딩 해시값

**Returns:**
- `str`: FAISS 인덱스 파일 경로

#### to_dict() -> Dict[str, Any]
설정을 딕셔너리로 변환합니다.

**Returns:**
- `Dict[str, Any]`: 설정 딕셔너리

### 전역 함수

#### get_config(config_path: Optional[str] = None) -> Config
전역 Config 인스턴스를 반환합니다.

**Parameters:**
- `config_path` (Optional[str]): config.json 파일 경로

**Returns:**
- `Config`: 싱글톤 인스턴스

**사용 예:**
```python
from src.config import get_config

config = get_config()
print(config.CHUNK_SIZE)
```

## 설정 파일 예시

```json
{
  "version": "1.0.0",
  "OPENAI_MODEL": "gpt-5-mini",
  "OPENAI_TEMPERATURE": 0.0,
  "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
  "CHUNK_SIZE": 1500,
  "CHUNK_OVERLAP": 300,
  "SIMILARITY_THRESHOLD": 0.75,
  "TOP_K_SUMMARY": 5,
  "TOP_K_FINAL": 2
}
```

## 주의사항

1. **싱글톤 패턴**: Config 클래스는 싱글톤 패턴을 사용하므로 항상 동일한 인스턴스를 반환합니다.
2. **환경 변수 우선**: `OPENAI_API_KEY`는 환경 변수가 JSON 파일보다 우선합니다.
3. **동적 속성**: JSON에만 존재하는 키는 동적 속성으로 추가되므로 런타임에 새로운 설정을 추가할 수 있습니다.
4. **검증 필수**: 설정 변경 후에는 반드시 `validate()` 메서드로 검증해야 합니다.
