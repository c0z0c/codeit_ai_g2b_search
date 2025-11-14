---
layout: default
title: "RAG 시스템 인터페이스 문서 - Database 인터페이스 문서"
description: "RAG 시스템 인터페이스 문서 - Database 인터페이스 문서"
date: 2025-11-14
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# Database 인터페이스 문서

## 개요
데이터베이스 관련 클래스들은 SQLite를 사용하여 문서 메타데이터 및 채팅 이력을 관리합니다.

---

## 1. ChatHistoryDB

### 파일 정보
- **경로**: `src/db/chat_history_db.py`
- **목적**: 채팅 세션 및 메시지 관리

### 클래스: ChatHistoryDB

#### 생성자
```python
ChatHistoryDB(db_path: str = 'data/chat_history.db')
```

**Parameters:**
- `db_path` (str): 데이터베이스 파일 경로 (기본값: 'data/chat_history.db')

### 메서드

#### create_session(session_name: Optional[str] = None) -> str
새로운 채팅 세션을 생성합니다.

**Parameters:**
- `session_name` (Optional[str]): 세션 이름 (기본값: 현재 시간 기반 자동 생성)

**Returns:**
- `str`: 생성된 세션 ID (UUID)

**사용 예:**
```python
db = ChatHistoryDB()
session_id = db.create_session("사용자 상담 세션")
```

#### add_message(session_id: str, role: str, content: str, retrieved_chunks: Optional[List[Dict[str, Any]]] = None) -> int
특정 세션에 메시지를 추가합니다.

**Parameters:**
- `session_id` (str): 메시지를 추가할 세션 ID
- `role` (str): 메시지의 역할 ('user' 또는 'assistant')
- `content` (str): 메시지 내용
- `retrieved_chunks` (Optional[List[Dict[str, Any]]]): 검색된 청크 데이터 (JSON 형식)

**Returns:**
- `int`: 추가된 메시지의 ID

**사용 예:**
```python
message_id = db.add_message(
    session_id=session_id,
    role='user',
    content='공고 요건이 무엇인가요?',
    retrieved_chunks=[{'file_name': 'doc.pdf', 'page': 1}]
)
```

#### get_chat_stats() -> Dict[str, int]
채팅 데이터베이스의 통계 정보를 반환합니다.

**Returns:**
- `Dict[str, int]`: 통계 정보
  - `total_sessions`: 총 세션 수
  - `active_sessions`: 활성 세션 수
  - `total_messages`: 총 메시지 수
  - `user_messages`: 사용자 메시지 수
  - `assistant_messages`: 어시스턴트 메시지 수

**사용 예:**
```python
stats = db.get_chat_stats()
print(f"총 메시지 수: {stats['total_messages']}")
```

### 데이터베이스 테이블

#### chat_sessions

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| session_id | TEXT PRIMARY KEY | 세션 고유 ID (UUID) |
| session_name | TEXT NOT NULL | 세션 이름 |
| created_at | TIMESTAMP | 생성 시각 |
| updated_at | TIMESTAMP | 수정 시각 |
| is_active | BOOLEAN | 활성 상태 (1: 활성, 0: 비활성) |

#### chat_messages

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| message_id | INTEGER PRIMARY KEY | 메시지 고유 ID (자동 증가) |
| session_id | TEXT NOT NULL | 세션 ID (외래 키) |
| role | TEXT NOT NULL | 역할 ('user' 또는 'assistant') |
| content | TEXT NOT NULL | 메시지 내용 |
| retrieved_chunks | TEXT | 검색된 청크 JSON |
| timestamp | TIMESTAMP | 메시지 시각 |

---

## 2. DocumentsDB

### 파일 정보
- **경로**: `src/db/documents_db.py`
- **목적**: 마크다운 문서 메타데이터 관리

### 클래스: DocumentsDB

#### 생성자
```python
DocumentsDB(db_path: str = 'data/documents.db')
```

**Parameters:**
- `db_path` (str): 데이터베이스 파일 경로 (기본값: 'data/documents.db')

### 메서드

#### insert_text_content(file_name: str, file_hash: str, total_pages: Optional[int] = 0, file_size: Optional[int] = 0, text_content: Optional[str] = "") -> bool
마크다운 파일 정보를 데이터베이스에 저장합니다.

**Parameters:**
- `file_name` (str): 파일 이름 (필수, 고유)
- `file_hash` (str): 파일의 해시값
- `total_pages` (Optional[int]): 총 페이지 수 (기본값: 0)
- `file_size` (Optional[int]): 파일 크기 (바이트, 기본값: 0)
- `text_content` (Optional[str]): 파일의 전체 텍스트 콘텐츠 (기본값: "")

**Returns:**
- `bool`: 저장 성공 여부

**Raises:**
- `ValueError`: file_name이 None 또는 빈 값인 경우

**사용 예:**
```python
db = DocumentsDB()
success = db.insert_text_content(
    file_name="공고문.pdf",
    file_hash="abc123...",
    total_pages=10,
    file_size=1024000,
    text_content="# 페이지 1\n내용..."
)
```

#### get_document_by_hash(file_hash: str) -> Optional[Dict[str, Any]]
파일 해시값으로 문서 정보를 조회합니다.

**Parameters:**
- `file_hash` (str): 조회할 파일의 해시값

**Returns:**
- `Optional[Dict[str, Any]]`: 문서 정보 딕셔너리 (없으면 None 반환)

**사용 예:**
```python
doc = db.get_document_by_hash("abc123...")
if doc:
    print(doc['file_name'])
```

#### get_documents_all() -> List[Dict[str, Any]]
데이터베이스의 모든 문서를 조회합니다.

**Returns:**
- `List[Dict[str, Any]]`: 문서 리스트 (file_name, file_hash, text_content 등)

**사용 예:**
```python
all_docs = db.get_documents_all()
for doc in all_docs:
    print(f"{doc['file_name']}: {doc['total_pages']}페이지")
```

#### get_document_stats() -> Dict[str, Any]
데이터베이스에 저장된 문서 통계 정보를 반환합니다.

**Returns:**
- `Dict[str, Any]`: 문서 통계 정보
  - `total_files`: 총 파일 수
  - `total_pages`: 총 페이지 수
  - `total_size_bytes`: 총 파일 크기 (바이트 단위)
  - `total_size_mb`: 총 파일 크기 (MB 단위)

**사용 예:**
```python
stats = db.get_document_stats()
print(f"총 {stats['total_files']}개 파일, {stats['total_size_mb']}MB")
```

#### search_documents(search_term: str, search_type: str = 'auto') -> List[Dict[str, Any]]
파일명 또는 file_hash로 문서를 검색합니다.

**Parameters:**
- `search_term` (str): 검색어 (파일명 또는 file_hash)
- `search_type` (str): 검색 타입 (기본값: 'auto')
  - `'auto'`: file_hash 형식(64자 hex)이면 hash 검색, 아니면 filename 검색
  - `'filename'`: file_name LIKE 검색
  - `'hash'`: file_hash 완전 일치 검색

**Returns:**
- `List[Dict[str, Any]]`: 검색 결과 리스트

**사용 예:**
```python
# file_hash로 검색
results = db.search_documents("abc123def456...")

# 파일명으로 검색
results = db.search_documents("공고문", search_type='filename')

# 자동 판별
results = db.search_documents("test.pdf", search_type='auto')
```

#### execute_query(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]
SELECT 쿼리를 실행하고 결과를 반환합니다.

**Parameters:**
- `query` (str): 실행할 SQL 쿼리
- `params` (Optional[tuple]): 쿼리 파라미터 (기본값: None)

**Returns:**
- `List[Dict[str, Any]]`: 쿼리 결과 (각 행이 딕셔너리)

**사용 예:**
```python
results = db.execute_query(
    "SELECT * FROM TB_DOCUMENTS WHERE file_hash = ?",
    ("abc123",)
)
for row in results:
    print(row['file_name'])
```

#### summary() -> None
데이터베이스의 모든 테이블 요약 정보를 출력합니다.

**사용 예:**
```python
db = DocumentsDB()
db.summary()
```

### 데이터베이스 테이블

#### TB_DOCUMENTS

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| file_hash | TEXT PRIMARY KEY | 파일 고유 해시값 (중복 불가) |
| file_name | TEXT NOT NULL | 파일 이름 (변환 전 이름, 고유) |
| total_pages | INTEGER NOT NULL | 원본 파일 총 페이지 수 |
| file_size | INTEGER NOT NULL | 원본 파일 크기 (바이트 단위) |
| text_content | TEXT | 변환된 파일의 텍스트 콘텐츠 |
| created_at | TIMESTAMP | 생성 시각 (기본값: 현재 시각 +9시간) |
| updated_at | TIMESTAMP | 수정 시각 (기본값: 현재 시각 +9시간) |

## 주의사항

1. **트랜잭션 관리**: 모든 DB 작업은 context manager를 통해 자동으로 커밋됩니다.
2. **고유 제약**: DocumentsDB의 file_name과 file_hash는 고유해야 합니다.
3. **마이그레이션**: 데이터베이스 버전은 PRAGMA user_version으로 관리되며, 버전 변경 시 _migrate_database() 메서드를 구현해야 합니다.
4. **시간대**: DocumentsDB는 KST(+9시간) 기준으로 시간을 저장합니다.
