# VectorStore 인터페이스 문서

## 파일 정보
- **경로**: `src/vectorstore/vector_store_manager.py`
- **목적**: FAISS 벡터 인덱스 관리

## 클래스: VectorStoreManager

### 개요
FAISS 벡터 인덱스의 생성, 로드, 저장, 검색을 관리하는 클래스입니다. LangChain FAISS 통합을 통해 일관된 인터페이스를 제공합니다.

### 생성자
```python
VectorStoreManager(config=None)
```

**Parameters:**
- `config`: 설정 객체 (기본값: get_config() 호출)

**의존성:**
- LangChain
- FAISS
- OpenAI

**Attributes:**
- `vector_path` (Path): FAISS 인덱스 파일 경로
- `embedding_model` (str): 임베딩 모델명
- `embeddings` (OpenAIEmbeddings): LangChain OpenAI 임베딩 객체
- `vectorstore` (Optional[FAISS]): LangChain FAISS 벡터스토어
- `chunk_map` (Dict): (file_hash, chunk_index) → (faiss_idx, chunk_hash, embedding_config_hash) 매핑

---

## 주요 메서드

### 인덱스 관리

#### load() -> bool
기존 FAISS 인덱스를 로드합니다. 파일이 없으면 더미 인덱스를 자동 생성합니다.

**Returns:**
- `bool`: 로드 성공 여부

**사용 예:**
```python
vm = VectorStoreManager()
if vm.load():
    print("인덱스 로드 완료")
```

#### save() -> bool
FAISS 인덱스를 로컬 파일로 저장합니다.

**Returns:**
- `bool`: 저장 성공 여부

**사용 예:**
```python
vm.save()
```

#### create_from_documents(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool
문서 리스트로부터 새로운 FAISS 인덱스를 생성합니다.

**Parameters:**
- `texts` (List[str]): 텍스트 리스트
- `metadatas` (Optional[List[Dict[str, Any]]]): 메타데이터 리스트

**Returns:**
- `bool`: 생성 성공 여부

**사용 예:**
```python
texts = ["문서 1", "문서 2"]
metadatas = [{'file_hash': 'abc'}, {'file_hash': 'def'}]
success = vm.create_from_documents(texts, metadatas)
```

---

### 벡터 추가/삭제

#### add_texts(texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, int]
기존 FAISS 인덱스에 텍스트를 추가합니다. chunk_map을 활용하여 중복 체크 및 chunk_hash 비교를 수행합니다.

**Parameters:**
- `texts` (List[str]): 추가할 텍스트 리스트
- `metadatas` (Optional[List[Dict[str, Any]]]): 메타데이터 리스트

**Returns:**
- `Tuple[bool, int]`: (성공 여부, 추가 전 벡터 인덱스 시작 위치)

**Raises:**
- `ValueError`: chunk_index 누락 시

**중복 처리 전략:**
- 동일 (file_hash, chunk_index) 발견 시:
  - chunk_hash 동일: 무시
  - chunk_hash 다름: 기존 벡터 삭제 후 추가
  - config_hash 다름: 기존 벡터 삭제 후 추가

**사용 예:**
```python
texts = ["새 청크 1", "새 청크 2"]
metadatas = [
    {'file_hash': 'abc', 'chunk_index': 0, 'chunk_hash': 'hash1'},
    {'file_hash': 'abc', 'chunk_index': 1, 'chunk_hash': 'hash2'}
]
success, start_idx = vm.add_texts(texts, metadatas)
```

#### remove_by_file_hash(file_hash: str) -> bool
특정 파일 해시에 해당하는 모든 벡터를 삭제합니다.

**Parameters:**
- `file_hash` (str): 삭제할 파일의 해시값

**Returns:**
- `bool`: 삭제 성공 여부

**사용 예:**
```python
success = vm.remove_by_file_hash("abc123...")
```

#### remove_by_metadata(filter_metadata: Dict[str, Any]) -> bool
메타데이터 필터 조건에 맞는 벡터를 삭제합니다.

**Parameters:**
- `filter_metadata` (Dict[str, Any]): 삭제 조건 (예: {'file_hash': 'abc123'})

**Returns:**
- `bool`: 삭제 성공 여부

**사용 예:**
```python
# 특정 파일 해시의 모든 벡터 삭제
success = vm.remove_by_metadata({'file_hash': 'abc123'})

# 특정 청크 타입의 모든 벡터 삭제
success = vm.remove_by_metadata({'chunk_type': 'dummy'})
```

---

### 검색

#### search(query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]
유사도 검색을 수행합니다.

**Parameters:**
- `query` (str): 검색 쿼리
- `top_k` (int): 반환할 상위 결과 개수 (기본값: 5)
- `filter_metadata` (Optional[Dict[str, Any]]): 메타데이터 필터

**Returns:**
- `List[Tuple[Document, float]]`: (문서, 유사도 점수) 리스트

**사용 예:**
```python
results = vm.search("입찰 요건", top_k=10)
for doc, score in results:
    print(f"Score: {score}, Text: {doc.page_content[:100]}")
```

#### get_by_metadata(file_hash: Optional[str] = None, start_page: Optional[int] = None, end_page: Optional[int] = None, chunk_index_start: Optional[int] = None, chunk_index_end: Optional[int] = None) -> List[Tuple[Document, float]]
메타데이터 기반으로 벡터를 직접 조회합니다.

**Parameters:**
- `file_hash` (Optional[str]): 파일 해시
- `start_page` (Optional[int]): 시작 페이지
- `end_page` (Optional[int]): 종료 페이지
- `chunk_index_start` (Optional[int]): 청크 인덱스 시작
- `chunk_index_end` (Optional[int]): 청크 인덱스 종료

**Returns:**
- `List[Tuple[Document, float]]`: (문서, 0.0) 리스트 (score는 항상 0.0)

**사용 예:**
```python
# 특정 파일의 모든 청크 조회
results = vm.get_by_metadata(file_hash="abc123")

# 특정 파일의 페이지 범위 조회
results = vm.get_by_metadata(file_hash="abc123", start_page=1, end_page=5)

# 특정 파일의 청크 인덱스 범위 조회
results = vm.get_by_metadata(file_hash="abc123", chunk_index_start=0, chunk_index_end=10)
```

---

### 유틸리티

#### calculate_embedding_config_hash(file_hash: str) -> str
파일 + 청킹 설정 기반 embedding_config_hash를 계산합니다.

**Parameters:**
- `file_hash` (str): 원본 파일 해시

**Returns:**
- `str`: SHA-256 해시 (64자 hex)

**변경 감지 항목:**
- 파일 내용 변경 (file_hash)
- 청킹 설정 (CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS, CHUNKING_MODE)
- 전처리 설정 (MARKDOWN_PROTECT_BLOCKS, MARKDOWN_REMOVE_ELEMENTS, MARKDOWN_MAX_LINES)
- 임베딩 모델

**사용 예:**
```python
vm = VectorStoreManager()
hash1 = vm.calculate_embedding_config_hash("abc123")
# config 변경 후
hash2 = vm.calculate_embedding_config_hash("abc123")
if hash1 != hash2:
    print("재임베딩 필요")
```

#### get_vector_count() -> int
현재 저장된 벡터 수를 반환합니다.

**Returns:**
- `int`: 벡터 개수

**사용 예:**
```python
count = vm.get_vector_count()
print(f"총 {count}개 벡터")
```

#### summary() -> None
벡터스토어의 현재 상태를 테이블 형식으로 출력합니다.

**출력 정보:**
- 기본 정보: 벡터 수, 임베딩 모델, 차원, 파일 경로, 청크 개수, 토탈 사이즈
- 메타데이터 통계: file_hash별 벡터 수 분포
- 인덱스 정보: 차원, 인덱스 타입

**사용 예:**
```python
vm.summary()
```

**출력 예시:**
```
================================================================================
VectorStore Summary
================================================================================
벡터 수 (Vector Count)       : 1,234
청크 개수 (Chunk Count)       : 1,234
차원 (Dimension)              : 1536
인덱스 타입 (Index Type)      : IndexFlatL2
임베딩 모델 (Embedding Model) : text-embedding-3-small
파일 경로 (File Path)         : data/vectorstore/vectorstore.faiss
토탈 사이즈 (Total Size)      : 15.23 MB
--------------------------------------------------------------------------------
파일별 벡터 분포 (총 5개 파일)
--------------------------------------------------------------------------------
File Name                                Hash         Count
--------------------------------------------------------------------------------
공고문_2024.pdf                          abc12345        500
제안요청서.pdf                           def67890        350
...
================================================================================
```

---

### 내부 메서드 (Private)

#### _create_dummy_index() -> bool
더미 문서로 빈 FAISS 인덱스를 생성합니다.

#### _build_chunk_map() -> None
docstore를 순회하여 chunk_map을 구축합니다.

#### _remove_by_indices(indices: List[int]) -> None
특정 인덱스의 벡터를 삭제합니다. (재구성 방식)

#### _rebuild_index(keep_indices: List[int], keep_docs: List[Document]) -> None
필터링된 벡터로 FAISS 인덱스를 재구성합니다.

---

## chunk_map 구조

`chunk_map`은 청크 추적을 위한 핵심 자료구조입니다:

```python
chunk_map: Dict[Tuple[str, int], Tuple[int, str, str]]
```

**Key:** `(file_hash, chunk_index)`
**Value:** `(faiss_idx, chunk_hash, embedding_config_hash)`

- `file_hash` (str): 파일 해시
- `chunk_index` (int): 청크 인덱스
- `faiss_idx` (int): FAISS 인덱스
- `chunk_hash` (str): 청크 내용 해시 (SHA-256)
- `embedding_config_hash` (str): 파일+config 통합 해시

**용도:**
- 중복 체크
- 내용 변경 감지
- 설정 변경 감지
- 빠른 삭제

---

## 벡터 재구성

VectorStoreManager는 벡터 삭제 시 **재구성 방식**을 사용합니다:

1. 유지할 벡터의 인덱스 리스트 생성
2. 유지할 Document 리스트 추출
3. 기존 벡터 추출 (FAISS reconstruct)
4. 새 인덱스 생성
5. vectorstore 재생성
6. chunk_map 재구축

이 방식은 FAISS의 제약으로 인해 필요하며, 삭제 후 반드시 `save()`를 호출해야 합니다.

---

## FAISS 인덱스 파일 구조

```
data/vectorstore/
├── vectorstore.faiss      # FAISS 인덱스 (벡터 데이터)
└── vectorstore.pkl        # LangChain 메타데이터 (docstore, index_to_docstore_id)
```

- `.faiss` 파일: FAISS 벡터 인덱스 (바이너리)
- `.pkl` 파일: LangChain 메타데이터 (pickle)

---

## 주의사항

1. **LangChain 통합**: VectorStoreManager는 LangChain FAISS 래퍼를 사용하므로 LangChain의 Document 및 메타데이터 규칙을 따릅니다.
2. **chunk_index 필수**: add_texts() 호출 시 메타데이터에 `chunk_index`가 반드시 포함되어야 합니다.
3. **저장 필수**: 벡터 추가/삭제 후에는 반드시 `save()`를 호출하여 디스크에 저장해야 합니다.
4. **재구성 비용**: 벡터 삭제는 인덱스 재구성을 동반하므로 대량 삭제 시 시간이 소요될 수 있습니다.
5. **더미 인덱스**: 초기 로드 시 인덱스가 없으면 더미 인덱스가 자동 생성됩니다. 첫 번째 실제 벡터 추가 시 더미는 자동 삭제됩니다.
6. **distance vs similarity**: FAISS는 L2 거리를 반환하므로 점수가 낮을수록 유사도가 높습니다.

---

## 사용 예시

### 전체 워크플로우

```python
from src.vectorstore import VectorStoreManager

# 1. VectorStoreManager 초기화 및 로드
vm = VectorStoreManager()
vm.load()

# 2. 벡터 추가
texts = ["청크 1", "청크 2"]
metadatas = [
    {
        'file_hash': 'abc123',
        'chunk_index': 0,
        'chunk_hash': 'hash1',
        'file_name': 'doc.pdf',
        'start_page': 1,
        'end_page': 1
    },
    {
        'file_hash': 'abc123',
        'chunk_index': 1,
        'chunk_hash': 'hash2',
        'file_name': 'doc.pdf',
        'start_page': 2,
        'end_page': 2
    }
]
success, start_idx = vm.add_texts(texts, metadatas)

# 3. 저장
if success:
    vm.save()

# 4. 검색
results = vm.search("입찰 요건", top_k=5)
for doc, score in results:
    print(f"[{doc.metadata['file_name']}] Score: {score:.4f}")
    print(doc.page_content[:200])

# 5. 메타데이터 조회
page_chunks = vm.get_by_metadata(file_hash="abc123", start_page=1, end_page=3)

# 6. 벡터 삭제
vm.remove_by_file_hash("abc123")
vm.save()

# 7. 통계 출력
vm.summary()
```
