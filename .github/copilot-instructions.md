# copilot-instructions.md

## 0. 핵심 원칙
- 항상 최신 정보를 검색·반영
- 소스 및 문서는 매번 새로 읽어 사용
- 이모지 사용 금지, 불필요한 장황함 금지
- try except 금지, 오류 발생 시 즉시 중단

## 1. 개발 원칙
- **MVP 최적화**: 재현성(reproducibility) 우선
- **코드 제안 단위**: 함수/메서드/클래스 단위
- **코드 위치**: 셀 번호 기준으로 제안
- **로깅**: `print` 최소화, 로깅 우선
- **코드 길이 규칙**  
  - MVP
  - 20라인 이하 → 바로 제안 가능  
  - 20라인 이상 → 반드시 확인 질문 후 제안
- **스타일 가이드**  
  - PEP 8 + Black + isort  
  - PEP 484 (타입힌트), PEP 257 (Docstring)
- **헬퍼 활용**: `helper_utils.py`, `helper_c0z0c_dev.py` 존재 시 적극 활용

## 2. 상호작용 프로토콜
1. 요구 요약: 목표·제약·산출물
2. 접근 전략 설명: 선택지, 트레이드오프, 변경 파일, 성능/재현성 영향
3. 확인 질문: “이 전략으로 진행할까요? (예/아니오/수정)”
4. 승인 후 코딩: 최소 실행 예제, 경로/의존성/헬퍼 호출, 타입힌트, Docstring 포함
5. 단문·20라인 이하 → 확인 질문 생략 가능

## 3. 프로젝트 개요
**RAG 기반 공공데이터 포털(PEP) 문서 처리 시스템**
- PDF/HWP → Markdown → 벡터 임베딩 → FAISS 검색 → LLM 답변 생성
- 4계층 아키텍처: UI (Streamlit) → Application (Processor) → Data Access (DB/VectorStore) → Storage (SQLite×2 + FAISS)
- 파일 해시(SHA-256) 기반 중복 제거 및 증분 업데이트

## 4. 핵심 워크플로우

### 문서 처리 파이프라인
1. **PDF/HWP → DocumentsDB**  
   `DocumentProcessor.process_pdf()` → 페이지별 Markdown 추출 → `DocumentsDB.insert_document()`
   
2. **DocumentsDB → FAISS Index**  
   `EmbeddingProcessor.sync_with_docs_db()` → 청킹(LangChain) → OpenAI 임베딩 → `VectorStoreManager.add_embeddings()`
   
3. **질의응답 (RAG)**  
   `Retrieval.search()` → FAISS 유사도 검색 → `LLMProcessor.generate_response()` → 답변 + 출처

### 주요 실행 스크립트
```bash
# 1. PDF 처리 → DB 저장
python scripts/pipeline_pdf_to_document_db.py "path/to/file.pdf"

# 2. DB → 벡터 임베딩 (전체 동기화)
python scripts/pipeline_document_db_to_vector.py

# 3. Streamlit 앱 실행
streamlit run app.py
```

### 개발/디버깅 워크플로우
- **Jupyter Notebook 개발**: `scripts/김명환/로직검토_*.ipynb` 참고 (테스트 코드 포함)
- **헬퍼 함수**: `helper_utils.py`(로깅, 파일 I/O), `helper_c0z0c_dev.py`(폰트, pandas 확장, 캐시)
- **Config 중심 설계**: `Config.get_instance()` 싱글톤으로 모든 설정 로드 (`config/config.json`)

## 5. 아키텍처 핵심 패턴

### 싱글톤 Config 패턴
```python
from src.config import get_config
config = get_config("config/config.json")  # 모든 모듈에서 동일한 인스턴스 공유
```
- 모든 DB 경로, 청킹 파라미터, LLM 설정을 `Config` 클래스에서 관리
- 환경 변수(`OPENAI_API_KEY`) 자동 로드 및 오버라이드

### 파일 해시 기반 추적
- **file_hash**: `DocumentProcessor.calculate_file_hash()` → SHA-256 해시
- **중복 제거**: 동일 파일 재처리 방지 (`DocumentsDB.file_exists()`)
- **증분 업데이트**: `EmbeddingProcessor.sync_with_docs_db()` → 새 문서만 임베딩

### 통합 FAISS 관리
- **단일 인덱스**: `VectorStoreManager`가 모든 문서의 벡터를 하나의 FAISS 인덱스로 관리
- **메타데이터 통합**: `Document.metadata`에 `file_name`, `start_page`, `end_page`, `file_hash` 포함
- **검색 필터링**: `filter_metadata={'file_hash': ...}` 파라미터로 특정 문서 내 검색

### 데이터베이스 구조
- **documents.db**: `TB_DOCUMENTS` (파일 해시, 페이지별 텍스트, 토큰 수)
- **chat_history.db**: `chat_sessions`, `chat_messages` (세션 관리, 대화 내역, 검색 컨텍스트)
- **FAISS Index**: `data/vectorstore.faiss` (벡터 + 메타데이터 통합)

## 6. 주요 모듈 인터페이스

### DocumentProcessor (`src/processors/document_processor.py`)
```python
processor = DocumentProcessor(config=config)
file_hash, success = processor.process_pdf("path/to/file.pdf")
# → SHA-256 해시 반환, DocumentsDB에 페이지별 텍스트 저장
```

### EmbeddingProcessor (`src/processors/embedding_processor.py`)
```python
embedder = EmbeddingProcessor(config=config)
result = embedder.sync_with_docs_db(api_key=openai_api_key)
# → DocumentsDB의 모든 문서를 FAISS 인덱스로 동기화
```

### Retrieval (`src/llm/retrieval.py`)
```python
retrieval = Retrieval(config=config)
results = retrieval.search(query="질문", top_k=5)
# → FAISS 검색 결과 (file_name, start_page, end_page, text, distance)
```

### LLMProcessor (`src/llm/llm_processor.py`)
```python
llm = LLMProcessor(session_id=session_id, config=config)
response = llm.generate_response(query="질문", retrieved_chunks=results)
# → RAG 답변 생성 + ChatHistoryDB에 저장
```

## 7. 문서화 (GitHub Pages)
- 목차: `1.` `1.1.` `1.1.1` 형식
- mermaid: 노드 라벨 큰따옴표 `A["노드"]`
- 수식: `$$` 블록 우선
- 용어: 한영 병기 (예: normalization, 노멀라이제이션)
- 이모지 금지

## 8. 빠른 참조

### 필수 파일 위치
- Config: `config/config.json` (읽기), `src/config.py` (클래스 정의)
- DB: `data/documents.db`, `data/chat_history.db`
- FAISS: `data/vectorstore.faiss`
- 로그: `logs/` (자동 생성)

### 일반적인 오류 해결
- **API 키 누락**: `.env` 파일 확인 또는 `os.environ['OPENAI_API_KEY']` 설정
- **DB 초기화 실패**: 경로 확인 (`config.DOCUMENTS_DB_PATH` 등)
- **FAISS 인덱스 없음**: `EmbeddingProcessor.sync_with_docs_db()` 실행
- **import 오류**: `sys.path.insert(0, str(project_root))` 추가
