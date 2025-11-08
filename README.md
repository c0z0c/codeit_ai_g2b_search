# RAG 기반 PEP 문서 처리 시스템

OpenAI LLM과 LangChain을 활용한 문서 검색 및 질의응답 시스템

## 📋 프로젝트 개요

이 시스템은 PDF 문서를 처리하여 벡터 임베딩을 생성하고, 사용자 질의에 대해 관련 문서를 검색하여 LLM 기반 답변을 제공하는 RAG(Retrieval-Augmented Generation) 시스템입니다.

### 주요 기능

- ✅ PDF 문서 자동 처리 및 마크다운 변환
- ✅ OpenAI 임베딩 기반 벡터 검색
- ✅ FAISS 벡터 스토어
- ✅ LangChain + OpenAI GPT 기반 질의응답
- ✅ Streamlit 웹 UI
- ✅ 채팅 히스토리 관리
- ✅ 다중 세션 지원

## 🏗️ 시스템 아키텍처

```
codeit_ai_g2b_search/
├── src/
│   ├── db/                  # 데이터베이스 모듈
│   │   ├── documents_db.py     # 원본 문서 DB
│   │   ├── embeddings_db.py    # 임베딩 DB
│   │   └── chat_history_db.py  # 채팅 히스토리 DB
│   ├── processors/          # 처리 모듈
│   │   ├── document_processor.py   # PDF → Markdown
│   │   └── embedding_processor.py  # 임베딩 생성
│   ├── llm/                 # LLM 모듈
│   │   ├── retrieval.py        # 벡터 검색
│   │   └── llm_processor.py    # LLM 응답 생성
│   └── utils/               # 유틸리티
│       └── logging_config.py   # 로깅 설정
├── config/
│   └── settings.yaml        # 설정 파일
├── data/                    # 데이터 디렉토리
│   ├── raw/                    # 원본 파일
│   ├── processed/              # 처리된 파일
│   ├── vectorstore/            # FAISS 인덱스
│   ├── documents.db            # 문서 DB
│   ├── embeddings.db           # 임베딩 DB
│   └── chat_history.db         # 채팅 DB
├── scripts/                 # 스크립트
│   └── generate_dummy_simple.py  # 더미 데이터 생성
└── app.py                   # Streamlit 앱

```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.11+ 권장
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. 더미 데이터 생성 (테스트용)

```bash
python scripts/generate_dummy_simple.py
```

### 4. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`에 접속합니다.

## 📊 데이터베이스 구조

### 1. documents.db - 원본 문서 DB

**file_info 테이블**
- `file_hash`: 파일 해시값 (PRIMARY KEY)
- `file_name`: 파일명
- `total_pages`: 총 페이지 수
- `file_size`: 파일 크기
- `total_chars`: 총 글자 수
- `total_tokens`: 총 토큰 수

**page_data 테이블**
- `id`: 자동 증가 ID
- `file_hash`: 파일 해시값 (FOREIGN KEY)
- `page_number`: 페이지 번호
- `markdown_content`: 마크다운 콘텐츠
- `token_count`: 페이지별 토큰 수
- `is_empty`: 빈 페이지 여부

### 2. embeddings.db - 임베딩 DB

**embedding_meta 테이블**
- `embedding_hash`: 임베딩 해시값 (PRIMARY KEY)
- `file_hash`: 원본 파일 해시값
- `chunk_size`: 청킹 크기
- `chunk_overlap`: 청크 오버랩
- `embedding_model`: 임베딩 모델명
- `total_chunks`: 총 청크 수
- `faiss_index_path`: FAISS 인덱스 경로

**chunk_mapping 테이블**
- `chunk_id`: 자동 증가 ID
- `embedding_hash`: 임베딩 해시값 (FOREIGN KEY)
- `file_hash`: 파일 해시값
- `file_name`: 파일명
- `chunk_text`: 청크 텍스트
- `vector_index`: FAISS 벡터 인덱스

### 3. chat_history.db - 채팅 히스토리 DB

**chat_sessions 테이블**
- `session_id`: 세션 ID (PRIMARY KEY)
- `session_name`: 세션 이름
- `is_active`: 활성 상태

**chat_messages 테이블**
- `message_id`: 자동 증가 ID
- `session_id`: 세션 ID (FOREIGN KEY)
- `role`: 역할 (user/assistant)
- `content`: 메시지 내용
- `retrieved_chunks`: 검색된 청크 정보 (JSON)

## 🔧 사용 방법

### PDF 문서 처리 (Python API)

```python
from src.processors.document_processor import DocumentProcessor

processor = DocumentProcessor()
file_hash = processor.process_pdf("path/to/document.pdf")
print(f"처리 완료: {file_hash}")
```

### 임베딩 생성

```python
from src.processors.embedding_processor import EmbeddingProcessor

embedder = EmbeddingProcessor(chunk_size=1000, chunk_overlap=200)
embedding_hash = embedder.process_document(
    file_hash=file_hash,
    api_key="your_openai_api_key"
)
print(f"임베딩 생성 완료: {embedding_hash}")
```

### 문서 검색 및 질의응답

```python
from src.llm.retrieval import Retrieval
from src.llm.llm_processor import LLMProcessor

# 검색
retrieval = Retrieval()
results = retrieval.search(
    query="공공데이터 품질관리에서 완전성이란?",
    embedding_hash=embedding_hash,
    top_k=3
)

# LLM 응답 생성
llm = LLMProcessor()
response = llm.generate_response(
    query="공공데이터 품질관리에서 완전성이란?",
    retrieved_chunks=results
)
print(response)
```

## ⚙️ 설정 (config/settings.yaml)

```yaml
# 청킹 설정
chunking:
  chunk_size: 1000
  chunk_overlap: 200

# 임베딩 설정
embedding:
  model: "text-embedding-3-small"
  dimension: 1536

# LLM 설정
llm:
  model: "gpt-4o-mini"
  temperature: 0.7

# 검색 설정
retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

## 📦 주요 의존성

- **langchain**: LLM 체인 구성
- **langchain-openai**: OpenAI 통합
- **openai**: OpenAI API 클라이언트
- **faiss-cpu**: 벡터 검색
- **tiktoken**: 토큰 카운팅
- **streamlit**: 웹 UI
- **pymupdf**: PDF 처리

## 🎯 개발 원칙

- **MVP 최적화**: 재현성(reproducibility) 우선
- **로깅 우선**: `print` 최소화, 로깅 활용
- **타입 힌트**: PEP 484 준수
- **Docstring**: PEP 257 준수
- **스타일 가이드**: PEP 8, Black, isort

## 📝 더미 데이터

시스템에는 테스트용 더미 데이터가 포함되어 있습니다:

1. **공공데이터_품질관리_가이드라인_2024.pdf** (3페이지)
   - 공공데이터 품질관리 절차 및 지표

2. **AI_학습용_데이터_구축_지침서_v2.pdf** (3페이지)
   - AI 데이터 수집, 가공, 라벨링 방법

샘플 채팅 세션 2개도 포함되어 있습니다.

## 🧪 테스트

```bash
# 더미 데이터 생성 및 확인
python scripts/generate_dummy_simple.py

# Streamlit 앱 실행 및 테스트
streamlit run app.py
```

## 📌 주요 특징

### 1. 파일 해시 기반 중복 방지
- SHA-256 해시로 파일 식별
- 동일 파일 재처리 방지

### 2. 증분 업데이트 지원
- 새로운 문서만 처리
- 변경된 문서 자동 감지

### 3. 유연한 청킹 옵션
- 크기 조절 가능
- 오버랩 설정 가능

### 4. 메타데이터 추적
- 모든 처리 단계 기록
- 설정 및 버전 관리

## 🤝 기여

프로젝트 개선 제안이나 버그 리포트는 Issue를 통해 제출해주세요.

## 📄 라이센스

이 프로젝트는 교육용으로 제작되었습니다.

## 📞 문의

프로젝트 관련 문의사항이 있으시면 Issue를 생성해주세요.
