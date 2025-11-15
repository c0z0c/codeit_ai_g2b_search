# RAG 기반 PEP 문서 처리 시스템

> **[중급 프로젝트]** OpenAI LLM + LangChain 기반 문서 검색 및 질의응답 시스템
>
> 📅 **프로젝트 기간**: 2025.11.08 ~ 2025.11.28 (3주)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector-red.svg)](https://github.com/facebookresearch/faiss)

## 📋 프로젝트 개요

PDF/HWP 형태의 PEP(공공데이터) 문서를 자동으로 처리하여 벡터 임베딩을 생성하고, 사용자 질의에 대해 관련 문서를 검색하여 LLM 기반 답변을 제공하는 **RAG(Retrieval-Augmented Generation)** 시스템입니다.

### 📅 프로젝트 기간
**2025년 11월 10일 ~ 2025년 11월 28일**

- **신승목**: 문서 수집 및 원본 전처리 (PDF/HWP → Markdown 변환 및 DB 저장)
- **[김명환](https://c0z0c.github.io)**: 임베딩 처리 (Markdown → 벡터 임베딩 및 FAISS 저장)
- **이민규**: LLM 기반 정보 추출 및 요약 시스템
- **오형주**: Streamlit UI 개발 및 통합

```mermaid
gantt
    title RAG PEP 프로젝트 타임라인 (11월 10일~28일)
    dateFormat  YYYY-MM-DD

    section Week 1: 기반 구축 (11/10~14)
    환경 설정 및 초기화        :w1d1, 2025-11-10, 1d
    더미 데이터 생성           :w1d2, 2025-11-11, 1d
    DB 스키마 구축             :w1d3, 2025-11-12, 2d
    UI 프로토타입 개발         :w1d4, 2025-11-12, 2d
    Week 1 통합 테스트         :milestone, m1, 2025-11-14, 0d

    section Week 2: 핵심 기능 개발 (11/17~21)
    문서 수집 및 변환 (신승목) :w2d1, 2025-11-17, 3d
    임베딩 처리 (김명환)       :w2d2, 2025-11-17, 3d
    LLM 챗봇 개발 (이민규)     :w2d3, 2025-11-17, 3d
    UI 통합 개발 (오형주)      :w2d4, 2025-11-17, 3d
    모듈 통합 작업              :w2d5, 2025-11-19, 3d
    Week 2 통합 완료            :milestone, m2, 2025-11-21, 0d

    section Week 3: 최적화 및 마무리 (11/24~28)
    전체 통합 테스트           :w3d1, 2025-11-24, 2d
    성능 평가 및 최적화        :w3d2, 2025-11-25, 2d
    문서화 및 README           :w3d3, 2025-11-26, 1d
    발표 자료 준비             :w3d4, 2025-11-27, 2d
    최종 발표                  :milestone, m3, 2025-11-28, 0d
```

## 📝 협업일지

팀원별 개발 과정 및 학습 내용을 기록한 협업일지입니다.

- [김명환 협업일지 (Project Manager)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/김명환/)
- [신승목 협업일지 (Data Engineer)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/신승목/)
- [오형주 협업일지 (Model Architect)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/오형주/)
- [이민규 협업일지 (Experimentation Lead)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/이민규/)

- [팀 회의록](https://c0z0c.github.io/codeit_ai_g2b_search/회의록/)


### 핵심 기능

- ✅ **문서 처리**: PDF → Markdown 자동 변환 (페이지 단위)
- ✅ **벡터 임베딩**: OpenAI text-embedding-3-small 모델
- ✅ **벡터 검색**: FAISS 기반 유사도 검색 (L2 distance)
- ✅ **RAG 답변**: LangChain + GPT-4o-mini
- ✅ **웹 UI**: Streamlit 기반 대화형 인터페이스
- ✅ **세션 관리**: 채팅 히스토리 저장 및 복원
- ✅ **출처 추적**: 답변의 근거 문서 및 페이지 표시

### 🎯 핵심 설계 원칙

- **파일 해시 기반 추적**: SHA-256 해시로 중복 제거 및 증분 업데이트
- **모듈화**: 독립적인 DB/Processor/LLM 모듈 구성
- **메타데이터 관리**: 모든 처리 단계 및 설정 기록

## 🏗️ 시스템 아키텍처

4계층 구조 (UI → Application → Data Access → Storage)로 구성된 모듈화 시스템입니다.

### 핵심 구성

- **UI 계층**: Streamlit 웹 앱
- **Application 계층**: DocumentProcessor, EmbeddingProcessor, LLMProcessor
- **Data Access 계층**: DocumentsDB, VectorStoreManager, ChatHistoryDB
- **Storage 계층**: SQLite × 2 (documents.db, chat_history.db) + FAISS Index

### 주요 특징

- **통합 FAISS 관리**: 모든 문서의 임베딩을 단일 FAISS 인덱스로 관리
- **메타데이터 통합**: Document.metadata에 파일명, 페이지 번호 포함
- **파일 해시 기반 추적**: SHA-256으로 중복 제거 및 증분 업데이트

> 📚 **상세 문서**: [시스템 아키텍처 설계서](docs/doc/시스템_아키텍처_설계서.md)

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

### 핵심 DB: 2개 SQLite + FAISS

| DB | 테이블 | 주요 역할 |
|----|--------|-----------|
| **documents.db** | TB_DOCUMENTS | 문서 텍스트 콘텐츠 저장 |
| **chat_history.db** | chat_sessions<br/>chat_messages | 세션 관리<br/>대화 내역 및 출처 저장 |
| **FAISS Index** | vectorstore.faiss | 벡터 + 메타데이터 통합 관리 |

### 특징

- **EmbeddingsDB 제거**: FAISS 내부 메타데이터만 사용
- **VectorStoreManager**: FAISS 인덱스 및 메타데이터 통합 관리
- **파일 해시 중심**: SHA-256 기반 중복 제거

> 📚 **상세 스키마**: [시스템 아키텍처 설계서](docs/doc/시스템_아키텍처_설계서.md#데이터베이스-er-다이어그램)

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

## 📦 기술 스택

| 카테고리 | 기술 | 용도 |
|---------|------|------|
| **언어** | Python 3.11+ | 주 개발 언어 |
| **LLM** | OpenAI API | GPT-4o-mini, text-embedding-3-small |
| **프레임워크** | LangChain | RAG 파이프라인 구성 |
| **벡터 DB** | FAISS | 유사도 검색 (L2 distance) |
| **문서 처리** | PyMuPDF | PDF 파싱 및 텍스트 추출 |
| **토큰화** | tiktoken | GPT tokenizer |
| **데이터베이스** | SQLite 3.x | 메타데이터 저장 |
| **UI** | Streamlit | 웹 인터페이스 |

## 🔄 데이터 처리 흐름

1. **문서 처리**: PDF → DocumentProcessor → DocumentsDB (텍스트 저장)
2. **임베딩 생성**: EmbeddingProcessor → FAISS 인덱스 (벡터 + 메타데이터)
3. **질의응답**: 사용자 질의 → Retrieval (FAISS 검색) → LLMProcessor (RAG 답변)
4. **히스토리 저장**: ChatHistoryDB (대화 내역 및 출처)

> 📚 **상세 흐름도**: [시스템 아키텍처 설계서](docs/doc/시스템_아키텍처_설계서.md#데이터-흐름)

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

## 🎯 핵심 특징

- **중복 제거**: SHA-256 파일 해시로 동일 파일 자동 감지
- **증분 업데이트**: 변경된 문서만 재처리 (파일 해시 기반)
- **출처 추적**: Document.metadata에 파일명, 페이지 번호 포함
- **통합 FAISS**: 단일 인덱스로 전체 문서 관리, 메타데이터 통합
- **거리 기반 검색**: FAISS L2 거리 직접 사용 (작을수록 유사도 높음)

> 📚 **구현 세부사항**: [시스템 아키텍처 설계서](docs/doc/시스템_아키텍처_설계서.md#핵심-컴포넌트)

## 📚 프로젝트 문서

- 📖 [RAG 기반 PEP 문서 처리 시스템 설계서](docs/doc/RAG_기반_PEP_문서_처리_시스템_설계서.md)
- 🏗️ [시스템 아키텍처 설계서](docs/doc/시스템_아키텍처_설계서.md) (Mermaid 다이어그램 포함)
- ✅ [프로젝트 체크리스트](docs/doc/프로젝트_체크리스트.md) (3주 일정)
- 👥 [개발자별 체크리스트](docs/doc/개발자별_체크리스트.md)

## 👥 팀 구성

| 역할 | 담당 모듈 | 주요 작업 |
|------|-----------|-----------|
| **신승목** | 문서 수집 및 전처리 | PDF/HWP → Markdown, DocumentsDB |
| **김명환** | 임베딩 처리 | 텍스트 청킹, 벡터 임베딩, FAISS 인덱싱 |
| **이민규** | LLM 기반 정보 추출 | RAG 파이프라인, 프롬프트 엔지니어링 |
| **오형주** | UI 개발 및 통합 | Streamlit 앱, 전체 모듈 통합 |

## 🤝 기여 및 문의

- **이슈**: 버그 리포트 및 기능 제안
- **문서**: `docs/` 디렉토리 참고
- **라이선스**: 교육용 프로젝트

---

**프로젝트 기간**: 2025.11.08 ~ 2025.11.28 (3주)
**문서 버전**: 1.0
**최종 업데이트**: 2025-11-08
