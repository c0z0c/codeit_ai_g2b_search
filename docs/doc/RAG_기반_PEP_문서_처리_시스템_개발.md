# RAG 기반 PEP 문서 처리 시스템 개발 가이드

## 프로젝트 개요

4명의 팀원이 협업하여 PEP 문서를 수집, 전처리, 임베딩, 검색하는 RAG 시스템을 개발합니다.

## 팀 역할 분담

- **신승목**: 문서 수집 및 원본 전처리
- **김명환**: 임베딩 처리 및 FAISS 관리
- **이민규**: LLM 기반 정보 추출 및 요약
- **오형주**: Streamlit UI 개발 및 통합

## 시스템 아키텍처

### 핵심 설계

- **2개 SQLite + FAISS**: documents.db, chat_history.db, vectorstore.faiss
- **VectorStoreManager**: FAISS 인덱스 및 메타데이터 통합 관리
- **파일 해시 중심**: SHA-256 기반 중복 제거 및 증분 업데이트

### 데이터 흐름

1. PDF → DocumentProcessor → documents.db (TB_DOCUMENTS)
2. EmbeddingProcessor → FAISS (벡터 + 메타데이터)
3. Retrieval → LLMProcessor → chat_history.db

> 📚 **상세 아키텍처**: [시스템 아키텍처 설계서](./시스템_아키텍처_설계서.md)

## 개발 우선순위

### Step 1: 더미 데이터 생성
AI를 활용하여 DB 스키마에 맞는 더미 데이터 생성

### Step 2: DB 스키마 구축
2개 SQLite DB 생성 및 FAISS 인덱스 초기화

### Step 3: 모듈 구조 설계
클래스 및 함수 인터페이스 정의 (초기에 더미 데이터 반환)

### Step 4: UI 프로토타입
더미 데이터로 동작하는 Streamlit 앱 구현

### Step 5: 기능 통합
실제 구현체 개발 및 통합 테스트

## 핵심 설계 원칙

1. **파일 해시 중심**: 모든 데이터는 file_hash로 연결
2. **증분 업데이트**: 변경된 파일만 재처리
3. **독립적 모듈**: 각 개발자가 독립적으로 작업 가능
4. **더미 우선 개발**: UI와 로직을 병렬 개발 가능
5. **메타데이터 추적**: FAISS Document.metadata에 모든 정보 저장

## 개발 원칙

- **MVP 최적화**: 재현성(reproducibility) 우선
- **로깅**: `print` 최소화, 로깅 우선
- **스타일 가이드**: PEP 8 + Black + isort
- **타입힌트**: PEP 484, Docstring: PEP 257

> 📚 **상세 구현**: [시스템 아키텍처 설계서](./시스템_아키텍처_설계서.md)
