---
layout: default
title: "[중급프로젝트] - RAG 기반 PEP 문서 처리 시스템 설계서"
description: "[중급프로젝트] - RAG 기반 PEP 문서 처리 시스템 설계서"
date: 2025-11-08
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# RAG 기반 PEP 문서 처리 시스템 설계서

## 프로젝트 개요

4명의 팀원이 협업하여 PEP 문서를 수집, 전처리, 임베딩, 검색하는 RAG 시스템을 개발합니다. 각 모듈은 독립적으로 개발 가능하며, 파일 해시값을 공통 키로 사용하여 데이터베이스 간 연결합니다.

## 팀 역할 분담

- **신승목**: 문서 수집 및 원본 전처리 (PDF/HWP → Markdown 변환 및 DB 저장)
- **김명환**: 임베딩 처리 (Markdown → 벡터 임베딩 및 FAISS 저장)
- **이민규**: LLM 기반 정보 추출 및 요약 시스템
- **오형주**: Streamlit UI 개발 및 통합

## 데이터베이스 설계

### 1. 원본 문서 DB (SQLite: `documents.db`)

**파일 정보 테이블 (file_info)**
```
- file_hash (TEXT, PRIMARY KEY): 원본 파일 해시값 (SHA-256)
- file_name (TEXT): 파일명 (경로 제외)
- total_pages (INTEGER): 총 페이지 수
- file_size (INTEGER): 파일 크기 (bytes)
- total_chars (INTEGER): 총 글자 수
- total_tokens (INTEGER): 총 토큰 수 (GPT tokenizer 기준)
- created_at (TIMESTAMP): 생성 시간
- updated_at (TIMESTAMP): 수정 시간
```

**페이지 데이터 테이블 (page_data)**
```
- id (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- file_hash (TEXT, FOREIGN KEY): 파일 해시값
- page_number (INTEGER): 페이지 번호
- markdown_content (TEXT): 마크다운 변환 내용
- token_count (INTEGER): 페이지별 토큰 수
- is_empty (BOOLEAN): 빈 페이지 여부
- created_at (TIMESTAMP)
```

### 2. 임베딩 DB (SQLite: `embeddings.db`)

**임베딩 메타데이터 테이블 (embedding_meta)**
```
- embedding_hash (TEXT, PRIMARY KEY): 임베딩 설정 해시값
- file_hash (TEXT): 원본 파일 해시값
- chunk_size (INTEGER): 청킹 크기
- chunk_overlap (INTEGER): 청크 오버랩
- preprocessing_option (TEXT): 전처리 옵션 (JSON)
- embedding_model (TEXT): 임베딩 모델명
- total_chunks (INTEGER): 총 청크 수
- vector_path (TEXT): FAISS 인덱스 파일 경로
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
```

**청크 매핑 테이블 (chunk_mapping)**
```
- chunk_id (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- embedding_hash (TEXT, FOREIGN KEY): 임베딩 해시값
- file_hash (TEXT): 원본 파일 해시값
- file_name (TEXT): 파일명
- start_page (INTEGER): 시작 페이지 번호
- end_page (INTEGER): 종료 페이지 번호
- chunk_text (TEXT): 청크 텍스트
- estimated_tokens (INTEGER): 추정 토큰 수
- vector_index (INTEGER): FAISS 벡터 인덱스
```

### 3. 채팅 히스토리 DB (SQLite: `chat_history.db`)

**세션 테이블 (chat_sessions)**
```
- session_id (TEXT, PRIMARY KEY): 브라우저 세션 ID
- session_name (TEXT): 세션 이름
- created_at (TIMESTAMP)
- updated_at (TIMESTAMP)
- is_active (BOOLEAN): 활성 상태
```

**대화 내역 테이블 (chat_messages)**
```
- message_id (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- session_id (TEXT, FOREIGN KEY): 세션 ID
- role (TEXT): 역할 (user/assistant)
- content (TEXT): 메시지 내용
- retrieved_chunks (TEXT): 검색된 청크 정보 (JSON)
- timestamp (TIMESTAMP)
```

## 핵심 처리 흐름

### Phase 1: 문서 수집 및 변환
1. PDF/HWP 파일 수집 (로컬 또는 데이터 포털 API)
2. 파일 해시 계산 및 중복 확인
3. Markdown 변환 시 페이지 구분자 삽입
   - 빈 페이지: `\n--- [빈페이지] ---\n`
   - 일반 페이지: `\n\n--- 페이지 {page_num} ---\n\n`
4. 기본 전처리 (연속된 개행 정리: `\n\n\n` → `\n\n`)
5. GPT tokenizer로 토큰 수 계산
6. `documents.db`에 저장

### Phase 2: 임베딩 처리
1. 마크다운 문서를 청킹 (설정된 chunk_size 기준)
2. 임베딩 전처리 옵션 적용
   - Markdown 태그 제거 여부
   - HTML 태그 제거 여부
   - 표 구조 유지 여부
3. 파일 해시로 변경사항 감지 (증분 업데이트)
4. 벡터 임베딩 생성 및 FAISS 인덱스 구축
5. 청크별 페이지 범위 및 토큰 수 추정 저장
6. `embeddings.db` 및 FAISS 파일 저장

### Phase 3: 정보 추출 및 요약
1. 사용자 질의 입력
2. 질의 임베딩 생성
3. FAISS에서 유사 청크 검색 (top-k)
4. 검색된 청크로 LLM 프롬프트 구성
5. Langchain 활용한 답변 생성
6. 대화 내역을 `chat_history.db`에 저장

### Phase 4: UI 구현
**왼쪽 사이드바**
- OpenAI API Key 입력
- 데이터 업데이트 버튼
- 임베딩 업데이트 버튼
- 채팅 세션 관리

**메인 영역**
- 채팅 인터페이스
- 검색 결과 및 출처 표시
- 히스토리 뷰어

## 개발 우선순위

### Step 1: 더미 데이터 생성
- AI를 활용하여 각 DB 스키마에 맞는 더미 데이터 3세트 생성
- 실제 PEP 문서 구조를 반영한 샘플 markdown 생성

### Step 2: DB 스키마 구축
- 3개 SQLite DB 생성 및 더미 데이터 삽입
- 각 테이블 간 관계 검증

### Step 3: 모듈 구조 설계
- 각 기능별 클래스 및 함수 인터페이스 정의
- 모든 함수는 초기에 더미 데이터 반환

### Step 4: UI 프로토타입
- 더미 데이터만으로 동작하는 Streamlit 앱 구현
- 실제 기능 연결 전 UI/UX 검증

### Step 5: 기능 통합
- 각 모듈의 실제 구현체 개발
- 더미 함수를 실제 로직으로 대체
- 통합 테스트

## 핵심 설계 원칙

1. **파일 해시 중심 설계**: 모든 테이블은 file_hash를 통해 연결
2. **증분 업데이트**: 변경된 파일만 재처리
3. **독립적 모듈**: 각 개발자가 독립적으로 작업 가능
4. **더미 우선 개발**: UI와 로직을 병렬 개발 가능
5. **메타데이터 추적**: 모든 처리 단계의 설정 및 버전 기록