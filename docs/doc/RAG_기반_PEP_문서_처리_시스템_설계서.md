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

4명의 팀원이 협업하여 PEP 문서를 수집, 전처리, 임베딩, 검색하는 RAG 시스템을 개발합니다.

## 팀 역할 분담

- **신승목**: 문서 수집 및 원본 전처리
- **김명환**: 임베딩 처리 및 FAISS 관리
- **이민규**: LLM 기반 정보 추출 및 요약
- **오형주**: Streamlit UI 개발 및 통합

## 데이터베이스 설계

### 1. 원본 문서 DB (documents.db)

**TB_DOCUMENTS 테이블**:
```sql
CREATE TABLE TB_DOCUMENTS (
    file_hash TEXT PRIMARY KEY,          -- SHA-256 해시
    file_name TEXT NOT NULL,             -- 파일명
    total_pages INTEGER NOT NULL,        -- 총 페이지 수
    file_size INTEGER NOT NULL,          -- 파일 크기 (bytes)
    text_content TEXT,                   -- 전체 텍스트 콘텐츠
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 2. 채팅 히스토리 DB (chat_history.db)

**chat_sessions 테이블**:
- session_id (TEXT, PRIMARY KEY)
- session_name (TEXT)
- created_at, updated_at (TIMESTAMP)
- is_active (BOOLEAN)

**chat_messages 테이블**:
- message_id (INTEGER, PRIMARY KEY)
- session_id (TEXT, FOREIGN KEY)
- role (TEXT: 'user' or 'assistant')
- content (TEXT)
- retrieved_chunks (TEXT: JSON)
- timestamp (TIMESTAMP)

### 3. FAISS 벡터 인덱스

**VectorStoreManager**가 관리하는 통합 인덱스:
- vectorstore.faiss: 모든 문서의 벡터 임베딩
- Document.metadata: 파일명, 페이지 번호, 청크 정보 포함

## 핵심 처리 흐름

### Phase 1: 문서 수집 및 변환
1. PDF 파일 수집
2. 파일 해시 계산 (SHA-256)
3. PyMuPDF로 텍스트 추출
4. documents.db에 저장

### Phase 2: 임베딩 처리
1. DocumentsDB에서 텍스트 조회
2. RecursiveCharacterTextSplitter로 청킹
3. OpenAI API로 임베딩 생성
4. FAISS 인덱스에 추가 (메타데이터 포함)

### Phase 3: 질의응답
1. 사용자 질의 임베딩 생성
2. FAISS 유사도 검색 (L2 distance)
3. LangChain + LLM으로 답변 생성
4. chat_history.db에 저장

### Phase 4: UI
- Streamlit 웹 앱
- 채팅 인터페이스, 세션 관리
- 검색 결과 및 출처 표시

> 📚 **상세 구현**: [시스템 아키텍처 설계서](./시스템_아키텍처_설계서.md)

## 핵심 설계 원칙

1. **파일 해시 중심**: file_hash로 모든 데이터 연결
2. **증분 업데이트**: 변경된 파일만 재처리
3. **독립적 모듈**: 각 개발자 독립 작업 가능
4. **더미 우선 개발**: UI와 로직 병렬 개발
5. **메타데이터 통합**: FAISS Document.metadata 활용
