# -*- coding: utf-8 -*-
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hashlib
from datetime import datetime
from src.db import DocumentsDB, ChatHistoryDB
import tiktoken

# tiktoken 인코더 초기화
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
except:
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(tokenizer.encode(text))

def calculate_file_hash(file_name, content):
    data = f"{file_name}_{content}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()

# DB 초기화
docs_db = DocumentsDB()
chat_db = ChatHistoryDB()

print("더미 데이터 생성 시작...")

# 문서 1: 공공데이터 품질관리 가이드라인
doc1_name = "공공데이터_품질관리_가이드라인_2024.pdf"
doc1_pages = [
    """# 공공데이터 품질관리 가이드라인

## 제1장 총칙

### 제1조 (목적)
이 가이드라인은 공공데이터의 품질관리에 필요한 사항을 규정함을 목적으로 한다.

### 제2조 (적용범위)
이 가이드라인은 공공기관이 보유·관리하는 모든 공공데이터에 적용한다.""",

    """## 제2장 품질관리 체계

### 제4조 (품질관리 조직)
공공기관의 장은 공공데이터 품질관리를 위한 전담조직을 구성·운영하여야 한다.

### 제5조 (품질관리 절차)
공공데이터 품질관리는 계획 수립, 데이터 진단, 개선 실행, 모니터링 단계로 수행한다.""",

    """## 제3장 품질 지표

### 제6조 (품질 지표)
공공데이터의 품질은 다음의 지표로 측정한다:
- 정확성(Accuracy): 데이터가 실제 값과 일치하는 정도
- 완전성(Completeness): 필수 항목이 누락 없이 기록된 정도
- 일관성(Consistency): 데이터가 규칙과 형식에 부합하는 정도"""
]

doc1_content = "\n\n".join(doc1_pages)
doc1_hash = calculate_file_hash(doc1_name, doc1_content)
doc1_tokens = count_tokens(doc1_content)

docs_db.insert_file_info(
    file_hash=doc1_hash,
    file_name=doc1_name,
    total_pages=len(doc1_pages),
    file_size=len(doc1_content.encode('utf-8')),
    total_chars=len(doc1_content),
    total_tokens=doc1_tokens
)

for i, page_content in enumerate(doc1_pages, 1):
    docs_db.insert_page_data(
        file_hash=doc1_hash,
        page_number=i,
        markdown_content=page_content,
        token_count=count_tokens(page_content),
        is_empty=False
    )

print(f"문서 1 삽입 완료: {doc1_name}")

# 문서 2: AI 학습용 데이터 구축 지침서
doc2_name = "AI_학습용_데이터_구축_지침서_v2.pdf"
doc2_pages = [
    """# AI 학습용 데이터 구축 지침서

## 1. 데이터 기획

### 1.1 목적 정의
AI 모델의 학습 목표를 명확히 하고 활용 분야를 정의한다.

### 1.2 데이터 설계
구축할 데이터의 유형과 규모를 결정한다.""",

    """## 2. 데이터 수집

### 2.1 수집 방법
크롤링, API 활용, 직접 생성, 구매/제휴 등의 방법으로 데이터를 수집한다.

### 2.2 데이터 다양성 확보
AI 모델의 일반화 성능을 위해 다양한 데이터를 수집한다.""",

    """## 3. 데이터 가공

### 3.1 전처리
수집된 원시 데이터를 AI 학습에 적합한 형태로 변환한다.

### 3.2 라벨링
데이터에 정답 레이블을 부여하는 과정을 수행한다."""
]

doc2_content = "\n\n".join(doc2_pages)
doc2_hash = calculate_file_hash(doc2_name, doc2_content)
doc2_tokens = count_tokens(doc2_content)

docs_db.insert_file_info(
    file_hash=doc2_hash,
    file_name=doc2_name,
    total_pages=len(doc2_pages),
    file_size=len(doc2_content.encode('utf-8')),
    total_chars=len(doc2_content),
    total_tokens=doc2_tokens
)

for i, page_content in enumerate(doc2_pages, 1):
    docs_db.insert_page_data(
        file_hash=doc2_hash,
        page_number=i,
        markdown_content=page_content,
        token_count=count_tokens(page_content),
        is_empty=False
    )

print(f"문서 2 삽입 완료: {doc2_name}")

# 채팅 세션 생성
session1_id = chat_db.create_session("공공데이터 품질관리 문의")
chat_db.add_message(
    session1_id,
    "user",
    "공공데이터 품질관리에서 완전성 지표는 무엇을 의미하나요?"
)
chat_db.add_message(
    session1_id,
    "assistant",
    "완전성(Completeness)은 필수 항목이 누락 없이 기록된 정도를 의미합니다.",
    retrieved_chunks=[{"file_name": doc1_name, "page": 3, "similarity": 0.92}]
)

session2_id = chat_db.create_session("AI 데이터 구축 질문")
chat_db.add_message(
    session2_id,
    "user",
    "AI 학습용 데이터의 라벨링은 어떻게 하나요?"
)
chat_db.add_message(
    session2_id,
    "assistant",
    "라벨링은 데이터에 정답 레이블을 부여하는 과정입니다. 분류, 객체 탐지, 세그멘테이션 등의 방법이 있습니다.",
    retrieved_chunks=[{"file_name": doc2_name, "page": 3, "similarity": 0.89}]
)

print("채팅 세션 생성 완료")

# 통계 출력
doc_stats = docs_db.get_document_stats()
chat_stats = chat_db.get_chat_stats()

print("\n=== 데이터 삽입 완료 ===")
print(f"\n[문서 통계]")
print(f"- 총 파일 수: {doc_stats['total_files']}개")
print(f"- 총 페이지 수: {doc_stats['total_pages']}페이지")
print(f"- 총 토큰 수: {doc_stats['total_tokens']:,}개")

print(f"\n[채팅 통계]")
print(f"- 총 세션 수: {chat_stats['total_sessions']}개")
print(f"- 총 메시지 수: {chat_stats['total_messages']}개")
