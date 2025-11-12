---
layout: default
title: "[중급프로젝트] - embedding_hash의 필요성"
description: "embedding_hash의 필요성"
date: 2025-11-08
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
mermaid: true
math: true
---


## 요구사항 요약

`embedding_hash`의 필요성 질문 - `file_hash`만으로 충분한지 검토 요청

---

## embedding_hash가 필요한 이유

### 핵심 문제 상황

**동일 문서, 다른 임베딩 설정:**

````python
# 시나리오: 김명환가 최적 설정을 찾기 위해 실험
file_hash = "abc123..."  # 동일한 PEP-8.pdf

# 실험 1: 작은 청크
embedding_hash_1 = create_and_save_embeddings(
    file_hash=file_hash,
    chunk_size=500,
    chunk_overlap=100,
    preprocessing_options={'remove_markdown': True}
)
# 결과: embedding_hash_1 = "e7a9b3..."

# 실험 2: 큰 청크
embedding_hash_2 = create_and_save_embeddings(
    file_hash=file_hash,
    chunk_size=1500,
    chunk_overlap=300,
    preprocessing_options={'remove_markdown': False}
)
# 결과: embedding_hash_2 = "f4c2d1..."

# 두 실험은 다른 FAISS 인덱스를 생성
# → file_hash만으로는 구분 불가
````

---

## 실제 사용 시나리오

### 1. A/B 테스트

````python
def compare_retrieval_quality():
    """청크 크기별 검색 품질 비교"""
    
    file_hash = load_pep_document("PEP-484.pdf")
    
    # 설정 A: 작은 청크
    emb_hash_small = create_embeddings(
        file_hash, 
        chunk_size=500, 
        chunk_overlap=100
    )
    
    # 설정 B: 큰 청크
    emb_hash_large = create_embeddings(
        file_hash,
        chunk_size=1500,
        chunk_overlap=300
    )
    
    # 동일 질의로 성능 비교
    query = "타입 힌트 사용 방법은?"
    
    results_small = search_similar_chunks(query, emb_hash_small)
    results_large = search_similar_chunks(query, emb_hash_large)
    
    # 각 설정별 독립적인 결과 추적 가능
    print(f"Small chunks precision: {evaluate(results_small)}")
    print(f"Large chunks precision: {evaluate(results_large)}")
````

### 2. 점진적 최적화

````python
# Week 1: 프로토타입
embedding_v1 = create_embeddings(
    file_hash,
    chunk_size=1000,
    chunk_overlap=200,
    preprocessing_options={'remove_markdown': False}
)
# → embedding_hash = "aaa111..."

# Week 2: 마크다운 제거 실험
embedding_v2 = create_embeddings(
    file_hash,
    chunk_size=1000,
    chunk_overlap=200,
    preprocessing_options={'remove_markdown': True}  # 변경
)
# → embedding_hash = "bbb222..." (다른 해시)

# Week 3: 최종 설정
embedding_v3 = create_embeddings(
    file_hash,
    chunk_size=800,
    chunk_overlap=150,
    preprocessing_options={'remove_markdown': True}
)
# → embedding_hash = "ccc333..."

# v1, v2, v3 모두 보존하여 비교 가능
````

---

## file_hash만 사용 시 문제점

### embeddings.db 스키마 비교

**embedding_hash 없이 설계하면:**

```sql
-- 잘못된 설계
CREATE TABLE embedding_meta (
    file_hash TEXT PRIMARY KEY,  -- 문제: 하나의 설정만 저장 가능
    chunk_size INTEGER,
    faiss_index_path TEXT
);
```

**문제 상황:**

````python
# 첫 번째 임베딩 생성
create_embeddings(file_hash, chunk_size=500, chunk_overlap=100)
# → embeddings.db에 저장

# 두 번째 임베딩 생성 (다른 설정)
create_embeddings(file_hash, chunk_size=1500, chunk_overlap=300)
# → 첫 번째 데이터를 덮어씀 (손실)
# → 이전 FAISS 인덱스 파일은 남아있지만 메타데이터 없음
````

---

## 올바른 설계 (현재 방식)

### embedding_hash로 구분

```sql
-- 올바른 설계
CREATE TABLE embedding_meta (
    embedding_hash TEXT PRIMARY KEY,  -- 설정별 고유 식별자
    file_hash TEXT,                   -- 원본 문서 참조
    chunk_size INTEGER,
    chunk_overlap INTEGER,
    preprocessing_option TEXT,
    faiss_index_path TEXT
);
```

**장점:**

1. **다중 설정 보존**: 동일 문서에 여러 임베딩 설정 공존
2. **실험 추적**: 각 설정별 성능 메트릭 저장
3. **롤백 가능**: 이전 설정으로 복원 가능
4. **재계산 방지**: 동일 설정 재사용

---

## 데이터베이스 예시

### embeddings.db 실제 데이터

| embedding_hash | file_hash | chunk_size | chunk_overlap | preprocessing_option | faiss_index_path |
|----------------|-----------|------------|---------------|---------------------|------------------|
| `e7a9b3...` | `abc123...` | 500 | 100 | `{"remove_markdown": true}` | `faiss_e7a9b3.index` |
| `f4c2d1...` | `abc123...` | 1500 | 300 | `{"remove_markdown": false}` | `faiss_f4c2d1.index` |
| `a1b2c3...` | `def456...` | 800 | 150 | `{"remove_markdown": true}` | `faiss_a1b2c3.index` |

→ 첫 두 행은 **동일 문서** (`file_hash=abc123...`)지만 **다른 임베딩 설정**

---

## 결론: embedding_hash는 필수

### 필요성 정리

| 구분 | file_hash만 사용 | file_hash + embedding_hash |
|------|-----------------|---------------------------|
| 다중 설정 저장 | ✗ 불가능 (덮어씀) | ✓ 가능 |
| A/B 테스트 | ✗ 불가능 | ✓ 가능 |
| 실험 추적 | ✗ 어려움 | ✓ 용이 |
| 재계산 방지 | △ 부분적 | ✓ 완전 |
| 설정 롤백 | ✗ 불가능 | ✓ 가능 |

### embedding_hash 계산 공식

$$
\text{embedding\_hash} = \text{SHA256}(\text{file\_hash} \| \text{chunk\_size} \| \text{chunk\_overlap} \| \text{options})
$$

````python
import hashlib
import json

def calculate_embedding_hash(file_hash, chunk_size, chunk_overlap, preprocessing_option):
    """임베딩 설정 해시 계산
    
    Args:
        file_hash: 원본 파일 해시 (SHA-256)
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        preprocessing_option: 전처리 옵션 (dict)
    
    Returns:
        embedding_hash (str): 임베딩 설정 고유 식별자
    """
    config_str = (
        f"{file_hash}_"
        f"{chunk_size}_"
        f"{chunk_overlap}_"
        f"{json.dumps(preprocessing_option, sort_keys=True)}"
    )
    return hashlib.sha256(config_str.encode()).hexdigest()
````

**핵심**: `embedding_hash`는 "어떤 설정으로 임베딩했는가"를 추적하는 필수 키입니다.