---
layout: default
title: "[중급프로젝트] - Google Cloud VM 시스템 정보"
description: "Google Cloud VM 시스템 정보"
date: 2025-11-15
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
mermaid: true
math: true
---

# Google Cloud VM 시스템 정보

## 개요

본 문서는 Codeit AI G2B Search 프로젝트가 실행되는 Google Cloud VM의 하드웨어 및 소프트웨어 사양을 설명합니다.

---

## 하드웨어 사양

### CPU

- **모델**: Intel(R) Xeon(R) CPU @ 2.20GHz
- **코어 수**: 2 physical cores, 4 logical processors (Hyper-Threading)
- **CPU Family**: 6
- **Model**: 85
- **Stepping**: 7
- **BogoMIPS**: 4400.30
- **캐시 크기**: 39424 KB (L3: 38.5 MiB)
- **아키텍처**: x86_64

**CPU 기능**:
- AVX, AVX2, AVX512F, AVX512DQ, AVX512BW, AVX512VL, AVX512CD, AVX512_VNNI
- SSE, SSE2, SSE4.1, SSE4.2
- AES-NI, RDRAND
- Virtualization (KVM)

**가상화**:
- Hypervisor: KVM
- Virtualization Type: full

### 메모리 (RAM)

- **총 메모리**: 16 GB (16,374,260 KB)
- **사용 가능**: ~13 GB
- **Swap**: 없음 (0 GB)

**메모리 상세**:
```
Total:     16 GB
Used:      ~2.4 GB
Free:      ~524 MB
Buff/Cache: ~12 GB
Available: ~12 GB
```

### 스토리지

**디스크 구성**:

| 파일시스템 | 크기 | 사용됨 | 가용 | 사용률 | 마운트 위치 |
|-----------|------|--------|------|--------|------------|
| /dev/root | 97G | 39G | 58G | 41% | / |
| /dev/nvme0n1p15 | 105M | 6.1M | 99M | 6% | /boot/efi |
| tmpfs | 7.9G | 0 | 7.9G | 0% | /dev/shm |
| tmpfs | 3.2G | 2.6M | 3.2G | 1% | /run |

**스토리지 타입**: NVMe SSD

### GPU

- **모델**: NVIDIA L4
- **드라이버 버전**: 580.95.05
- **CUDA 버전**: 13.0
- **메모리**: 23,034 MiB
- **전력**: 72W (TDP)
- **현재 온도**: 43°C
- **현재 전력 사용량**: 16W
- **Persistence Mode**: On
- **Compute Mode**: Default

---

## 소프트웨어 사양

### 운영체제

- **배포판**: Ubuntu 22.04.5 LTS
- **코드명**: Jammy Jellyfish
- **커널**: Linux (정확한 버전은 `uname -r`로 확인 가능)

### Python 환경

- **Python 버전**: 3.10.19
- **Conda 환경**: py310_openai (base)
- **패키지 관리자**: pip 25.2, setuptools 80.9.0

### 주요 설치 라이브러리

#### AI/ML 프레임워크

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| torch | 2.9.0 | PyTorch 딥러닝 프레임워크 |
| torchvision | 0.24.0 | 컴퓨터 비전 라이브러리 |
| transformers | 4.57.1 | Hugging Face 트랜스포머 모델 |
| accelerate | 1.11.0 | 분산 학습 가속화 |
| peft | 0.17.1 | Parameter-Efficient Fine-Tuning |
| datasets | 4.3.0 | 데이터셋 관리 |
| tokenizers | 0.22.1 | 토크나이저 |
| safetensors | 0.6.2 | 안전한 텐서 저장 |

#### LangChain 및 관련 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| langchain | 0.2.16 | LLM 애플리케이션 프레임워크 |
| langchain-community | 0.2.16 | 커뮤니티 통합 |
| langchain-core | 0.2.38 | 핵심 기능 |
| langchain-openai | 0.1.20 | OpenAI 통합 |
| langchain-text-splitters | 0.2.4 | 텍스트 분할 |
| langgraph | 0.0.51 | 그래프 기반 워크플로우 |
| langsmith | 0.1.147 | LangChain 모니터링 |

#### OpenAI 및 Google AI

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| openai | 1.109.1 | OpenAI API 클라이언트 |
| tiktoken | 0.9.0 | OpenAI 토크나이저 |
| google-genai | 1.46.0 | Google AI API |
| google-generativeai | 0.8.5 | Google Generative AI |

#### 벡터 데이터베이스 및 검색

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| faiss-cpu | 1.12.0 | Facebook AI Similarity Search |
| simsimd | 6.5.3 | SIMD 기반 유사도 계산 |

#### 데이터 처리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| pandas | 2.3.3 | 데이터 분석 |
| numpy | 1.26.4 | 수치 연산 |
| pyarrow | 21.0.0 | Apache Arrow |
| scikit-learn | 1.5.2 | 머신러닝 |
| scipy | 1.14.1 | 과학 연산 |

#### 머신러닝 - Boosting

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| lightgbm | 4.6.0 | Gradient Boosting |
| xgboost | 3.1.1 | XGBoost |

#### 이미지 처리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| opencv-python-headless | 4.11.0.86 | OpenCV (GUI 없음) |
| albumentations | 2.0.8 | 이미지 증강 |
| albucore | 0.0.24 | Albumentations 핵심 |
| pillow | 11.3.0 | 이미지 처리 |

#### 문서 처리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| pymupdf | 1.26.6 | PDF 처리 (PyMuPDF/fitz) |
| pymupdf4llm | 0.2.0 | LLM용 PDF 파싱 |
| pypdf | 6.2.0 | PDF 조작 |

#### 웹 애플리케이션

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| streamlit | 1.40.1 | 웹 애플리케이션 프레임워크 |
| fastapi | (requirements.txt 확인 필요) | API 서버 |

#### Jupyter 환경

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| jupyter | 1.1.1 | Jupyter 메타 패키지 |
| jupyterlab | 4.4.7 | JupyterLab IDE |
| ipykernel | 6.30.1 | IPython 커널 |
| ipywidgets | 8.1.5 | 인터랙티브 위젯 |
| notebook | 7.4.5 | Jupyter Notebook |

#### 시각화

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| matplotlib | 3.9.4 | 데이터 시각화 |
| seaborn | 0.13.2 | 통계 시각화 |
| altair | 5.5.0 | 선언적 시각화 |

#### 유틸리티

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| python-dotenv | 1.1.1 | 환경 변수 관리 |
| pydantic | 2.12.3 | 데이터 검증 |
| pydantic-settings | 2.12.0 | 설정 관리 |
| tqdm | 4.67.1 | 진행률 표시 |
| rich | 13.9.4 | 터미널 포매팅 |
| wandb | 0.22.3 | 실험 추적 |
| sentry-sdk | 2.43.0 | 에러 추적 |

#### 네트워크 및 HTTP

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| httpx | 0.28.1 | 비동기 HTTP 클라이언트 |
| aiohttp | 3.13.1 | 비동기 HTTP |
| requests | 2.32.5 | HTTP 라이브러리 |
| websockets | 15.0.1 | WebSocket 통신 |

#### 데이터베이스

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| sqlalchemy | 2.0.44 | SQL 툴킷 및 ORM |

---

## 네트워크 및 접속 정보

- **사용자 계정**: spai0433
- **호스트명**: codeit-ai-g2b-search
- **작업 디렉토리**: ~/DATA/Dev/Work/test_model

---

## 보안 취약점

CPU는 다음과 같은 알려진 취약점을 가지고 있습니다:
- Spectre v1, v2
- Meltdown (완화됨)
- MMIO Stale Data
- TAA (TSX Asynchronous Abort)
- Retbleed (Enhanced IBRS로 완화)

대부분의 취약점은 커널 수준에서 완화 조치가 적용되어 있습니다.

---

## 참고 사항

1. **환경 재현**: 전체 Python 환경은 `environment.yml` 파일을 통해 재현 가능합니다:
   ```bash
   conda env create -f environment.yml
   conda activate py310_openai
   ```

2. **GPU 모니터링**: `nvidia-smi` 명령으로 GPU 사용 현황을 실시간으로 확인할 수 있습니다.

3. **Swap 미설정**: 현재 Swap 메모리가 설정되어 있지 않습니다. 메모리 부족 시 프로세스가 강제 종료될 수 있으니 주의가 필요합니다.

4. **디스크 사용률**: 루트 파일시스템이 41% 사용 중이며, 약 58GB의 여유 공간이 있습니다.

---

**문서 작성일**: 2025-11-15
**VM 스냅샷 기준**: 2025-11-15 07:40 (KST)
