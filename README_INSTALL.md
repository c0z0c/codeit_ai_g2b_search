# 설치 가이드 - 빠른 시작

Ubuntu 22.04 서버에서 프로젝트를 실행하기 위한 설치 가이드입니다.

## 🚀 빠른 설치 (권장)

Streamlit 앱을 빠르게 실행하려면 **최소 환경 설치**를 권장합니다.

```bash
# 서버에서 실행
cd /home/spai0433/work/codeit_ai_g2b_search

# 최소 환경 설치 (5-10분)
chmod +x install_minimal.sh
./install_minimal.sh

# 앱 실행
streamlit run app.py
```

## 📋 설치 옵션 비교

| 옵션 | 설치 시간 | 용량 | 사용 사례 | 스크립트 |
|------|----------|------|----------|---------|
| **최소 환경** | 5-10분 | 2-3GB | Streamlit 앱만 실행 | `install_minimal.sh` |
| **pip 전체** | 30-60분 | 10-15GB | 모든 기능 (Conda 없이) | `install_pip_only.sh` |
| **Conda 전체** | 30-60분 | 10-15GB | 개발 환경 | `install_ubuntu.sh` |

## 💡 각 설치 옵션 상세

### 1️⃣ 최소 환경 (권장 - 빠른 시작)

**포함된 기능:**
- ✅ Streamlit 웹 UI
- ✅ LangChain + OpenAI
- ✅ PDF 문서 처리 (PyMuPDF)
- ✅ FAISS 벡터 검색
- ✅ 기본 데이터 처리

**제외된 기능:**
- ❌ HWP 파일 처리
- ❌ PyTorch/Transformers
- ❌ 웹 스크래핑
- ❌ 고급 ML 모델

**설치 방법:**
```bash
chmod +x install_minimal.sh
./install_minimal.sh
streamlit run app.py
```

**필요한 환경 변수:** (`.env` 파일 생성)
```bash
OPENAI_API_KEY=your_api_key_here
```

---

### 2️⃣ pip 전용 전체 환경

Conda 없이 Python 3.10 가상환경에서 모든 기능을 사용합니다.

**사전 요구사항:**
- Python 3.10
- Rust (자동 설치됨)
- Java JDK (자동 설치됨)

**설치 방법:**
```bash
# 가상환경 생성 (선택사항)
python3.10 -m venv venv
source venv/bin/activate

# 설치
chmod +x install_pip_only.sh
./install_pip_only.sh

# 앱 실행
streamlit run app.py
```

---

### 3️⃣ Conda 전체 환경

개발 및 데이터 분석을 위한 완전한 환경입니다.

**사전 요구사항:**
- Anaconda 또는 Miniconda

**설치 방법:**
```bash
chmod +x install_ubuntu.sh
./install_ubuntu.sh

conda activate py310_openai
streamlit run app.py
```

---

## 🔧 수동 설치 (문제 해결용)

### 최소 패키지만 설치

```bash
# 필수 시스템 패키지
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libssl-dev

# Python 패키지
pip install -r requirements-minimal.txt
```

### 누락된 패키지 개별 설치

```bash
# dotenv 오류 해결
pip install python-dotenv==1.1.1

# PyYAML 오류 해결
pip install pyyaml==6.0.2

# Streamlit 오류 해결
pip install streamlit==1.40.1
```

---

## ⚠️ 문제 해결

### ModuleNotFoundError: No module named 'dotenv'

```bash
pip install python-dotenv pyyaml
```

### Rust 관련 오류

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### HWP 파일 처리 오류

HWP 파일 처리가 필요하면 전체 환경을 설치하세요:
```bash
./install_pip_only.sh  # 또는 install_ubuntu.sh
```

---

## 📦 설치 파일 목록

| 파일 | 용도 |
|------|------|
| `install_minimal.sh` | 최소 환경 빠른 설치 |
| `install_pip_only.sh` | pip 전용 전체 설치 |
| `install_ubuntu.sh` | Conda 전체 설치 |
| `requirements-minimal.txt` | 최소 패키지 목록 |
| `requirements.txt` | 전체 패키지 목록 |
| `environment.yml` | Conda 환경 정의 |

---

## 🎯 다음 단계

설치 후:

1. **환경 변수 설정**
   ```bash
   cp .env.example .env  # 템플릿이 있는 경우
   # 또는
   nano .env
   ```

   `.env` 파일 내용:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. **앱 실행**
   ```bash
   streamlit run app.py
   ```

3. **브라우저에서 접속**
   ```
   http://localhost:8501
   ```

---

## 📚 추가 문서

- 상세 설치 가이드: [INSTALL_UBUNTU.md](INSTALL_UBUNTU.md)
- 환경 검토 결과: [ENVIRONMENT_REVIEW.md](ENVIRONMENT_REVIEW.md)
- 프로젝트 문서: [docs/](docs/)
