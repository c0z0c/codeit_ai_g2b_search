# Ubuntu 22.04 서버 설치 가이드

이 가이드는 Ubuntu 22.04 서버에서 `py310_openai` Conda 환경을 설치하는 방법을 설명합니다.

## 사전 요구사항

- Ubuntu 22.04 LTS
- Conda (Anaconda 또는 Miniconda)
- sudo 권한

## 자동 설치 (권장)

```bash
# 저장소 클론 또는 파일 업로드 후
cd /path/to/codeit_ai_g2b_search

# 실행 권한 부여
chmod +x install_ubuntu.sh

# 설치 스크립트 실행
./install_ubuntu.sh

# 환경 활성화
conda activate py310_openai
```

## 수동 설치

### 1. 시스템 패키지 설치

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    python3-dev \
    pkg-config \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    default-jdk \
    wkhtmltopdf \
    pandoc
```

### 2. Rust 설치

`libhwp`, `cryptography` 등의 패키지는 Rust 컴파일러가 필요합니다.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Rust 버전 확인
rustc --version
```

Rust 경로를 영구적으로 추가:

```bash
echo 'source $HOME/.cargo/env' >> ~/.bashrc
source ~/.bashrc
```

### 3. Conda 환경 생성

```bash
# 새로운 환경 생성
conda env create -f environment.yml

# 또는 기존 환경 업데이트
conda env update -f environment.yml --prune
```

### 4. 환경 활성화

```bash
conda activate py310_openai
```

### 5. 설치 확인

```bash
python --version  # Python 3.10.19 확인
python -c "import torch; print(torch.__version__)"
python -c "import langchain; print(langchain.__version__)"
python -c "import streamlit; print(streamlit.__version__)"
```

## 문제 해결

### Rust 관련 오류

```
error: Cargo, the Rust package manager, is not installed
```

**해결방법:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### Java 관련 오류 (jpype1, pyhwp)

```
error: Java is not installed
```

**해결방법:**
```bash
sudo apt-get install -y default-jdk
```

### wkhtmltopdf 관련 오류 (pdfkit)

```
error: wkhtmltopdf is not installed
```

**해결방법:**
```bash
sudo apt-get install -y wkhtmltopdf
```

### libhwp 설치 실패

`libhwp`가 설치되지 않는 경우, 이 패키지 없이도 대부분의 기능은 작동합니다. HWP 파일 처리가 필요한 경우에만 필수입니다.

```bash
# libhwp를 제외하고 설치
pip install -r requirements.txt --no-deps
# 또는 requirements.txt에서 libhwp 라인 제거
```

### 메모리 부족 오류

대용량 패키지(torch, transformers 등) 설치 시 메모리가 부족할 수 있습니다.

**해결방법:**
```bash
# 패키지를 개별적으로 설치
conda activate py310_openai
pip install torch torchvision --no-cache-dir
pip install transformers --no-cache-dir
pip install -r requirements.txt --no-cache-dir
```

## 선택적 패키지

프로젝트 요구사항에 따라 일부 패키지는 선택적으로 설치할 수 있습니다:

### HWP 처리가 필요없는 경우
- `helper-hwp`
- `libhwp`
- `pyhwp`
- `hwp-extract`
- `jpype1`

### 웹 스크래핑이 필요없는 경우
- `selenium`
- `webdriver-manager`
- `pyautogui`
- 관련 의존성들

### 딥러닝 모델 학습이 필요없는 경우
- `torch`
- `torchvision`
- `accelerate`
- `peft`

이러한 패키지들은 `requirements.txt`에서 주석 처리하거나 제거할 수 있습니다.

## 추가 설정

### Streamlit 포트 설정

```bash
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
port = 8501
address = "0.0.0.0"
headless = true

[browser]
gatherUsageStats = false
EOF
```

### 환경 변수 설정

프로젝트 디렉토리에 `.env` 파일을 생성하여 API 키 등을 설정:

```bash
OPENAI_API_KEY=your_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## 참고사항

- Python 버전: 3.10.19
- Conda 환경명: py310_openai
- 주요 라이브러리:
  - LangChain 0.2.16
  - OpenAI 1.109.1
  - Streamlit 1.40.1
  - PyTorch 2.9.0
  - Transformers 4.57.1
