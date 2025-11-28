#!/bin/bash
# Ubuntu 22.04 서버 환경 설치 스크립트
# Python 3.10 + py310_openai 환경

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "Ubuntu 22.04 서버 환경 설치 시작"
echo "=========================================="

# 1. 시스템 업데이트 및 필수 패키지 설치
echo ""
echo "[1/6] 시스템 업데이트 및 필수 패키지 설치..."
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
    libgdk-pixbuf2.0-dev

# 3. Java 설치 (jpype1, pyhwp에 필요)
echo ""
echo "[3/6] Java 설치..."
if ! command -v java &> /dev/null; then
    sudo apt-get install -y default-jdk
    echo "Java 설치 완료: $(java -version 2>&1 | head -n 1)"
else
    echo "Java가 이미 설치되어 있습니다: $(java -version 2>&1 | head -n 1)"
fi

# 4. wkhtmltopdf 설치 (pdfkit에 필요)
echo ""
echo "[4/6] wkhtmltopdf 설치..."
if ! command -v wkhtmltopdf &> /dev/null; then
    sudo apt-get install -y wkhtmltopdf
    echo "wkhtmltopdf 설치 완료"
else
    echo "wkhtmltopdf가 이미 설치되어 있습니다"
fi

# 5. Pandoc 설치 (pypandoc에 필요)
echo ""
echo "[5/6] Pandoc 설치..."
if ! command -v pandoc &> /dev/null; then
    sudo apt-get install -y pandoc
    echo "Pandoc 설치 완료: $(pandoc --version | head -n 1)"
else
    echo "Pandoc이 이미 설치되어 있습니다: $(pandoc --version | head -n 1)"
fi

# 6. Conda 환경 생성 및 패키지 설치
echo ""
echo "[6/6] Conda 환경 생성 및 패키지 설치..."

# Rust 경로를 현재 세션에 추가
export PATH="$HOME/.cargo/bin:$PATH"

# Conda가 설치되어 있는지 확인
if ! command -v conda &> /dev/null; then
    echo ""
    echo "WARNING: Conda가 설치되어 있지 않습니다."
    echo "다음 중 하나를 선택하세요:"
    echo "  1) Anaconda/Miniconda 설치 후 다시 실행"
    echo "  2) pip로 최소 환경 설치: pip install -r requirements-minimal.txt"
    echo "  3) pip로 전체 환경 설치: pip install -r requirements.txt"
    exit 1
fi

# Conda 환경이 이미 존재하는지 확인
if conda env list | grep -q "^py310_openai "; then
    echo "py310_openai 환경이 이미 존재합니다. 업데이트를 진행합니다..."
    conda env update -f environment.yml --prune
else
    echo "py310_openai 환경을 새로 생성합니다..."
    conda env create -f environment.yml
fi

echo ""
echo "=========================================="
echo "설치가 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 명령어로 환경을 활성화하세요:"
echo "  conda activate py310_openai"
echo ""
echo "Rust 경로를 영구적으로 추가하려면 다음을 실행하세요:"
echo "  echo 'source \$HOME/.cargo/env' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
