# environment.yml 검토 결과 (Ubuntu 22.04 서버용)

## 검토 요약

environment.yml 파일을 Ubuntu 22.04 서버 환경에 맞게 검토하고 개선했습니다.

## 주요 발견 사항

### 1. **Rust 컴파일 필요 패키지** ⚠️

다음 패키지들은 설치 시 Rust 컴파일러가 필요합니다:

```yaml
# Rust 필요 (pip 섹션)
- cryptography==46.0.3      # 보안/암호화
- helper-hwp==0.5.4            # HWP 파일 처리
- pydantic-core==2.41.4    # 데이터 검증 (Pydantic 의존성)
```

**해결 방법:**
```bash
# 설치 전에 Rust 설치 필수
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### 2. **Java 필요 패키지** ☕

HWP 파일 처리를 위해 Java가 필요합니다:

```yaml
- jpype1==1.6.0           # Python-Java 브리지
- pyhwp==0.1b15          # HWP 파일 파서
```

**해결 방법:**
```bash
sudo apt-get install -y default-jdk
```

### 3. **시스템 라이브러리 필요 패키지** 📦

다음 패키지들은 시스템 레벨 라이브러리가 필요합니다:

#### 이미지 처리 관련
```bash
sudo apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libwebp-dev
```

필요한 Python 패키지:
- `pillow==11.3.0`
- `opencv-python-headless==4.11.0.86`
- `matplotlib==3.9.4`

#### XML/HTML 처리 관련
```bash
sudo apt-get install -y \
    libxml2-dev \
    libxslt1-dev
```

필요한 Python 패키지:
- `lxml==6.0.2`
- `weasyprint==66.0`

#### PDF 처리 관련
```bash
sudo apt-get install -y \
    wkhtmltopdf \
    pandoc
```

필요한 Python 패키지:
- `pdfkit==1.0.0`
- `pypandoc==1.16.2`

#### Cairo 그래픽 (WeasyPrint용)
```bash
sudo apt-get install -y \
    libcairo2-dev \
    libpango1.0-dev \
    libgdk-pixbuf2.0-dev \
    libharfbuzz-dev \
    libfribidi-dev
```

필요한 Python 패키지:
- `weasyprint==66.0`

### 4. **GUI 자동화 패키지** 🖱️ (헤드리스 서버 주의)

다음 패키지들은 GUI 환경이 필요하므로 헤드리스 서버에서 문제가 될 수 있습니다:

```yaml
# GUI automation (some packages may not work on headless servers)
- pyautogui==0.9.54
- pygetwindow==0.0.9
- pymsgbox==2.0.1
- pyperclip==1.11.0
- pyrect==0.2.0
- pyscreeze==1.0.1
- pytweening==1.2.0
- mouseinfo==0.1.3
```

**참고:** 이러한 패키지들은 설치는 되지만 실제로는 작동하지 않을 수 있습니다. 필요하지 않다면 주석 처리하는 것을 권장합니다.

**웹 스크래핑을 위한 대안:**
- `selenium==4.38.0` + `pyvirtualdisplay==3.0` 조합 사용 (가상 디스플레이)

### 5. **Intel MKL 라이브러리** 🔢

Intel 최적화 수학 라이브러리가 포함되어 있습니다:

```yaml
# Math and computation libraries
- blas=1.0
- intel-openmp=2025.0.0
- mkl=2025.0.0
- mkl-service=2.5.2
- mkl_fft=1.3.11
- mkl_random=1.2.8
- tbb=2022.0.0
- tbb-devel=2022.0.0
```

**장점:** NumPy, SciPy, scikit-learn 등의 성능 향상
**단점:** 설치 용량이 크고 시간이 오래 걸림

**대안 (경량화가 필요한 경우):**
```yaml
- nomkl  # MKL 없이 설치
- numpy
- scipy
```

### 6. **conda-forge 채널 추가** ✅

```yaml
channels:
  - defaults
  - conda-forge
```

Ubuntu 22.04에서 더 나은 호환성을 위해 `conda-forge` 채널이 추가되었습니다.

## 개선 사항

### 적용된 변경사항

1. **HWP 처리 패키지에 주석 추가**
   - Rust와 Java 필요성 명시
   - 설치 명령어 포함

2. **GUI 자동화 패키지 분리**
   - 헤드리스 서버에서 작동하지 않을 수 있음을 경고
   - 별도 섹션으로 구분

3. **cryptography 패키지 주석 추가**
   - Rust 필요성 명시

## 권장 설치 순서

### 1단계: 시스템 패키지 설치
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

### 2단계: Rust 설치
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### 3단계: Conda 환경 생성
```bash
conda env create -f environment.yml
```

### 4단계: 환경 활성화
```bash
conda activate py310_openai
```

## 선택적 패키지 제거

프로젝트에서 필요하지 않은 기능이 있다면 다음 패키지들을 제거할 수 있습니다:

### HWP 파일 처리 불필요 시
```yaml
# 제거 가능:
- helper-hwp==0.5.1
- libhwp==0.2.0
- pyhwp==0.1b15
- hwp-extract==0.1.0
- jpype1==1.6.0
```

### GUI 자동화 불필요 시
```yaml
# 제거 가능:
- pyautogui==0.9.54
- pygetwindow==0.0.9
- pymsgbox==2.0.1
- pyperclip==1.11.0
- pyrect==0.2.0
- pyscreeze==1.0.1
- pytweening==1.2.0
- mouseinfo==0.1.3
```

### 딥러닝 모델 학습 불필요 시 (추론만 사용)
```yaml
# 제거 가능:
- accelerate==1.11.0
- peft==0.17.1
- wandb==0.22.3
```

### 웹 스크래핑 불필요 시
```yaml
# 제거 가능:
- selenium==4.38.0
- webdriver-manager==4.0.2
- pyvirtualdisplay==3.0
```

## 설치 예상 시간 및 용량

- **설치 시간:** 약 30-60분 (네트워크 속도 및 서버 사양에 따라 다름)
- **디스크 용량:** 약 10-15GB
  - Conda 기본 패키지: ~3GB
  - PyTorch: ~2GB
  - Transformers 모델: ~1GB
  - 기타 의존성: ~4-9GB

## 문제 해결

### Rust 관련 오류
```
error: Cargo, the Rust package manager, is not installed
```
→ 2단계(Rust 설치) 먼저 수행

### Java 관련 오류
```
JPypeException: Unable to find Java Runtime Environment
```
→ `sudo apt-get install -y default-jdk` 실행

### 메모리 부족 오류
대용량 패키지 설치 시 메모리 부족 발생 가능
```bash
# --no-cache-dir 옵션 사용
pip install torch --no-cache-dir
pip install transformers --no-cache-dir
```

### MKL 라이브러리 충돌
```bash
# MKL 관련 충돌 시 환경 변수 설정
export MKL_THREADING_LAYER=GNU
```

## 참고 파일

- [INSTALL_UBUNTU.md](INSTALL_UBUNTU.md) - 상세 설치 가이드
- [install_ubuntu.sh](install_ubuntu.sh) - 자동 설치 스크립트
- [requirements.txt](requirements.txt) - pip 전용 설치 파일
