---
layout: default
title: "GCP 환경 설정 가이드"
description: "Google Cloud VM Ubuntu 환경 설정 및 JupyterHub 설치"
date: 2025-11-15
author: "김명환"
---

# GCP 환경 설정 가이드

> Google Cloud VM에서 VSFTPD를 통한 파일 전송 환경과 JupyterHub 개발 환경을 구축하는 완벽 가이드

## 빠른 시작 요약

### FTP 환경 구축 (3단계)
1. **VM에서**: `sudo apt install vsftpd -y` → `/etc/vsftpd.conf` 설정
2. **로컬 PC에서**: GCP 방화벽 규칙 생성 (`gcloud compute firewall-rules create allow-ftp ...`)
3. **Windows에서**: IPDisk로 Z 드라이브 연결

### Colab 로컬 런타임 연결 (3단계)
1. **VM에서**: Jupyter Server 설정 (`jupyter server --generate-config`) → 토큰 고정
2. **로컬 PC에서**: SSH 터널링 (`gcloud compute ssh ... --ssh-flag="-L 8888:localhost:8888"`)
3. **Colab에서**: `http://localhost:8888/?token=mysecrettoken1234` 연결

### 주요 명령어 치트시트
```bash
# 503 퍼미션 오류 해결
chmod u+w /home/계정명 && sudo systemctl restart vsftpd

# 방화벽 규칙 생성 (Windows PowerShell)
gcloud compute firewall-rules create allow-ftp --description="Allow FTP Control (21) and Passive Data Ports (30000-30009)" --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules="tcp:21,tcp:30000-30009" --source-ranges=0.0.0.0/0 --target-tags=ftp-server --project=sprint-ai-chunk2-03

# VM 태그 추가 (Windows PowerShell)
gcloud compute instances add-tags codeit-ai-g2b-search --tags=ftp-server --zone=us-central1-c --project=sprint-ai-chunk2-03

gcloud compute ssh spai0433@codeit-ai-g2b-search --project=sprint-ai-chunk2-03 --zone=us-central1-c

```

---

## 목차

### Part 1: 파일 전송 환경 구축
1. [VSFTPD 설치 및 설정](#1-vsftpd-설치-및-설정)
2. [GCP 방화벽 설정](#2-gcp-방화벽-설정)
3. [Windows PC에서 FTP 연결](#3-windows-pc에서-ftp-연결)

### Part 2: JupyterHub 설치
4. [환경 준비](#4-환경-준비)
5. [Miniconda 설치](#5-miniconda-설치)
6. [JupyterHub 설치](#6-jupyterhub-설치)
7. [설정 파일 작성](#7-설정-파일-작성)
8. [사용자 계정 생성](#8-사용자-계정-생성)
9. [Configurable HTTP Proxy 설치](#9-configurable-http-proxy-설치)
10. [시스템 서비스 등록](#10-시스템-서비스-등록)
11. [JupyterHub 방화벽 설정](#11-jupyterhub-방화벽-설정)
12. [접속 및 테스트](#12-접속-및-테스트)
13. [관리 명령어](#13-관리-명령어)
14. [Jupyter 커널 등록](#14-jupyter-커널-등록-선택사항)

### Part 3: Colab 로컬 런타임 연결
15. [Colab과 GCP VM 연결](#15-colab과-gcp-vm-연결)

---

# Part 1: 파일 전송 환경 구축

## 1. VSFTPD 설치 및 설정

### VSFTPD 설치

```bash
# GCP VM에 접속
gcloud compute ssh spai0433@codeit-ai-g2b-search --project=sprint-ai-chunk2-03 --zone=us-central1-c

# 패키지 업데이트
sudo apt update

# VSFTPD 설치
sudo apt install vsftpd -y

# 설치 확인
vsftpd -v
```

### VSFTPD 설정 파일 편집

```bash
# 기존 설정 파일 백업
sudo cp /etc/vsftpd.conf /etc/vsftpd.conf.backup

# 설정 파일 편집
sudo vi /etc/vsftpd.conf
```

**다음 내용으로 수정 또는 추가:**

```ini
# 로컬 사용자 로그인 허용
local_enable=YES

# 파일 쓰기(업로드) 허용 설정 (치명적 오류 해결)
write_enable=YES

# 로컬 사용자를 홈 디렉토리에 격리
chroot_local_user=YES

# Ubuntu 20.04/VSFTPD 3.x에서 필수 (쓰기 가능한 chroot 허용)
allow_writeable_chroot=YES

# Passive 모드(Passive Mode) 포트 범위 설정
pasv_min_port=30000
pasv_max_port=30009

# VM의 실제 외부 IP 주소로 변경 (중요!)
# 🚨 VM의 실제 외부 IP 주소로 정확히 설정해야 합니다.
pasv_address=34.9.92.3

# 익명 사용자 비활성화 (보안)
anonymous_enable=NO

# 로컬 사용자 기본 umask
local_umask=022

# 로그 활성화
xferlog_enable=YES
xferlog_file=/var/log/vsftpd.log
```

**저장:** `ESC` → `:wq` → `Enter`

**주의사항:**
- `pasv_address`는 반드시 VM의 실제 외부 IP로 변경하세요
- 외부 IP는 GCP 콘솔에서 확인하거나 다음 명령어로 확인:
  ```bash
  gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
  ```

### VSFTPD 서비스 재시작

```bash
# 서비스 재시작
sudo systemctl restart vsftpd

# 서비스 상태 확인
sudo systemctl status vsftpd

# 부팅 시 자동 시작 설정
sudo systemctl enable vsftpd
```

### 포트 확인

```bash
# FTP 포트 리스닝 확인
sudo netstat -tulpn | grep vsftpd

# 출력 예시:
# tcp   0   0 0.0.0.0:21   0.0.0.0:*   LISTEN   [PID]/vsftpd
```

---

## 2. GCP 방화벽 설정

### 방화벽 규칙 생성

**주의**: 아래 명령어를 로컬 PC의 PowerShell 또는 CMD에서 실행하세요 (VM 내부가 아님)

**Windows PowerShell/CMD 한 줄 명령어:**
```powershell
gcloud compute firewall-rules create allow-ftp --description="Allow FTP Control (21) and Passive Data Ports (30000-30009)" --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules="tcp:21,tcp:30000-30009" --source-ranges=0.0.0.0/0 --target-tags=ftp-server --project=sprint-ai-chunk2-03
```

**테스트 명령어 (PowerShell):**
```powershell
Test-NetConnection -ComputerName 34.9.92.3 -Port 21
Test-NetConnection -ComputerName 34.9.92.3 -Port 30000
```

### VM에 네트워크 태그 추가

**Windows PowerShell/CMD 한 줄 명령어:**
```powershell
gcloud compute instances add-tags codeit-ai-g2b-search --tags=ftp-server --zone=us-central1-c --project=sprint-ai-chunk2-03
```

### 방화벽 규칙 확인

```bash
# 방화벽 규칙 상세 정보 확인
gcloud compute firewall-rules describe allow-ftp --project=sprint-ai-chunk2-03

# 모든 방화벽 규칙 목록 확인
gcloud compute firewall-rules list --project=sprint-ai-chunk2-03

# VM 태그 확인
gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(tags.items)"


gcloud compute instances add-tags codeit-ai-g2b-search --zone=us-central1-c --project=sprint-ai-chunk2-03 --tags=streamlit-server
gcloud compute firewall-rules create allow-streamlit-8501-new --project=sprint-ai-chunk2-03 --network=default --action=ALLOW --rules=tcp:8501 --source-ranges=0.0.0.0/0 --target-tags=streamlit-server --description="Allow Streamlit traffic on TCP port 8501 using new tag"

gcloud compute firewall-rules create allow-http-streamlit-80 --project=sprint-ai-chunk2-03 --network=default --action=ALLOW --rules=tcp:80 --source-ranges=0.0.0.0/0 --target-tags=streamlit-server --description="Allow HTTP traffic on TCP port 80 for Streamlit server"

sudo -E /opt/miniconda3/envs/py310_openai/bin/python -m streamlit run app.py --server.port 80

```

---

## 3. Windows PC에서 FTP 연결

### 포트 연결 테스트 (PowerShell)

```powershell
# FTP Control 포트(21) 테스트
Test-NetConnection -ComputerName 34.9.92.3 -Port 21

# Passive 데이터 포트 테스트
Test-NetConnection -ComputerName 34.9.92.3 -Port 30000
```

**정상 출력 예시:**
```
ComputerName     : 34.9.92.3
RemoteAddress    : 34.9.92.3
RemotePort       : 21
TcpTestSucceeded : True
```

### IPDisk를 통한 Z 드라이브 연결

#### IPDisk 다운로드 및 설치
1. IPDisk 프로그램 다운로드: [IPDisk 공식 사이트](http://www.ipdisk.co.kr)
2. 설치 파일 실행 및 설치 진행

#### FTP 연결 설정
1. **IPDisk 실행**
2. **파일 > 새 연결** 클릭
3. **연결 정보 입력:**
   - **프로토콜**: FTP
   - **서버 주소**: `34.9.92.3` (VM 외부 IP)
   - **포트**: `21`
   - **사용자 이름**: `spai0433` (본인의 리눅스 계정)
   - **비밀번호**: 리눅스 계정 비밀번호
   - **드라이브 문자**: `Z:`
4. **연결** 클릭

#### 연결 확인
- Windows 탐색기에서 `Z:` 드라이브 확인
- VM의 홈 디렉토리(`/home/spai0433`) 내용이 표시됨

#### 자동 연결 설정 (선택)
1. IPDisk 설정에서 **시작 시 자동 연결** 옵션 활성화
2. Windows 부팅 시 자동으로 Z 드라이브 연결됨

### Windows 네트워크 드라이브 연결 (대안)

IPDisk 대신 Windows 기본 기능 사용:

```
1. 파일 탐색기 열기
2. '내 PC' 우클릭 > '네트워크 드라이브 연결'
3. 드라이브 문자: Z
4. 폴더: ftp://34.9.92.3
5. '다른 자격 증명을 사용하여 연결' 체크
6. '마침' 클릭
7. 사용자 이름과 비밀번호 입력
```

---

# Part 2: JupyterHub 설치

---

## 4. 환경 준비

### GCP VM 접속

```bash
# gcloud를 통한 VM 접속
gcloud compute ssh spai0433@codeit-ai-g2b-search --project=sprint-ai-chunk2-03 --zone=us-central1-c
```

### 시스템 업데이트

```bash
# 패키지 업데이트
sudo apt update
sudo apt upgrade -y

# 필수 도구 설치
sudo apt install -y git wget curl vim
```

---

## 5. Miniconda 설치

### 다운로드 및 설치

```bash
# Miniconda 다운로드
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 시스템 전체 설치
sudo bash miniconda.sh -b -p /opt/miniconda3

# 권한 설정
sudo chmod -R 755 /opt/miniconda3

# PATH 설정
echo 'export PATH="/opt/miniconda3/bin:$PATH"' | sudo tee -a /etc/profile
source /etc/profile

# 설치 확인
conda --version
```

---

## 6. JupyterHub 설치

### 가상 환경 생성 및 패키지 설치

```bash
# Python 3.10 가상 환경 생성
sudo /opt/miniconda3/bin/conda create -n jhub-env python=3.10 -y

# JupyterHub, JupyterLab, Notebook 설치
sudo /opt/miniconda3/bin/conda run -n jhub-env pip install jupyterhub jupyterlab notebook

# 버전 확인
sudo /opt/miniconda3/bin/conda run -n jhub-env jupyterhub --version
sudo /opt/miniconda3/bin/conda run -n jhub-env jupyter lab --version
```

---

## 7. 설정 파일 작성

### 설정 디렉토리 및 파일 생성

```bash
# 설정 디렉토리 생성
sudo mkdir -p /etc/jupyterhub

# 설정 파일 생성
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub --generate-config -f /etc/jupyterhub/jupyterhub_config.py

# 백업 생성
sudo cp /etc/jupyterhub/jupyterhub_config.py /etc/jupyterhub/jupyterhub_config.py.org
```

### 설정 파일 편집

```bash
# 설정 파일 열기
sudo vi /etc/jupyterhub/jupyterhub_config.py
```

**다음 내용으로 작성:**

```python
# JupyterHub 설정 파일
c = get_config()  #noqa

# 네트워크 설정 (모든 인터페이스에서 8000 포트로 접속)
c.JupyterHub.bind_url = "http://0.0.0.0:8000/"

# JupyterLab을 기본 인터페이스로 사용
c.Spawner.default_url = "/lab"

# 시스템 사용자 자동 생성 비활성화
c.LocalAuthenticator.create_system_users = False

# 허용된 사용자 목록 (자신의 사용자명으로 변경)
c.Authenticator.allowed_users = {
    "spai0409",
    "spai0427",
    "spai0433",
    "spai0438"
}

# 단일 사용자 노트북 서버 실행 명령
c.Spawner.cmd = ['/opt/miniconda3/envs/jhub-env/bin/jupyterhub-singleuser']
```

**저장:** `ESC` → `:wq` → `Enter`

---

## 8. 사용자 계정 생성

### 시스템 사용자 추가

```bash
# 사용자 추가 (각 사용자별로 실행)
sudo adduser spai0409
sudo adduser spai0427
sudo adduser spai0433
sudo adduser spai0438

# 비밀번호 입력 후 나머지는 Enter로 건너뛰기
```

### 일괄 생성 (선택)

```bash
# 여러 사용자 한번에 생성
for user in spai0409 spai0427 spai0433 spai0438; do
    sudo adduser --disabled-password --gecos "" $user
    echo "$user:초기비밀번호" | sudo chpasswd
done

# 사용자 확인
cat /etc/passwd | grep spai
```

---

## 9. Configurable HTTP Proxy 설치

### Node.js 및 Proxy 설치

```bash
# Node.js와 npm 설치
sudo apt install nodejs npm -y

# Configurable HTTP Proxy 설치
sudo npm install -g configurable-http-proxy

# 설치 확인
configurable-http-proxy --version
```

---

## 10. 시스템 서비스 등록

### systemd 서비스 파일 생성

```bash
# 서비스 파일 생성
sudo vi /etc/systemd/system/jupyterhub.service
```

**다음 내용 입력:**

```ini
[Unit]
Description=JupyterHub
After=network.target

[Service]
User=root
ExecStart=/opt/miniconda3/envs/jhub-env/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py
WorkingDirectory=/etc/jupyterhub
Restart=always
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
```

**저장:** `ESC` → `:wq` → `Enter`

### 서비스 등록 및 시작

```bash
# systemd 리로드
sudo systemctl daemon-reload

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable jupyterhub.service

# 서비스 시작
sudo systemctl start jupyterhub.service

# 서비스 상태 확인
sudo systemctl status jupyterhub.service
```

**정상 실행 시 출력:**
```
● jupyterhub.service - JupyterHub
     Loaded: loaded
     Active: active (running)
```

---

## 11. JupyterHub 방화벽 설정

### GCP 방화벽 규칙 생성

**Windows PowerShell/CMD 한 줄 명령어:**
```powershell
# 방화벽 규칙 생성 (8000 포트 개방)
gcloud compute firewall-rules create allow-jupyterhub --description="Allow JupyterHub on port 8000" --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:8000 --source-ranges=0.0.0.0/0 --target-tags=jupyterhub-server --project=sprint-ai-chunk2-03

# VM에 네트워크 태그 추가
gcloud compute instances add-tags codeit-ai-g2b-search --tags=jupyterhub-server --zone=us-central1-c --project=sprint-ai-chunk2-03
```

**테스트 명령어 (PowerShell):**
```powershell
Test-NetConnection -ComputerName 34.9.92.3 -Port 8000
```

### 포트 개방 확인

```bash
# VM 내부에서 포트 확인
sudo netstat -tulpn | grep 8000

# 출력 예시:
# tcp   0   0 0.0.0.0:8000   0.0.0.0:*   LISTEN   12345/python
```

---

## 12. 접속 및 테스트

### 웹 브라우저 접속

```
http://[VM_외부_IP]:8000
```

**예시:**
```
http://34.123.45.67:8000
```

### 로그인

1. **Username**: 시스템 사용자 이름 (예: `spai0433`)
2. **Password**: 해당 사용자의 비밀번호
3. **Sign in** 클릭

### 노트북 테스트

JupyterLab이 열리면 새 노트북 생성:

```python
# 테스트 코드
import sys
print(f"Python version: {sys.version}")

import os
print(f"User: {os.getenv('USER')}")
print(f"Home: {os.getenv('HOME')}")
```

---

## 14. Jupyter 커널 등록 (선택사항)

### 개별 사용자 환경을 위한 커널 등록

각 사용자가 자신만의 Python 환경을 사용하려면 개별 커널을 등록할 수 있습니다.

### Conda 환경 생성 및 커널 등록

```bash
# 1. Conda 환경 생성
conda create -n py310_openai python=3.10 -y

# 2. 환경 활성화
conda activate py310_openai

# 3. 필요한 패키지 설치
pip install -r requirements.txt

# 4. ipykernel 설치
conda install ipykernel -y

# 5. Jupyter 커널 등록
python -m ipykernel install --user --name py310_openai --display-name "Python 3.10 (OpenAI Env)"

# 6. 등록된 커널 확인
jupyter kernelspec list
```

### 출력 예시

```
Available kernels:
  py310_openai    /home/spai0433/.local/share/jupyter/kernels/py310_openai
  python3         /opt/miniconda3/envs/jhub-env/share/jupyter/kernels/python3
```

### 커널 관리 명령어

```bash
# 등록된 커널 목록 확인
jupyter kernelspec list

# 특정 커널 삭제
jupyter kernelspec uninstall py310_openai

# 커널 정보 확인
jupyter kernelspec list --json
```

### JupyterHub에서 커널 사용

1. JupyterHub에 로그인
2. 새 노트북 생성 시 **"Python 3.10 (OpenAI Env)"** 커널 선택
3. 또는 기존 노트북에서 **Kernel > Change Kernel** 메뉴로 커널 변경

### 여러 환경 예시

```bash
# 데이터 분석용 환경
conda create -n data_analysis python=3.10 pandas numpy matplotlib -y
conda activate data_analysis
conda install ipykernel -y
python -m ipykernel install --user --name data_analysis --display-name "Python 3.10 (Data Analysis)"

# 머신러닝 환경
conda create -n ml_env python=3.10 scikit-learn tensorflow -y
conda activate ml_env
conda install ipykernel -y
python -m ipykernel install --user --name ml_env --display-name "Python 3.10 (ML)"
```

---

## 13. 관리 명령어

### 서비스 관리

```bash
# 서비스 시작
sudo systemctl start jupyterhub.service

# 서비스 중지
sudo systemctl stop jupyterhub.service

# 서비스 재시작
sudo systemctl restart jupyterhub.service

# 서비스 상태 확인
sudo systemctl status jupyterhub.service
```

### 로그 확인

```bash
# 전체 로그 확인
sudo journalctl -u jupyterhub.service

# 최근 100줄 로그
sudo journalctl -u jupyterhub.service -n 100

# 실시간 로그 (tail -f)
sudo journalctl -u jupyterhub.service -f
```

### 사용자 추가

```bash
# 1. 시스템 사용자 생성
sudo adduser 새사용자명

# 2. 설정 파일 수정
sudo vi /etc/jupyterhub/jupyterhub_config.py
# allowed_users에 새 사용자 추가

# 3. 서비스 재시작
sudo systemctl restart jupyterhub.service
```

### 설정 변경 후 적용

```bash
# 설정 파일 수정
sudo vi /etc/jupyterhub/jupyterhub_config.py

# 서비스 재시작
sudo systemctl restart jupyterhub.service
```

---

# Part 3: Colab 로컬 런타임 연결

---

## 15. Colab과 GCP VM 연결

Google Colab에서 GCP VM의 Jupyter Server를 로컬 런타임으로 연결하여 VM의 고성능 자원(GPU, 대용량 메모리)을 활용할 수 있습니다.

### 15.1. VM Jupyter Server 설정 (Token 고정)

#### 설정 파일 생성

VM에 접속하여 Jupyter Server 설정 파일을 생성합니다.

```bash
# VM 터미널에서 실행
/opt/miniconda3/bin/jupyter server --generate-config
```

**생성 경로**: `/home/spai0433/.jupyter/jupyter_server_config.py`

#### 기존 Jupyter Server 중지

포트 충돌을 방지하기 위해 기존 프로세스를 중지합니다.

```bash
# 실행 중인 Jupyter Server 확인
ps ax | grep jupyter-server

# 출력 예시:
# 12345 pts/0    S      0:00 jupyter-server
# 12346 pts/0    S      0:00 /opt/miniconda3/bin/python -m jupyter-server

# PID를 사용하여 프로세스 종료
kill 12345
kill 12346

# 또는 모든 jupyter-server 프로세스 일괄 종료
pkill -f jupyter-server
```

#### 설정 파일 수정 (Token 및 Port 고정)

```bash
# 설정 파일 열기
nano /home/spai0433/.jupyter/jupyter_server_config.py
```

**다음 설정을 추가 또는 수정:**

```python
# 외부 접속 허용 (모든 IP에서 접속 가능)
c.ServerApp.ip = '*'

# 고정 포트 설정 (Colab 연결 시 사용)
c.ServerApp.port = 8888

# 비밀번호 인증 제거 (Token 방식 사용)
c.ServerApp.password = ''

# 고정 토큰 설정 (예시: mysecrettoken1234)
c.ServerApp.token = 'mysecrettoken1234'

# 브라우저 자동 실행 비활성화
c.ServerApp.open_browser = False

# 루트 디렉토리 설정 (선택)
# c.ServerApp.root_dir = '/home/spai0433'
```

**저장**: `Ctrl + O` → `Enter` → `Ctrl + X`

**주요 설정 항목 설명:**

| 설정 항목 | 설명 | 설정 값 |
|----------|------|---------|
| `c.ServerApp.ip` | 외부 접속 허용 | `'*'` |
| `c.ServerApp.port` | 고정 포트 설정 | `8888` |
| `c.ServerApp.password` | 비밀번호 인증 제거 | `''` |
| `c.ServerApp.token` | 고정 토큰 설정 | `'mysecrettoken1234'` |

**보안 주의사항:**
- 프로덕션 환경에서는 강력한 토큰 사용 권장
- 토큰은 최소 16자 이상의 무작위 문자열 사용 권장
- 토큰 생성 예시: `openssl rand -hex 32`

#### Jupyter Server 실행

```bash
# Jupyter Server 백그라운드 실행
jupyter server &

# 또는 nohup으로 실행 (터미널 종료 후에도 유지)
nohup jupyter server > jupyter.log 2>&1 &

# 실행 확인
ps ax | grep jupyter-server

# 포트 확인
sudo netstat -tulpn | grep 8888

# 출력 예시:
# tcp   0   0 0.0.0.0:8888   0.0.0.0:*   LISTEN   12345/python
```

**서버 종료 방법:**
```bash
# 프로세스 ID 확인 후 종료
ps ax | grep jupyter-server
kill [PID]

# 또는 일괄 종료
pkill -f jupyter-server
```

---

### 15.2. SSH 터널링 설정 (로컬 PC)

로컬 PC에서 GCP VM의 8888 포트를 로컬로 전달합니다.

#### Windows PowerShell 실행

```powershell
# SSH 터널링 설정
gcloud compute ssh spai0433@codeit-ai-g2b-search --project sprint-ai-chunk2-03 --ssh-flag="-L 8888:localhost:8888"
```

**명령어 설명:**
- `-L 8888:localhost:8888`: VM의 8888 포트를 로컬 8888 포트로 포워딩
- 이 창을 **닫지 말고** 유지해야 터널링이 활성 상태로 유지됩니다

#### 연결 확인

```powershell
# 로컬 포트 확인 (다른 PowerShell 창)
Test-NetConnection -ComputerName localhost -Port 8888
```

**정상 출력:**
```
ComputerName     : localhost
RemoteAddress    : ::1
RemotePort       : 8888
TcpTestSucceeded : True
```

---

### 15.3. Colab 로컬 런타임 연결

#### 연결 절차

1. **Google Colab 접속**
   - https://colab.research.google.com/ 접속
   - 새 노트북 생성 또는 기존 노트북 열기

2. **로컬 런타임 연결**
   - 우측 상단 **'연결'** 메뉴 클릭
   - **'로컬 런타임에 연결'** 선택

3. **백엔드 URL 입력**
   ```
   http://localhost:8888/?token=mysecrettoken1234
   ```

4. **연결 클릭**
   - **'연결'** 버튼 클릭
   - 연결 성공 시 우측 상단에 녹색 체크 표시

#### 연결 확인

Colab 노트북에서 다음 코드를 실행하여 VM 자원을 사용하는지 확인:

```python
# 시스템 정보 확인
import os
import platform
import socket

print(f"호스트명: {socket.gethostname()}")
print(f"플랫폼: {platform.platform()}")
print(f"Python 버전: {platform.python_version()}")
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"사용자: {os.getenv('USER')}")

# GPU 확인 (GPU VM인 경우)
try:
    import torch
    print(f"\nGPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("\nPyTorch가 설치되지 않았습니다.")
```

**예상 출력:**
```
호스트명: codeit-ai-g2b-search
플랫폼: Linux-5.15.0-1052-gcp-x86_64-with-glibc2.35
Python 버전: 3.10.x
현재 작업 디렉토리: /home/spai0433
사용자: spai0433
```

---

### 15.4. 문제 해결

#### 연결 실패: "Unable to connect to the runtime"

**원인 1**: SSH 터널링이 끊김
```powershell
# SSH 터널링 재실행
gcloud compute ssh spai0433@codeit-ai-g2b-search --project sprint-ai-chunk2-03 --ssh-flag="-L 8888:localhost:8888"
```

**원인 2**: Jupyter Server가 중지됨
```bash
# VM에서 Jupyter Server 재시작
jupyter server &
```

**원인 3**: 잘못된 토큰
```bash
# 설정 파일에서 토큰 확인
grep token /home/spai0433/.jupyter/jupyter_server_config.py

# Colab에서 동일한 토큰 사용 확인
```

#### 포트 충돌 오류

**로컬 PC의 8888 포트가 이미 사용 중인 경우:**

```powershell
# 포트 확인
netstat -ano | findstr :8888

# 프로세스 종료 (관리자 권한)
taskkill /PID [PID번호] /F

# 또는 다른 포트 사용
gcloud compute ssh spai0433@codeit-ai-g2b-search --project sprint-ai-chunk2-03 --ssh-flag="-L 9999:localhost:8888"

# Colab URL도 변경
# http://localhost:9999/?token=mysecrettoken1234
```

#### 연결은 되지만 파일 접근 불가

**권한 확인:**
```bash
# VM에서 작업 디렉토리 권한 확인
ls -la /home/spai0433

# 필요시 권한 수정
chmod 755 /home/spai0433
```

---

### 15.5. 자동화 스크립트 (선택)

반복 작업을 자동화하는 스크립트입니다.

#### VM 자동 실행 스크립트

`/home/spai0433/start_jupyter.sh` 생성:

```bash
#!/bin/bash

# 기존 Jupyter Server 종료
pkill -f jupyter-server

# 로그 디렉토리 생성
mkdir -p ~/logs

# Jupyter Server 시작
nohup /opt/miniconda3/bin/jupyter server > ~/logs/jupyter_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Jupyter Server 시작됨. PID: $!"
echo "로그: ~/logs/"
```

**실행 권한 부여 및 실행:**
```bash
chmod +x ~/start_jupyter.sh
~/start_jupyter.sh
```

#### Windows 자동 터널링 배치 파일

`start_colab_tunnel.bat` 생성:

```batch
@echo off
echo Starting SSH tunnel for Colab...
gcloud compute ssh spai0433@codeit-ai-g2b-search --project sprint-ai-chunk2-03 --ssh-flag="-L 8888:localhost:8888"
pause
```

**사용법**: 배치 파일을 더블 클릭하여 실행

---

### 15.6. 보안 강화 (프로덕션 환경)

#### 강력한 토큰 생성

```bash
# 32자 무작위 토큰 생성
openssl rand -hex 32

# 출력 예시:
# a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2
```

#### IP 화이트리스트 설정

특정 IP만 접속 허용:

```python
# jupyter_server_config.py에 추가
c.ServerApp.ip = '127.0.0.1'  # SSH 터널링만 허용
```

#### HTTPS 설정 (고급)

```bash
# SSL 인증서 생성
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ~/.jupyter/mykey.key -out ~/.jupyter/mycert.pem

# jupyter_server_config.py에 추가
c.ServerApp.certfile = '/home/spai0433/.jupyter/mycert.pem'
c.ServerApp.keyfile = '/home/spai0433/.jupyter/mykey.key'
```

---

### 15.7. 사용 팁

#### Colab에서 VM 파일 접근

```python
# Colab 노트북에서 실행
!ls -la /home/spai0433
!cat /home/spai0433/myfile.txt
```

#### VM에서 대용량 데이터 처리

```python
# Colab에서 VM의 대용량 데이터 로드
import pandas as pd

# VM의 파일 경로 사용
df = pd.read_csv('/home/spai0433/data/large_dataset.csv')
print(f"Dataset shape: {df.shape}")
```

#### 패키지 설치

Colab에서 VM 환경에 패키지 설치:

```python
# Colab 노트북에서 실행
!pip install transformers accelerate

# 설치 확인
import transformers
print(transformers.__version__)
```

---

### 15.8. 주요 URL 및 포트 정리

| 항목 | URL/포트 | 용도 |
|------|---------|------|
| **Jupyter Server (VM)** | `0.0.0.0:8888` | VM 내부 서버 |
| **SSH 터널링** | `localhost:8888` | 로컬 PC 포트 포워딩 |
| **Colab 연결** | `http://localhost:8888/?token=...` | Colab 백엔드 URL |

---

## 문제 해결

### FTP 연결 문제

#### 1. FTP 연결이 안 될 때

```bash
# VSFTPD 서비스 상태 확인
sudo systemctl status vsftpd

# 서비스 재시작
sudo systemctl restart vsftpd

# FTP 포트 리스닝 확인
sudo netstat -tulpn | grep 21
```

#### 2. Passive 모드 연결 실패

**증상**: 디렉토리 목록을 가져올 수 없음

**원인**:
- `pasv_address`가 VM 외부 IP와 다름
- 방화벽에서 Passive 포트(30000-30009)가 차단됨

**해결**:
```bash
# VM 외부 IP 확인
gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# /etc/vsftpd.conf 수정
sudo vi /etc/vsftpd.conf
# pasv_address를 외부 IP로 변경

# 서비스 재시작
sudo systemctl restart vsftpd

# 방화벽 규칙 확인
gcloud compute firewall-rules describe allow-ftp --project=sprint-ai-chunk2-03
```

#### 3. 쓰기 권한 오류 (503 Permission denied)

**증상**: FTP 업로드 시 "553 Could not create file" 또는 "503 Permission denied" 오류

**원인**:
- `write_enable=YES` 설정이 없거나 주석 처리됨
- 사용자 홈 디렉토리에 쓰기 권한이 없음

**해결**:
```bash
# 1. VSFTPD 설정 확인
grep write_enable /etc/vsftpd.conf
# write_enable=YES 여야 함

# 2. 사용자 홈 디렉토리 권한 확인
ls -la /home/spai0433

# 3. 홈 디렉토리에 쓰기 권한 추가 (503 오류 해결)
chmod u+w /home/spai0433

# 또는 더 명확하게 권한 설정
sudo chmod 755 /home/spai0433

# 4. 특정 디렉토리에만 쓰기 권한 필요 시
chmod u+w /home/spai0433/upload_folder

# 5. 서비스 재시작
sudo systemctl restart vsftpd
```

#### 4. chroot 오류

**증상**: 로그인 후 500 OOPS 오류

**해결**:
```bash
# /etc/vsftpd.conf에 다음 옵션 추가
sudo vi /etc/vsftpd.conf
# allow_writeable_chroot=YES

# 서비스 재시작
sudo systemctl restart vsftpd
```

#### 5. VSFTPD 로그 확인

```bash
# VSFTPD 로그 확인
sudo tail -f /var/log/vsftpd.log

# 시스템 로그에서 VSFTPD 관련 확인
sudo journalctl -u vsftpd -f
```

### JupyterHub 문제 해결

### 포트 8000 사용 중 오류

```bash
# 포트 사용 프로세스 확인
sudo netstat -tulpn | grep 8000

# 프로세스 종료
sudo kill -9 [PID]

# 서비스 재시작
sudo systemctl restart jupyterhub.service
```

### 서비스 시작 실패

```bash
# 상세 로그 확인
sudo journalctl -u jupyterhub.service -xe

# 수동 실행으로 오류 확인
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py
```

### 접속 불가

```bash
# 1. 서비스 상태 확인
sudo systemctl status jupyterhub.service

# 2. 포트 리스닝 확인
sudo netstat -tulpn | grep 8000

# 3. 방화벽 규칙 확인
gcloud compute firewall-rules list | grep jupyterhub

# 4. VM 태그 확인
gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(tags.items)"
```

---

## 빠른 참조

### 주요 경로

| 항목 | 경로 |
|------|------|
| **FTP 관련** | |
| VSFTPD 설정 파일 | `/etc/vsftpd.conf` |
| VSFTPD 로그 | `/var/log/vsftpd.log` |
| 사용자 홈 디렉토리 | `/home/[사용자명]` |
| **JupyterHub 관련** | |
| Miniconda 설치 경로 | `/opt/miniconda3` |
| JupyterHub 환경 | `/opt/miniconda3/envs/jhub-env` |
| JupyterHub 설정 파일 | `/etc/jupyterhub/jupyterhub_config.py` |
| JupyterHub 서비스 파일 | `/etc/systemd/system/jupyterhub.service` |

### 주요 명령어

#### FTP 관련
```bash
# VSFTPD 서비스 상태
sudo systemctl status vsftpd

# VSFTPD 재시작
sudo systemctl restart vsftpd

# FTP 로그 확인
sudo tail -f /var/log/vsftpd.log

# 포트 확인
sudo netstat -tulpn | grep 21

# FTP 503 퍼미션 오류 해결
chmod u+w /home/계정명
sudo systemctl restart vsftpd
```

#### JupyterHub 관련
```bash
# 서비스 상태
sudo systemctl status jupyterhub.service

# 로그 확인
sudo journalctl -u jupyterhub.service -f

# 서비스 재시작
sudo systemctl restart jupyterhub.service

# 설정 테스트
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub -f /etc/jupyterhub/jupyterhub_config.py
```

#### GCP 관련
```bash
# VM 외부 IP 확인
gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(networkInterfaces[0].accessConfigs[0].natIP)"

# 방화벽 규칙 확인
gcloud compute firewall-rules list --project=sprint-ai-chunk2-03

# VM 태그 확인
gcloud compute instances describe codeit-ai-g2b-search --zone=us-central1-c --format="get(tags.items)"
```

---

## 설치 체크리스트

### Part 1: FTP 환경 구축
- [ ] VSFTPD 설치 완료
- [ ] VSFTPD 설정 파일 수정 완료 (`pasv_address` IP 확인)
- [ ] VSFTPD 서비스 시작 및 활성화 완료
- [ ] GCP FTP 방화벽 규칙 생성 완료
- [ ] VM에 ftp-server 태그 추가 완료
- [ ] Windows에서 FTP 포트 연결 테스트 완료
- [ ] IPDisk Z 드라이브 연결 확인

### Part 2: JupyterHub 설치
- [ ] 시스템 업데이트 완료
- [ ] Miniconda 설치 완료
- [ ] jhub-env 환경 생성 완료
- [ ] JupyterHub 설치 완료
- [ ] 설정 파일 작성 완료
- [ ] 사용자 계정 생성 완료
- [ ] Configurable HTTP Proxy 설치 완료
- [ ] systemd 서비스 등록 완료
- [ ] GCP JupyterHub 방화벽 규칙 생성 완료
- [ ] VM에 jupyterhub-server 태그 추가 완료
- [ ] 웹 브라우저 접속 확인
- [ ] 로그인 및 노트북 실행 확인

### Part 3: Colab 로컬 런타임 연결
- [ ] Jupyter Server 설정 파일 생성 완료
- [ ] 고정 토큰 설정 완료
- [ ] Jupyter Server 실행 확인
- [ ] SSH 터널링 설정 완료
- [ ] Colab 로컬 런타임 연결 완료
- [ ] VM 자원 사용 확인

---

## 주요 접속 정보

| 서비스 | 접속 주소 | 포트 | 용도 |
|--------|----------|------|------|
| FTP | `ftp://34.9.92.3` | 21, 30000-30009 | 파일 전송 |
| JupyterHub | `http://34.9.92.3:8000` | 8000 | 웹 기반 노트북 |
| Colab (SSH 터널링) | `http://localhost:8888/?token=...` | 8888 | Colab 로컬 런타임 |
| SSH | `gcloud compute ssh ...` | 22 | VM 관리 |

**주의**: IP 주소는 본인의 VM 외부 IP로 변경하세요.

---

**문서 버전**: 3.0
**최종 수정일**: 2025-11-15
**작성자**: 김명환

**변경 이력**:
- v3.0 (2025-11-15): Colab 로컬 런타임 연결 가이드 추가
- v2.0 (2025-11-15): VSFTPD 설정, GCP 방화벽, Windows FTP 연결 내용 추가
- v1.0 (2025-11-10): JupyterHub 설치 가이드 초안 작성