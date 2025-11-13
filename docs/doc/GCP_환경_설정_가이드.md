---
layout: default
title: "JupyterHub 설치 가이드 - 따라하기"
description: "Google Cloud VM Ubuntu에서 JupyterHub 빠른 설치"
date: 2025-11-10
author: "김명환"
---

# JupyterHub 설치 가이드 - 따라하기

## 목차

1. [환경 준비](#1-환경-준비)
2. [Miniconda 설치](#2-miniconda-설치)
3. [JupyterHub 설치](#3-jupyterhub-설치)
4. [설정 파일 작성](#4-설정-파일-작성)
5. [사용자 계정 생성](#5-사용자-계정-생성)
6. [Configurable HTTP Proxy 설치](#6-configurable-http-proxy-설치)
7. [시스템 서비스 등록](#7-시스템-서비스-등록)
8. [방화벽 설정](#8-방화벽-설정)
9. [접속 및 테스트](#9-접속-및-테스트)
10. [관리 명령어](#10-관리-명령어)

---

## 1. 환경 준비

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

## 2. Miniconda 설치

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

## 3. JupyterHub 설치

### 가상 환경 생성 및 패키지 설치

```bash
# Python 3.10 가상 환경 생성
sudo /opt/miniconda3/bin/conda create -n jhub-env python=3.10 -y

# JupyterHub, JupyterLab, Notebook 설치
sudo /opt/miniconda3/bin/conda run -n jhub-env \
  pip install jupyterhub jupyterlab notebook

# 버전 확인
sudo /opt/miniconda3/bin/conda run -n jhub-env jupyterhub --version
sudo /opt/miniconda3/bin/conda run -n jhub-env jupyter lab --version
```

---

## 4. 설정 파일 작성

### 설정 디렉토리 및 파일 생성

```bash
# 설정 디렉토리 생성
sudo mkdir -p /etc/jupyterhub

# 설정 파일 생성
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub \
  --generate-config \
  -f /etc/jupyterhub/jupyterhub_config.py

# 백업 생성
sudo cp /etc/jupyterhub/jupyterhub_config.py \
       /etc/jupyterhub/jupyterhub_config.py.org
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

## 5. 사용자 계정 생성

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

## 6. Configurable HTTP Proxy 설치

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

## 7. 시스템 서비스 등록

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

## 8. 방화벽 설정

### GCP 방화벽 규칙 생성

```bash
# 방화벽 규칙 생성 (8000 포트 개방)
gcloud compute firewall-rules create allow-jupyterhub \
  --description="Allow JupyterHub on port 8000" \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:8000 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=jupyterhub-server

# VM에 네트워크 태그 추가
gcloud compute instances add-tags codeit-ai-g2b-search \
  --tags=jupyterhub-server \
  --zone=us-central1-c
```

### 포트 개방 확인

```bash
# VM 내부에서 포트 확인
sudo netstat -tulpn | grep 8000

# 출력 예시:
# tcp   0   0 0.0.0.0:8000   0.0.0.0:*   LISTEN   12345/python
```

---

## 9. 접속 및 테스트

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

## 10. 관리 명령어

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

## 문제 해결

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
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub \
  -f /etc/jupyterhub/jupyterhub_config.py
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
gcloud compute instances describe codeit-ai-g2b-search \
  --zone=us-central1-c --format="get(tags.items)"
```

---

## 빠른 참조

### 주요 경로

| 항목 | 경로 |
|------|------|
| Miniconda 설치 경로 | `/opt/miniconda3` |
| JupyterHub 환경 | `/opt/miniconda3/envs/jhub-env` |
| 설정 파일 | `/etc/jupyterhub/jupyterhub_config.py` |
| 서비스 파일 | `/etc/systemd/system/jupyterhub.service` |
| 사용자 홈 | `/home/[사용자명]` |

### 주요 명령어

```bash
# 서비스 상태
sudo systemctl status jupyterhub.service

# 로그 확인
sudo journalctl -u jupyterhub.service -f

# 서비스 재시작
sudo systemctl restart jupyterhub.service

# 설정 테스트
sudo /opt/miniconda3/envs/jhub-env/bin/jupyterhub \
  -f /etc/jupyterhub/jupyterhub_config.py
```

---

## 설치 체크리스트

- [ ] 시스템 업데이트 완료
- [ ] Miniconda 설치 완료
- [ ] jhub-env 환경 생성 완료
- [ ] JupyterHub 설치 완료
- [ ] 설정 파일 작성 완료
- [ ] 사용자 계정 생성 완료
- [ ] Configurable HTTP Proxy 설치 완료
- [ ] systemd 서비스 등록 완료
- [ ] GCP 방화벽 규칙 생성 완료
- [ ] VM 네트워크 태그 설정 완료
- [ ] 웹 브라우저 접속 확인
- [ ] 로그인 및 노트북 실행 확인

---

**문서 버전**: 1.0  
**최종 수정일**: 2025-11-10  
**작성자**: 김명환