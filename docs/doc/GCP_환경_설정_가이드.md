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

### 주요 명령어 치트시트
```bash
# 503 퍼미션 오류 해결
chmod u+w /home/계정명 && sudo systemctl restart vsftpd

# 방화벽 규칙 생성 (Windows PowerShell)
gcloud compute firewall-rules create allow-ftp --description="Allow FTP Control (21) and Passive Data Ports (30000-30009)" --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules="tcp:21,tcp:30000-30009" --source-ranges=0.0.0.0/0 --target-tags=ftp-server --project=sprint-ai-chunk2-03

# VM 태그 추가 (Windows PowerShell)
gcloud compute instances add-tags codeit-ai-g2b-search --tags=ftp-server --zone=us-central1-c --project=sprint-ai-chunk2-03
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

---

## 주요 접속 정보

| 서비스 | 접속 주소 | 포트 | 용도 |
|--------|----------|------|------|
| FTP | `ftp://34.9.92.3` | 21, 30000-30009 | 파일 전송 |
| JupyterHub | `http://34.9.92.3:8000` | 8000 | 웹 기반 노트북 |
| SSH | `gcloud compute ssh ...` | 22 | VM 관리 |

**주의**: IP 주소는 본인의 VM 외부 IP로 변경하세요.

---

**문서 버전**: 2.0
**최종 수정일**: 2025-11-15
**작성자**: 김명환

**변경 이력**:
- v2.0 (2025-11-15): VSFTPD 설정, GCP 방화벽, Windows FTP 연결 내용 추가
- v1.0 (2025-11-10): JupyterHub 설치 가이드 초안 작성