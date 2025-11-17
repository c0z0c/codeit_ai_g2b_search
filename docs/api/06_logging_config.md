---
layout: default
title: "RAG 시스템 인터페이스 문서 - Logging Config 인터페이스 문서"
description: "RAG 시스템 인터페이스 문서 - Logging Config 인터페이스 문서"
date: 2025-11-14
author: "김명환"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# Logging Config 인터페이스 문서

## 파일 정보
- **경로**: `src/utils/logging_config.py`
- **목적**: 프로젝트 전체 로깅 설정 및 관리

## 개요
프로젝트 전체의 로깅 시스템을 중앙에서 관리하는 모듈입니다. 콘솔 및 파일 기반 로깅, KST 타임존 적용, 한글 로그 레벨 지원 등의 기능을 제공합니다.

### 주요 기능
- 프로젝트 전체 로거 초기화 및 설정
- 콘솔 핸들러 (실시간 로그 출력)
- 파일 핸들러 (로그 파일 저장, 자동 로테이션)
- KST 타임존 적용 및 한글 로그 레벨
- 로깅 레벨 및 포맷 커스터마이징
- 로거 인스턴스 조회 및 재사용
- 중복 핸들러 방지
- settings.yaml 설정 연동

## 클래스

### ShortLevelFormatter

#### 개요
로그 레벨을 단축 표기하는 커스텀 포맷터입니다. 로그 레벨을 한 글자로 축약하고 KST 시간을 적용합니다.

#### 레벨 축약 매핑
- `DEBUG` → `D`
- `INFO` → `I`
- `WARNING` → `W`
- `ERROR` → `E`
- `CRITICAL` → `C`

#### 메서드

##### format(record: LogRecord) -> str
로그 레코드를 포맷팅합니다.

**Parameters:**
- `record` (LogRecord): 로그 레코드

**Returns:**
- `str`: 포맷팅된 로그 문자열

##### formatTime(record: LogRecord, datefmt: Optional[str] = None) -> str
KST 시간으로 변환하여 포맷팅합니다.

**Parameters:**
- `record` (LogRecord): 로그 레코드
- `datefmt` (Optional[str]): 날짜/시간 포맷 문자열

**Returns:**
- `str`: 포맷팅된 시간 문자열

## 주요 함수

### load_settings(config: Optional[Config] = None) -> Dict

Config에서 로깅 설정을 로드합니다.

**Parameters:**
- `config` (Optional[Config]): Config 인스턴스 (None이면 자동 로드)

**Returns:**
- `Dict`: 로깅 설정 딕셔너리

**설정 항목:**
- `level`: 로깅 레벨 (기본값: "INFO")
- `log_dir`: 로그 디렉토리 경로 (기본값: "./logs")
- `log_file_name`: 로그 파일명 (기본값: "rag_system.log")
- `console`: 콘솔 출력 활성화 (기본값: True)
- `file`: 파일 출력 활성화 (기본값: True)
- `max_bytes`: 로그 파일 최대 크기 (기본값: 10MB)
- `backup_count`: 백업 파일 개수 (기본값: 5)

**사용 예:**
```python
settings = load_settings()
log_config = settings['logging']
print(log_config['level'])
```

### setup_logger(...)

로거를 설정하고 반환합니다. settings.yaml의 로깅 설정을 자동으로 반영하며, 콘솔 및 파일 핸들러를 설정합니다.

**Parameters:**
- `name` (str): 로거 이름 (기본값: __name__)
- `level` (Optional[int]): 로깅 레벨 (None이면 settings.yaml에서 로드)
- `format_string` (Optional[str]): 로그 포맷 문자열
- `enable_file` (Optional[bool]): 파일 로깅 활성화 (None이면 settings.yaml에서 로드)
- `enable_console` (Optional[bool]): 콘솔 로깅 활성화 (None이면 settings.yaml에서 로드)
- `log_dir` (Optional[str]): 로그 디렉토리 경로 (None이면 settings.yaml에서 로드)
- `config` (Optional[Config]): Config 인스턴스

**Returns:**
- `logging.Logger`: 설정된 로거 인스턴스

**특징:**
- 이미 설정된 로거가 있으면 재사용
- 중복 핸들러 방지 (기존 핸들러 제거 후 재설정)
- 루트 로거로 전파 방지 (propagate=False)
- RotatingFileHandler 사용 (자동 로그 로테이션)

**사용 예:**
```python
from src.utils.logging_config import setup_logger
import logging

# 기본 설정으로 로거 생성
logger = setup_logger(__name__)
logger.info("Hello, World!")

# 커스텀 설정으로 로거 생성
logger = setup_logger(
    name="my_module",
    level=logging.DEBUG,
    format_string='%(asctime)s [%(levelname)s] %(message)s',
    enable_console=True,
    enable_file=True,
    log_dir="./custom_logs"
)
```

### get_logger(name: str = __name__) -> Logger

기존 로거를 반환합니다. 없으면 기본 설정으로 생성합니다.

**Parameters:**
- `name` (str): 로거 이름 (기본값: __name__)

**Returns:**
- `logging.Logger`: 로거 인스턴스

**사용 예:**
```python
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
logger.debug("Debug information")
logger.error("Error occurred")
```

### reset_logger(name: str) -> None

특정 로거를 초기화하고 캐시에서 제거합니다.

**Parameters:**
- `name` (str): 로거 이름

**사용 예:**
```python
from src.utils.logging_config import reset_logger

# 로거 초기화
reset_logger("my_module")
```

### reset_all_loggers() -> None

모든 로거를 초기화하고 캐시를 클리어합니다.

**사용 예:**
```python
from src.utils.logging_config import reset_all_loggers

# 모든 로거 초기화
reset_all_loggers()
```

### set_level(name: str, level: int) -> None

특정 로거의 레벨을 변경합니다.

**Parameters:**
- `name` (str): 로거 이름
- `level` (int): 새로운 로깅 레벨 (logging.DEBUG, logging.INFO 등)

**사용 예:**
```python
from src.utils.logging_config import set_level
import logging

# 로거 레벨 변경
set_level("my_module", logging.DEBUG)
```

### get_log_file_path(name: str) -> Optional[Path]

로거의 로그 파일 경로를 반환합니다.

**Parameters:**
- `name` (str): 로거 이름

**Returns:**
- `Optional[Path]`: 로그 파일 경로 (파일 핸들러가 없으면 None)

**사용 예:**
```python
from src.utils.logging_config import get_log_file_path

log_path = get_log_file_path("my_module")
if log_path:
    print(f"Log file: {log_path}")
```

## 전역 변수

### _loggers: Dict[str, logging.Logger]
생성된 로거 인스턴스를 캐싱하는 딕셔너리입니다.

### _kst: pytz.timezone
KST(Asia/Seoul) 타임존 객체입니다.

## 로그 포맷

### 기본 포맷
```
%(asctime)s [%(levelname)s] %(name)s - %(message)s
```

### 출력 예시
```
2025-11-14 15:30:45 [I] my_module - Application started
2025-11-14 15:30:46 [D] my_module - Debug information
2025-11-14 15:30:47 [E] my_module - Error occurred
```

## 로그 로테이션

RotatingFileHandler를 사용하여 자동 로그 로테이션을 지원합니다.

**기본 설정:**
- 최대 파일 크기: 10MB
- 백업 파일 개수: 5개

**로테이션 예시:**
```
logs/
├── rag_system.log        # 현재 로그 파일
├── rag_system.log.1      # 백업 1
├── rag_system.log.2      # 백업 2
├── rag_system.log.3      # 백업 3
├── rag_system.log.4      # 백업 4
└── rag_system.log.5      # 백업 5 (가장 오래된 파일)
```

## 사용 패턴

### 기본 사용
```python
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)
logger.info("Starting process...")
logger.debug("Processing item 1")
logger.error("Failed to process item 2")
```

### 모듈별 로거
```python
# module_a.py
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def process():
    logger.info("Processing in module A")
```

```python
# module_b.py
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def process():
    logger.info("Processing in module B")
```

### 동적 레벨 변경
```python
from src.utils.logging_config import setup_logger, set_level
import logging

logger = setup_logger(__name__)
logger.info("This will be logged")
logger.debug("This will NOT be logged (default level is INFO)")

# 레벨을 DEBUG로 변경
set_level(__name__, logging.DEBUG)
logger.debug("Now this will be logged")
```

## Config 연동

logging_config는 [Config 인터페이스](01_config_interface.md)의 로깅 설정을 자동으로 반영합니다.

**관련 Config 속성:**
- `LOG_LEVEL`: 로깅 레벨
- `LOG_DIR`: 로그 디렉토리
- `LOG_FILE_NAME`: 로그 파일명
- `LOG_FILE_MAX_BYTES`: 최대 파일 크기
- `LOG_FILE_BACKUP_COUNT`: 백업 파일 개수

**사용 예:**
```python
from src.config import get_config
from src.utils.logging_config import setup_logger

# Config 설정 확인
config = get_config()
print(f"Log level: {config.LOG_LEVEL}")
print(f"Log directory: {config.LOG_DIR}")

# Config 기반 로거 생성 (자동으로 Config 설정 반영)
logger = setup_logger(__name__)
logger.info("Logger configured from Config")
```

## 주의사항

1. **로거 재사용**: 동일한 이름의 로거는 캐싱되어 재사용됩니다. 설정을 변경하려면 `reset_logger()`를 사용하세요.
2. **중복 핸들러 방지**: `setup_logger()`는 기존 핸들러를 제거하고 재설정하므로 중복 로그 출력을 방지합니다.
3. **파일 인코딩**: 로그 파일은 UTF-8 인코딩으로 저장됩니다.
4. **타임존**: 모든 로그 시간은 KST(Asia/Seoul)로 표시됩니다.
5. **로그 디렉토리**: 로그 디렉토리가 없으면 자동으로 생성됩니다.

## 참고 문서

- [Config 인터페이스](01_config_interface.md): 로깅 설정 관리
- Python logging 공식 문서: https://docs.python.org/3/library/logging.html
