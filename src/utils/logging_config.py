"""
로깅 설정 모듈

프로젝트 전체 로깅 설정 및 관리
콘솔 및 파일 기반 로깅 시스템 제공

주요 기능:
- 프로젝트 전체 로거 초기화 및 설정
- 콘솔 핸들러 (실시간 로그 출력)
- 파일 핸들러 (로그 파일 저장, 자동 로테이션)
- KST 타임존 적용 및 한글 로그 레벨
- 로깅 레벨 및 포맷 커스터마이징
- 로거 인스턴스 조회 및 재사용
- 중복 핸들러 방지
- settings.yaml 설정 연동
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import pytz
import yaml


# 전역 변수
_loggers: Dict[str, logging.Logger] = {}
_kst = pytz.timezone('Asia/Seoul')


class ShortLevelFormatter(logging.Formatter):
    """
    로그 레벨을 단축 표기하는 커스텀 포맷터
    
    로그 레벨을 한 글자로 축약:
    DEBUG→D, INFO→I, WARNING→W, ERROR→E, CRITICAL→C
    
    시간은 KST(Asia/Seoul) 적용
    """
    
    LEVEL_MAP: Dict[str, str] = {
        'DEBUG': 'D',
        'INFO': 'I',
        'WARNING': 'W',
        'ERROR': 'E',
        'CRITICAL': 'C'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """로그 레벨을 축약하여 포맷"""
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname)
        return super().format(record)
    
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None
    ) -> str:
        """KST 시간으로 변환하여 포맷"""
        ct = datetime.fromtimestamp(record.created, tz=_kst)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime('%Y-%m-%d %H:%M:%S')


def load_settings() -> Dict:
    """
    Config에서 로깅 설정 로드

    Returns:
        Dict: 설정 딕셔너리
    """
    try:
        # Config 임포트 (순환 참조 방지를 위해 함수 내부에서 임포트)
        from src.config import get_config

        config = get_config()
        return {
            'logging': {
                'level': config.LOG_LEVEL,
                'log_dir': config.LOG_DIR,
                'console': True,  # 콘솔 출력 기본 활성화
                'file': True,     # 파일 출력 기본 활성화
                'max_bytes': config.LOG_FILE_MAX_BYTES,
                'backup_count': config.LOG_FILE_BACKUP_COUNT
            }
        }
    except Exception as e:
        print(f"Config 로드 실패, 기본 설정 사용: {e}")
        return {
            'logging': {
                'level': 'INFO',
                'log_dir': './logs',
                'console': True,
                'file': True,
                'max_bytes': 10 * 1024 * 1024,
                'backup_count': 5
            }
        }


def setup_logger(
    name: str = __name__,
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    enable_file: Optional[bool] = None,
    enable_console: Optional[bool] = None,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    로거 설정 및 반환
    
    settings.yaml의 로깅 설정을 자동으로 반영하며,
    콘솔 및 파일 핸들러를 설정합니다.
    
    Args:
        name: 로거 이름
        level: 로깅 레벨 (None이면 settings.yaml에서 로드)
        format_string: 로그 포맷 문자열
        enable_file: 파일 로깅 활성화 (None이면 settings.yaml에서 로드)
        enable_console: 콘솔 로깅 활성화 (None이면 settings.yaml에서 로드)
        log_dir: 로그 디렉토리 경로 (None이면 settings.yaml에서 로드)
        
    Returns:
        logging.Logger: 설정된 로거
    """
    # 이미 설정된 로거가 있으면 재사용
    if name in _loggers:
        return _loggers[name]
    
    # 설정 파일 로드
    settings = load_settings()
    log_config = settings.get('logging', {})
    
    # 로깅 레벨 결정
    if level is None:
        level_str = log_config.get('level', 'INFO')
        level = getattr(logging, level_str, logging.INFO)
    
    # 핸들러 활성화 여부
    if enable_console is None:
        enable_console = log_config.get('console', True)
    if enable_file is None:
        enable_file = log_config.get('file', True)
    
    # 로그 디렉토리
    if log_dir is None:
        log_dir = log_config.get('log_dir', './logs')
    
    # 타입 안전성 보장
    assert level is not None
    assert log_dir is not None
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # propagate 설정 (루트 로거로 전파 방지)
    logger.propagate = False
    
    # 포맷터 설정
    if format_string is None:
        format_string = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    
    formatter = ShortLevelFormatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러 추가
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 핸들러 추가
    if enable_file:
        try:
            # 로그 디렉토리 생성
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # 로그 파일 경로
            log_file = log_path / f"{name.replace('.', '_')}.log"

            # Config에서 로테이션 설정 가져오기
            max_bytes = log_config.get('max_bytes', 10 * 1024 * 1024)
            backup_count = log_config.get('backup_count', 5)

            # RotatingFileHandler 사용
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"파일 핸들러 추가 실패: {e}")
    
    # 로거 캐싱
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    기존 로거 반환 (없으면 기본 설정으로 생성)
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 로거 인스턴스
    """
    # 캐시된 로거 반환
    if name in _loggers:
        return _loggers[name]
    
    # 없으면 새로 생성
    return setup_logger(name)


def reset_logger(name: str) -> None:
    """
    특정 로거 초기화 및 캐시에서 제거
    
    Args:
        name: 로거 이름
    """
    if name in _loggers:
        logger = _loggers[name]
        # 핸들러 제거
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        # 캐시에서 제거
        del _loggers[name]


def reset_all_loggers() -> None:
    """
    모든 로거 초기화 및 캐시 클리어
    """
    for name in list(_loggers.keys()):
        reset_logger(name)
    _loggers.clear()


def set_level(name: str, level: int) -> None:
    """
    특정 로거의 레벨 변경
    
    Args:
        name: 로거 이름
        level: 새로운 로깅 레벨
    """
    if name in _loggers:
        logger = _loggers[name]
        logger.setLevel(level)
        # 모든 핸들러 레벨도 변경
        for handler in logger.handlers:
            handler.setLevel(level)


def get_log_file_path(name: str) -> Optional[Path]:
    """
    로거의 로그 파일 경로 반환
    
    Args:
        name: 로거 이름
        
    Returns:
        Optional[Path]: 로그 파일 경로 (파일 핸들러가 없으면 None)
    """
    if name not in _loggers:
        return None
    
    logger = _loggers[name]
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            return Path(handler.baseFilename)
    
    return None
