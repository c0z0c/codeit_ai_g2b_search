# -*- coding: utf-8 -*-
"""
문서 검색 시스템 (PDF, HWP, DOCX 등)
Streamlit UI 초안 구현 및 테스트
"""

import streamlit as st
import os
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.db import DocumentsDB, ChatHistoryDB
from src.config import get_config

from styles.streamlit_styling import load_css, apply_default_styling

# 앱 시작 후 추가
load_css("scripts/오형주/styles/styles.css")  # CSS 파일 로드
apply_default_styling()  # 기본 Streamlit 오버라이드

# 페이지 설정
st.set_page_config(
    page_title="문서 검색 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Config 초기화
@st.cache_resource
def init_config():
    """Config 싱글톤 로드"""
    return get_config()

config = init_config()

# 세션 상태 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = config.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY', '')

# DB 초기화
@st.cache_resource
def init_dbs():
    """데이터베이스 초기화"""
    cfg = get_config()
    return {
        'docs': DocumentsDB(cfg.DOCUMENTS_DB_PATH),
        'chat': ChatHistoryDB(cfg.CHAT_HISTORY_DB_PATH)
    }

dbs = init_dbs()

"""
Phase2: 해야할것
- [ ] 사이드바 구현
- [ ] OpenAI API Key 입력 위젯
- [ ] 데이터 업데이트 버튼 (신승목 API 호출)
- [ ] 임베딩 업데이트 버튼 (김명환 API 호출)
- [ ] 채팅 세션 선택 드롭다운
- [ ] 새 세션 생성 버튼
- [ ] 메인 영역 구현
- [ ] 채팅 메시지 표시 컨테이너
- [ ] 사용자 입력 텍스트 박스
- [ ] 전송 버튼
"""
