# -*- coding: utf-8 -*-
# 기존 TestApp.py의 코드 -> App.py로 이동 (특정부분 에러발생확률 높음)
"""
문서 검색 시스템 (PDF, HWP, DOCX 등)
Streamlit UI 초안 구현 및 테스트
"""

import streamlit as st

# Streamlit 페이지 설정 - 반드시 첫 번째 Streamlit 명령
st.set_page_config(
    page_title="문서 검색 시스템",
    layout="wide",
)


import os
from openai import OpenAI
from pathlib import Path
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
import importlib
import tempfile
import shutil

import logging
from src.utils.logging_config import setup_logger
from src.utils.helper_utils import *
from src.utils.helper_c0z0c_dev import *


from src import config
from src.config import get_config, Config

from src.processors import document_processor
importlib.reload(document_processor)
from src.processors import document_processor

from src.processors import embedding_processor
importlib.reload(embedding_processor)
from src.processors import embedding_processor

from src.llm import llm_processor
importlib.reload(llm_processor)
from src.llm import llm_processor

from src.db import DocumentsDB, ChatHistoryDB
from src.vectorstore import VectorStoreManager

from src.processors.document_processor import DocumentProcessor
from src.processors.embedding_processor import EmbeddingProcessor

from src.llm import retrieval
from src.llm.retrieval import Retrieval
from src.llm import llm_processor
from src.llm.llm_processor import LLMProcessor

from src.ui.sidebar_scroll import scroll_sidebar_for_tab, add_section_anchor
from src.ui.streamlit_styling import load_css, apply_default_styling

# .env 파일 로드
PROJECT_ROOT_PATH = Path(__file__).resolve().parent  # app.py의 부모 = 프로젝트 루트
ENV_PATH = PROJECT_ROOT_PATH / '.env'
CONFIG_PATH = PROJECT_ROOT_PATH / "config" / 'config.json'
STYLES_PATH = PROJECT_ROOT_PATH / "src" / "ui" / 'styles.css'

# sys.path.insert(0, str(PROJECT_ROOT_PATH))  # src 폴더를 sys.path에 추가

# 환경 변수 읽어오기
if Path(ENV_PATH).exists():
    load_dotenv(ENV_PATH)
    
    
# CSS 로드 및 스타일 적용 - set_page_config 다음
load_css(str(STYLES_PATH))  # CSS 파일 로드
apply_default_styling()  # 기본 Streamlit 오버라이드

# Config 초기화
@st.cache_resource
def init_config():
    """Config 싱글톤 로드"""
    # 설정파일 읽어오기
    if Path(CONFIG_PATH).exists():
        cfg = get_config(CONFIG_PATH)
    else:
        cfg = get_config()

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        logger.warning("OpenAI API 키 필요")    

    cfg.OPENAI_API_KEY = openai_api_key
    cfg.DOCUMENTS_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "documents.db")
    cfg.EMBEDDINGS_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "embeddings.db")
    cfg.CHAT_HISTORY_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "chat_history.db")
    cfg.VECTORSTORE_PATH = str(PROJECT_ROOT_PATH / "data" / "vectorstore")
    cfg.CONFIG_PATH = CONFIG_PATH
    return cfg

if 'config' not in st.session_state:
    st.session_state.config = init_config()
config = st.session_state.config

# print_dic_tree(config.to_dict())

# 커스텀 설정으로 로거 생성
if 'logger' not in st.session_state:
    st.session_state.logger = setup_logger(
        name="app",
        level=logging.DEBUG,
        format_string='%(asctime)s [%(levelname)s] %(message)s',
        enable_console=True,
        enable_file=True,
        log_dir="logs"
    )
logger = st.session_state.logger    


# 세션 상태 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"}]

# API Key 초기화 및 검증
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.getenv('OPENAI_API_KEY', '').strip()

# ------------------------------------------------------------------------------------------------
# 프로세서 초기화
# open ai API 키가 없으면 입력을 받음
# ------------------------------------------------------------------------------------------------
# API Key 입력 강제 (비어있으면 입력 화면만 표시)
if not st.session_state.api_key:
    st.title("OpenAI API Key 입력 필요")
    st.markdown("---")
    st.info("앱을 시작하려면 OpenAI API Key를 입력해주세요.")
    
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        key="initial_api_key_input"
    )
    
    if st.button("시작하기", type="primary", use_container_width=True):
        if api_key_input and api_key_input.strip():
            st.session_state.api_key = api_key_input.strip()
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key
            config.OPENAI_API_KEY = st.session_state.api_key
            st.success("API Key가 설정되었습니다. 잠시 후 앱이 시작됩니다.")
            st.rerun()
        else:
            st.error("유효한 API Key를 입력해주세요.")
    
    st.stop()  # API Key 입력 전까지 아래 코드 실행 중단
# ===============================================================================================

# 현재 선택된 세션 표시명을 위한 초기화
# Streamlit의 session_state는 명시적 초기화가 필요합니다.
if 'selected_session' not in st.session_state:
    st.session_state.selected_session = "새 세션"

# 현재 사용 중인 언어모델 (초기값: .env 또는 gpt-5)
if 'current_model' not in st.session_state:
    st.session_state.current_model = os.getenv('OPENAI_MODEL', 'gpt-5')

# 세션 이름 변경 필요 여부
if 'session_needs_rename' not in st.session_state:
    st.session_state.session_needs_rename = False

# 현재 활성 탭 추적
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "AI 채팅"

# 임시 디렉토리 생성
# 업로드 파일 저장
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()


# ------------------------------------------------------------------------------------------------
# 미션 프로벡트 AI  인스턴스 선언
# ------------------------------------------------------------------------------------------------

@st.cache_resource
def init_process():
    """프로세스"""
    logger.debug("프로세스 초기화...")
    cfg = config
    return {
        'proc_doc': DocumentProcessor(config=config),
        'proc_emb': EmbeddingProcessor(config=config),
        'llm_retrieval': Retrieval(config=config),
        'llm_processor': LLMProcessor(config=config),
    }

if 'processes' not in st.session_state:
    logger.debug("프로세서 초기화 중...")
    st.session_state.processes = init_process()
processes = st.session_state.processes

proc_doc = st.session_state.processes['proc_doc']
proc_emb = st.session_state.processes['proc_emb']
llm_retrieval = st.session_state.processes['llm_retrieval']
llm_processor = st.session_state.processes['llm_processor']


# DB 초기화
@st.cache_resource
def init_dbs():
    """데이터베이스 초기화"""
    logger.debug("데이터베이스 초기화...")
    cfg = config
    return {
        'chat': ChatHistoryDB(cfg.CHAT_HISTORY_DB_PATH),
        'docs': proc_doc.docs_db,
    }

if 'dbs' not in st.session_state:
    logger.debug("프로세서 초기화 중...")
    st.session_state.dbs = init_dbs()
dbs = st.session_state.dbs    

# ===============================================================================================

# ----- 사이드바 구현 구간 -----
with st.sidebar:
    st.title("설정 및 세션")
    
    # OpenAI API Key 입력 위젯
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", 
                                    value=st.session_state.api_key, 
                                    type="password")
    
    # API 키가 유효하게 입력되었는지 확인하는 플래그
    api_key_valid = False 
    if openai_api_key:
        st.session_state.api_key = openai_api_key
        st.success("API Key 입력 완료!")
        api_key_valid = True 
    else:
        st.warning("API Key를 입력해주세요.")
    
    st.markdown("---")

    # 데이터 통계
    add_section_anchor("analytics-section")
    st.subheader("데이터 통계")
    try:
        logger.debug("데이터 통계 로드 시도...")
        doc_stats = dbs['docs'].get_document_stats()
        #embedding_stats = dbs['embeddings'].get_embedding_stats() 데이터 통계 로드 실패로 임시주석처리
        col1, col2 = st.columns(2)
        with col1:
            st.metric("문서 수", f"{doc_stats.get('total_files', 0)}")
            st.metric("페이지 수", f"{doc_stats.get('total_pages', 0)}")
        with col2:
            st.metric("토큰 수", f"{doc_stats.get('total_tokens', 0)}")
            st.metric("파일 크기", f"{doc_stats.get('total_size', 0)} bytes")
    except Exception as e:
        st.warning(f"데이터 통계 로드 실패: {str(e)}")
        st.info("더미 데이터를 생성하려면 '더미 데이터 생성' 버튼을 클릭하세요.")

    st.divider()

    add_section_anchor("document-search-section")
    st.title("업로드할 파일 선택 ") # 지금은 PDF파일만 업로드하고 추후 다양한 포맷 지원 예정

    # 파일 업로드 버튼 추가
    uploaded_file = st.file_uploader(
        "여기에 파일을 업로드하세요", # 사용자에게 보여줄 텍스트
        type=['pdf', 'hwp'] # 허용할 파일 확장자 목록 (선택 사항) ['csv', 'txt', 'pdf', 'png'...]
    )

    # 파일이 성공적으로 업로드되었는지 확인하고 처리
    if uploaded_file is not None:
        st.success(f"파일 '{uploaded_file.name}'이(가) 성공적으로 업로드되었습니다.")
    
        temp_file_path = Path(st.session_state.temp_dir) / uploaded_file.name
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
            logger.debug(f"업로드된 파일이 임시 경로에 저장됨: {str(temp_file_path)}")
    
        logger.debug(f"업로드된 파일 정보: 이름={uploaded_file.name}, 타입={uploaded_file.type}, 크기={uploaded_file.size} bytes")
        
        file_hash, result = proc_doc.process_doc(str(temp_file_path))
        
        summary = None
        if result is False:
            logger.error("파일 처리에 실패했습니다.")
            st.error("파일 처리에 실패했습니다.")
        else:
            logger.info("파일이 성공적으로 처리되었습니다. 임베딩을 동기화합니다...")   
            st.success("파일이 성공적으로 처리되었습니다. 임베딩을 동기화합니다...")
            proc_emb.sync_with_docs_db(config.OPENAI_API_KEY)
            summary = proc_emb.vector_manager.get_summary(file_hash)
        
        if temp_file_path.exists():
            temp_file_path.unlink()  # 업로드 후 임시 파일 삭제
            logger.debug(f"임시 파일 삭제됨: {str(temp_file_path)}")
    
        # 예시: 업로드된 파일의 타입과 크기 표시
        file_details = {
            "파일 이름": uploaded_file.name,
            "파일 타입": uploaded_file.type,
            "파일 크기 (바이트)": uploaded_file.size
        }
            
        st.write("---")
        st.subheader("업로드된 파일 상세 정보")
        st.json(file_details)
        
        if summary is not None:
            st.write("---")
            st.subheader("임베딩 요약 정보")
            st.json(summary)
        
        # 파일을 읽고 싶다면 (예: CSV 파일)
        # import pandas as pd
        # if uploaded_file.type == "text/csv":
        #     df = pd.read_csv(uploaded_file)
        #     st.dataframe(df.head())
    else:
        st.info("파일을 기다리고 있습니다...")

    # 데이터/임베딩 업데이트 버튼 (!!!여기는 만들어진 API를 버튼 눌렀을 시 작동하는 코드가 필요!!!)
    st.title("데이터 및 임베딩 업데이트")
    if st.button("데이터 업데이트 (A API)", use_container_width=True, key="btn_data_update", disabled=not api_key_valid):
        st.info("데이터 업데이트 시작...")
        st.success("데이터 업데이트 완료!")
        
    if st.button("임베딩 업데이트 (B API)", use_container_width=True, key="btn_embedding_update", disabled=not api_key_valid):
        st.info("새 데이터를 기반으로 임베딩 벡터를 갱신하고 있습니다...")
        st.success("임베딩 업데이트 완료!")

    # 채팅 세션 관리
    add_section_anchor("chat-session-section", "채팅 세션 관리") # 메인 영역 버튼 누르면 사이드바 이동
    
    model_options = ["gpt-5", "gpt-5-nano", "gpt-5-mini"]
    selected_model = st.selectbox(
        "언어모델 선택",
        options=model_options,
        index=model_options.index(st.session_state.get('current_model', 'gpt-5')),
        key="chat_model_select_below",
    )
    if selected_model != st.session_state.get('current_model'):
        st.session_state.current_model = selected_model
        st.info(f"언어모델이 '{selected_model}'(으)로 변경되었습니다.")

    # 새로운 세션 생성
    if st.button("새 세션 생성", use_container_width=True, key="btn_new_session"):
        session_name = f"새 채팅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        new_session_id = dbs['chat'].create_session(session_name)

        # 초기 환영 메시지 추가 (updated_at을 최신으로 만들기 위해)
        welcome_msg = "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"
        dbs['chat'].add_message(new_session_id, "assistant", welcome_msg)
        st.session_state.session_id = new_session_id
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.selected_session = session_name
        st.session_state.session_needs_rename = True
        st.rerun()
    
    # 세션 목록 불러오기 (최신 5개만)
    session_list = dbs['chat'].list_sessions()[:5]
    session_names = [s['session_name'] for s in session_list]
    session_ids = [s['session_id'] for s in session_list]

    # 세션 선택 (사이드바)
    if session_list:
        selected_idx = 0

        # 현재 선택된 세션이 있으면 해당 인덱스
        if st.session_state.session_id in session_ids:
            selected_idx = session_ids.index(st.session_state.session_id)

        selected_session_name = st.radio(
            "저장된 채팅 세션 목록 (최신 5개)",
            options=session_names,
            index=selected_idx,
            key="sidebar_session_radio",
        )

        # 선택 시 해당 세션의 메시지 불러오기
        if selected_session_name != st.session_state.selected_session:
            sel_idx = session_names.index(selected_session_name)
            sel_id = session_ids[sel_idx]
            st.session_state.session_id = sel_id
            st.session_state.selected_session = selected_session_name

            # DB에서 메시지 불러와서 role, content만 추출
            db_messages = dbs['chat'].get_session_messages(sel_id)
            st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
            st.session_state.session_needs_rename = False
            st.rerun()
        
        st.markdown("---")

        # 선택된 세션 삭제 버튼
        if st.button("선택된 세션 삭제", use_container_width=True, type="secondary", key="delete_current_session"):
            current_idx = session_names.index(st.session_state.selected_session)
            current_sess_id = session_ids[current_idx]

            if dbs['chat'].delete_session(current_sess_id):
                # 남은 세션 확인
                remaining_sessions = dbs['chat'].list_sessions()

                if remaining_sessions:
                    # 남은 세션 중 첫 번째 세션 선택
                    first_session = remaining_sessions[0]
                    st.session_state.session_id = first_session['session_id']
                    st.session_state.selected_session = first_session['session_name']
                    db_messages = dbs['chat'].get_session_messages(first_session['session_id'])
                    st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
                    st.session_state.session_needs_rename = False
                else:
                    # 세션이 하나도 없으면 새 세션 생성
                    session_name = f"새 채팅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    new_session_id = dbs['chat'].create_session(session_name)
                    welcome_msg = "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"
                    dbs['chat'].add_message(new_session_id, "assistant", welcome_msg)
                    st.session_state.session_id = new_session_id
                    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                    st.session_state.selected_session = session_name
                    st.session_state.session_needs_rename = True

                st.rerun()

    else:
        st.info("저장된 채팅 세션이 없습니다.")

# ----- 2. 메인 영역 구현 -----

# 메인 영역 제목
st.title("문서 검색 시스템")

# 탭 생성 및 선택 추적
selected_tab = st.radio(
    "메뉴 선택",
    ["AI 채팅", "문서 검색", "분석 및 통계"],
    horizontal=True,
    label_visibility="collapsed"
)

# 선택된 탭에 따라 사이드바 스크롤
if selected_tab == "AI 채팅":
    scroll_sidebar_for_tab("AI 채팅")

elif selected_tab == "문서 검색":
    scroll_sidebar_for_tab("문서 검색")

elif selected_tab == "분석 및 통계":
    scroll_sidebar_for_tab("분석 및 통계")

# ===== 1번 탭: AI 채팅 =====
if selected_tab == "AI 채팅":
    st.subheader(f"현재 세션: {st.session_state.selected_session}")

    # 세션이 없으면 생성
    if st.session_state.session_id is None:
        session_name = f"새 채팅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        new_session_id = dbs['chat'].create_session(session_name)
        welcome_msg = "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"
        dbs['chat'].add_message(new_session_id, "assistant", welcome_msg)
        st.session_state.session_id = new_session_id
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.selected_session = session_name
        st.session_state.session_needs_rename = True

    # 채팅 메시지 표시 컨테이너
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 텍스트 박스 & 10. 전송 버튼 구현
    if prompt := st.chat_input("여기에 메시지를 입력하세요...", disabled=not api_key_valid):

        # 사용자 입력 저장 및 화면에 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # DB에 사용자 메시지 저장
        dbs['chat'].add_message(st.session_state.session_id, "user", prompt)
    
        # 첫 메시지로 세션 이름 변경
        if st.session_state.session_needs_rename:
            # 메시지를 30자로 제한
            session_name = prompt[:30] + "..." if len(prompt) > 30 else prompt

            # DB에서 세션 이름 업데이트
            with dbs['chat']._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE chat_sessions SET session_name = ? WHERE session_id = ?", (session_name, st.session_state.session_id))
                conn.commit()

            st.session_state.selected_session = session_name
            st.session_state.session_needs_rename = False

        # API 키 유효성을 다시 확인하고, 유효한 경우에만 API 호출
        if api_key_valid:
            try:
                # OpenAI 클라이언트 초기화
                client = OpenAI(api_key=openai_api_key)

                # API 호출을 위한 메시지 리스트 준비
                # Streamlit 세션 상태의 messages를 OpenAI API 형식에 맞게 사용
                messages_for_api = st.session_state.messages

                # AI 응답 생성 (스트리밍 사용)
                with st.chat_message("assistant"):
                    # chat.completions.create 호출
                    # 모델은 세션 상태에서 가져오며, 없으면 .env 기본값 사용
                    model_to_use = st.session_state.get('current_model') or os.getenv('OPENAI_MODEL', 'gpt-5-nano')
                    stream = client.chat.completions.create(
                        model=model_to_use, # 사용가능모델: gpt-5, gpt-5-nano, gpt-5-mini
                        messages=messages_for_api,
                        stream=True,
                    )
                    
                    # Streamlit의 st.write_stream을 사용하여 응답을 실시간으로 화면에 출력
                    response = st.write_stream(stream)
                
                # AI 응답 저장
                st.session_state.messages.append({"role": "assistant", "content": response})
                # DB에 AI 응답 저장
                dbs['chat'].add_message(st.session_state.session_id, "assistant", response)

            except Exception as e:
                st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
                st.session_state.messages.pop() # 오류 발생 시 마지막 사용자 메시지 제거

        else:
            st.error("OpenAI API Key를 먼저 입력해주세요.")

# ===== 2번 탭: 문서 검색 =====
elif selected_tab == "문서 검색":
    # subheader와 selectbox를 같은 줄에 배치
    header_col1, header_col2, header_col3 = st.columns([3, 2, 1])
    with header_col1:
        st.subheader("문서 검색")
    with header_col2:
        search_type = st.selectbox("검색 유형", ["키워드 검색", "시맨틱 검색", "하이브리드 검색"], key="search_type_select")
    with header_col3:
        top_k = st.number_input("결과 수", min_value=1, max_value=20, value=5, key="top_k_input")
    
    # 검색어 입력과 버튼
    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        search_query = st.text_input("검색어", key="doc_search_input", label_visibility="collapsed", placeholder="검색어를 입력하세요")
    with search_col2:
        search_button = st.button("검색", key="btn_search", use_container_width=True)
    
    if search_button:
        if search_query:
            st.info(f"'{search_query}' 검색 중...")
            # TODO: 실제 검색 로직 구현
            # TODO: 검색 결과에 다음 정보 포함:
            #   - 문서 제목/파일명
            #   - 매칭된 페이지 번호 (page_number)
            #   - 해당 페이지의 텍스트 스니펫 (하이라이트)
            #   - 유사도 점수
            # 예시 결과 형식:
            # {
            #   "document": "제안요청서_2024.pdf",
            #   "page": 15,
            #   "snippet": "...검색어가 포함된 텍스트...",
            #   "score": 0.95
            # }
            st.success("검색 완료! (검색 기능 구현 예정)")
        else:
            st.warning("검색어를 입력해주세요.")
    
    # 검색 결과 표시 영역
    st.markdown("---")
    st.subheader("검색 결과")
    st.info("검색 결과가 여기에 표시됩니다.")
    
    # TODO: 검색 결과 표시 예시
    # for result in search_results:
    #     with st.expander(f"{result['document']} - 페이지 {result['page']}"):
    #         st.markdown(f"**유사도:** {result['score']:.2%}")
    #         st.markdown(f"**내용:** {result['snippet']}")
    #         st.markdown(f"[원본 페이지로 이동 →](#page-{result['page']})")

# ===== 3번 탭: 분석 및 통계 =====
elif selected_tab == "분석 및 통계":
    st.subheader("시스템 분석 및 통계")
    
    # 데이터베이스 통계
    try:
        doc_stats = dbs['docs'].get_document_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 문서 수", doc_stats.get('total_files', 0))
        with col2:
            st.metric("총 페이지 수", doc_stats.get('total_pages', 0))
        with col3:
            st.metric("총 토큰 수", doc_stats.get('total_tokens', 0))
        with col4:
            st.metric("총 파일 크기", f"{doc_stats.get('total_size', 0) / 1024:.1f} KB")
        
        st.markdown("---")
        
        # 채팅 통계
        chat_stats = dbs['chat'].get_chat_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 세션 수", chat_stats.get('total_sessions', 0))
        with col2:
            st.metric("총 메시지 수", chat_stats.get('total_messages', 0))
        with col3:
            st.metric("활성 세션 수", chat_stats.get('active_sessions', 0))
        
    except Exception as e:
        st.error(f"통계 로드 실패: {str(e)}")
    
    st.markdown("---")
    st.info("추가 분석 및 시각화 기능이 여기에 추가될 예정입니다.")
    
# 실행 예시: Streamlit을 실행하면 왼쪽에 "설정 및 세션" 제목이 있는 사이드바가 보입니다.
# ------- 사이드바  끝 구간 -------

