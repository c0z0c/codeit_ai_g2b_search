# -*- coding: utf-8 -*-
# app.py 원본저장용
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
import re
from openai import OpenAI
from pathlib import Path
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
import importlib
import tempfile
import shutil
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import logging
from src.utils.logging_config import setup_logger
from src.utils.helper_utils import *
from src.utils.helper_c0z0c_dev import *


from src import config
from src.config import get_config, Config


from src.processors import document_processor
from src.processors import embedding_processor
from src.llm import llm_processor
from src.llm import rag_evaluator
from src.db import chat_history_db
from src.db import documents_db
from src.db import DocumentsDB, ChatHistoryDB
from src.vectorstore import VectorStoreManager

from src.processors.document_processor import DocumentProcessor
from src.processors.embedding_processor import EmbeddingProcessor


from src.llm import retrieval
from src.llm.retrieval import Retrieval
from src.llm import llm_processor
from src.llm.llm_processor import LLMProcessor
from src.llm.rag_evaluator import RAGEvaluator


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
    data_go_kr_service_key = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()
    
    if openai_api_key:
        openai_api_key = openai_api_key.strip()
        os.environ["OPENAI_API_KEY"] = openai_api_key
    else:
        logger.warning("OpenAI API 키 필요")

    if data_go_kr_service_key:
        data_go_kr_service_key = data_go_kr_service_key.strip()
        os.environ["DATA_GO_KR_SERVICE_KEY"] = data_go_kr_service_key
    else:
        logger.warning("Data Portal API 키 필요")

    cfg.OPENAI_API_KEY = openai_api_key
    cfg.DATA_GO_KR_SERVICE_KEY = data_go_kr_service_key
    # cfg.DOCUMENTS_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "documents.db")
    # cfg.EMBEDDINGS_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "embeddings.db")
    # cfg.CHAT_HISTORY_DB_PATH = str(PROJECT_ROOT_PATH / "data" / "chat_history.db")
    # cfg.VECTORSTORE_PATH = str(PROJECT_ROOT_PATH / "data" / "vectorstore")
    # cfg.CONFIG_PATH = CONFIG_PATH
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

# Data Portal API Key 초기화 및 검증
if 'data_go_kr_service_key' not in st.session_state:
    st.session_state.data_go_kr_service_key = os.getenv('DATA_GO_KR_SERVICE_KEY', '').strip()

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

# 임시 디렉토리 생성
# 업로드 파일 저장
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# 파일 업로드 처리 완료 플래그
if 'file_upload_processed' not in st.session_state:
    st.session_state.file_upload_processed = False


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
        'rag_evaluator': RAGEvaluator(api_key="gpt-5"),
    }

if 'processes' not in st.session_state:
    logger.debug("프로세서 초기화 중...")
    st.session_state.processes = init_process()
processes = st.session_state.processes

proc_doc = st.session_state.processes['proc_doc']
proc_emb = st.session_state.processes['proc_emb']
llm_retrieval = st.session_state.processes['llm_retrieval']


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

    # 데이터 포털 업데이트 섹션
    st.subheader("데이터 포털 업데이트")

    with st.expander("날짜 범위 선택", expanded=False):
        col1, col2 = st.columns(2)
        # with col1:
        #     start_date = st.date_input(
        #         "시작 날짜",
        #         value=datetime.now() - timedelta(days=0),
        #         max_value=datetime.now(),
        #         key="update_start_date"
        #     )
        # with col2:
        #     end_date = st.date_input(
        #         "종료 날짜",
        #         value=datetime.now(),
        #         max_value=datetime.now(),
        #         key="update_end_date"
        #     )
        with col1:
            start_datetime = st.text_input(
                "시작 날짜/시간",
                value=(datetime.now() - timedelta(days=7)).strftime("%Y%m%d0000"),
                placeholder="202511191200",
                key="update_start_date",
                help="형식: YYYYMMDDHHMM (예: 202511261430)"
            )
        with col2:
            end_datetime = st.text_input(
                "종료 날짜/시간",
                value=datetime.now().strftime("%Y%m%d2359"),
                placeholder="202511262359",
                key="update_end_date",
                help="형식: YYYYMMDDHHMM (예: 202511262359)"
            )
    
        # 날짜 유효성 검사
        try:
            if len(start_datetime) == 12 and len(end_datetime) == 12:
                start_dt = datetime.strptime(start_datetime, "%Y%m%d%H%M")
                end_dt = datetime.strptime(end_datetime, "%Y%m%d%H%M")
                if start_dt > end_dt:
                    st.error("시작 날짜/시간은 종료 날짜/시간보다 이전이어야 합니다.")
            else:
                st.warning("날짜/시간 형식: YYYYMMDDHHMM (12자리)")
        except ValueError:
            st.error("올바른 날짜/시간 형식이 아닙니다. (예: 202511261430)")

        # data 키 값 입력
        data_key = st.text_input("데이터 포털 API Key",
                             value=st.session_state.data_go_kr_service_key,
                             type="password",
                             key="data_portal_api_key_input"
                             )
        config.DATA_GO_KR_SERVICE_KEY = data_key

    # 업데이트 버튼
    if st.button("데이터 포털 사이트 업데이트", use_container_width=True, key="btn_update_data_portal"):
        # 입력 형식 검증
        if len(start_datetime) != 12 or len(end_datetime) != 12:
            st.error("날짜/시간 형식이 올바르지 않습니다. YYYYMMDDHHMM (12자리)를 입력하세요.")
        else:
            try:
                # 날짜/시간 유효성 검증
                start_dt = datetime.strptime(start_datetime, "%Y%m%d%H%M")
                end_dt = datetime.strptime(end_datetime, "%Y%m%d%H%M")
                
                if start_dt > end_dt:
                    st.error("시작 날짜/시간은 종료 날짜/시간보다 이전이어야 합니다.")
                else:
                    with st.spinner("데이터 포털에서 문서를 가져오는 중..."):
                        logger.info(f"데이터 포털 업데이트 시작: {start_datetime} ~ {end_datetime}")
                        logger.info(f"DATA_GO_KR_SERVICE_KEY: {config.DATA_GO_KR_SERVICE_KEY}")
                        
                        file_hash, result_bool = proc_doc.process_date(config.DATA_GO_KR_SERVICE_KEY, start_datetime, end_datetime)
                    
                    with st.spinner("임베딩 벡터를 생성하는 중..."):
                        proc_emb.sync_with_docs_db(config.OPENAI_API_KEY)
                        proc_emb.vector_manager.summary()
                        logger.debug(f"Data Portal: {file_hash}")
                    
                    st.success(f"데이터 포털 사이트를 성공적으로 업데이트했습니다. ({start_datetime} ~ {end_datetime})")
            except ValueError:
                st.error("올바른 날짜/시간 형식이 아닙니다. (예: 202511261430)")
            except Exception as e:
                logger.error(f"데이터 포털 업데이트 실패: {str(e)}")
                st.error(f"데이터 포털 업데이트에 실패했습니다: {str(e)}")


    # 데이터 통계
    add_section_anchor("analytics-section")
    st.subheader("데이터 통계")

    logger.debug("데이터 통계 로드 시도...")
    doc_stats = dbs['docs'].get_document_stats()
    col1, col2 = st.columns(2)
    vm_result = proc_emb.vector_manager.all_summary()
    
    with col1:
        st.metric("문서 수", f"{doc_stats.get('total_files', 0)}")
        st.metric("페이지 수", f"{doc_stats.get('total_pages', 0)}")
    with col2:
        if vm_result:
            st.metric("청큰 수", f"{vm_result.get('chunk_count', 0)}")
            st.metric("파일 크기", f"{vm_result.get('total_size_mb', 0):.1f} MB")

    st.divider()

    add_section_anchor("document-search-section")
    st.title("업로드할 파일 선택 ")

    # 파일 업로드 버튼 추가
    uploaded_file = st.file_uploader(
        "여기에 파일을 업로드하세요", # 사용자에게 보여줄 텍스트
        type=['pdf', 'hwp'], # 허용할 파일 확장자 목록 (선택 사항) ['csv', 'txt', 'pdf', 'png'...]
        key="file_uploader"
    )

    # 파일이 성공적으로 업로드되었는지 확인하고 처리
    if uploaded_file is not None and not st.session_state.file_upload_processed:
        st.success(f"파일 '{uploaded_file.name}'이(가) 성공적으로 업로드되었습니다.")
    
        with st.spinner(f"파일 '{uploaded_file.name}' 처리 중..."):
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
            with st.spinner("임베딩 벡터를 생성하는 중..."):
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
        
        # session_state에 저장하여 재실행 후에도 유지
        st.session_state.last_file_details = file_details
        st.session_state.last_embedding_summary = summary
    
        # 처리 완료 플래그 설정 (rerun 없이 다음 렌더링에서 자동 반영)
        st.session_state.file_upload_processed = True
    
    elif uploaded_file is None:
        # 파일이 제거되면 플래그 리셋
        st.session_state.file_upload_processed = False
        # 저장된 정보도 리셋
        if 'last_file_details' in st.session_state:
            del st.session_state.last_file_details
        if 'last_embedding_summary' in st.session_state:
            del st.session_state.last_embedding_summary
        st.info("파일을 기다리고 있습니다...")
    else:
        # 이미 처리된 파일
        st.info("파일이 처리되었습니다. 새 파일을 업로드하려면 기존 파일을 제거하세요.")
    
    # 처리 완료된 파일 정보 표시
    if st.session_state.file_upload_processed and 'last_file_details' in st.session_state:
        st.write("---")
        st.subheader("업로드된 파일 상세 정보")
        st.json(st.session_state.last_file_details)
        
        if 'last_embedding_summary' in st.session_state and st.session_state.last_embedding_summary is not None:
            st.write("---")
            st.subheader("임베딩 요약 정보")
            st.json(st.session_state.last_embedding_summary)
    
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
        st.success(f"✅ 언어모델이 '{selected_model}'(으)로 변경되었습니다.")
    else:
        st.info(f"🤖 현재 사용 중: **{st.session_state.current_model}**")

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
    
    # 세션 목록 불러오기
    all_sessions = dbs['chat'].list_sessions()
    recent_sessions = all_sessions[:5]  # 최신 5개
    
    # 전체 세션 selectbox
    if all_sessions:
        st.markdown("**전체 채팅 세션**")
        
        # selectbox 옵션 생성 (세션 이름 + ID)
        session_options = {s['session_name']: s['session_id'] for s in all_sessions}
        session_display_names = list(session_options.keys())
        
        # 현재 세션의 인덱스 찾기
        current_session_name = st.session_state.get('selected_session', '')
        try:
            current_index = session_display_names.index(current_session_name)
        except ValueError:
            current_index = 0
        
        selected_session_name = st.selectbox(
            "세션 선택",
            options=session_display_names,
            index=current_index,
            key="session_selectbox"
        )
        
        # 선택한 세션 정보 가져오기
        selected_session_id = session_options[selected_session_name]
        is_current_session = (selected_session_id == st.session_state.session_id)
        
        # selectbox에서 선택한 세션이 현재 세션과 다르면 자동 전환
        session_switch_key = f"session_switch_{selected_session_id}"
        if session_switch_key not in st.session_state:
            st.session_state[session_switch_key] = False
            
        if not is_current_session and not st.session_state[session_switch_key]:
            st.session_state[session_switch_key] = True
            st.session_state.session_id = selected_session_id
            st.session_state.selected_session = selected_session_name
            
            # DB에서 메시지 불러오기
            db_messages = dbs['chat'].get_session_messages(selected_session_id)
            st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
            st.session_state.session_needs_rename = False
            
            # 세션 타임스탬프 업데이트 (최근 세션 목록 상단에 표시)
            dbs['chat'].update_session_timestamp(selected_session_id)
            
            st.rerun()
        elif is_current_session:
            # 현재 세션으로 복귀하면 플래그 리셋
            st.session_state[session_switch_key] = False
        
        # 삭제 버튼 및 확인
        if st.button("삭제", key="selectbox_delete_session", type="secondary", use_container_width=True):
            # 삭제 확인 상태 저장
            st.session_state.confirm_delete_selectbox = selected_session_id
        
        # 삭제 확인 대화상자
        if st.session_state.get('confirm_delete_selectbox') == selected_session_id:
            st.warning(f"정말로 '{selected_session_name}' 세션을 삭제하시겠습니까?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("예", key="confirm_yes_selectbox", type="primary", use_container_width=True):
                    if dbs['chat'].delete_session(selected_session_id):
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
                        
                        st.session_state.confirm_delete_selectbox = None
                        st.rerun()
            with col_no:
                if st.button("아니오", key="confirm_no_selectbox", use_container_width=True):
                    st.session_state.confirm_delete_selectbox = None
                    st.rerun()
    
    st.markdown("---")
    
    # 최근 5개 세션 Expander 표시
    if recent_sessions:
        st.markdown("**최근 채팅 세션 (5개)**")
        
        for idx, session in enumerate(recent_sessions):
            session_id = session['session_id']
            session_name = session['session_name']
            created_at = session.get('created_at', 'N/A')
            updated_at = session.get('updated_at', 'N/A')
            
            # 메시지 수 계산
            session_messages = dbs['chat'].get_session_messages(session_id)
            message_count = len(session_messages)
            
            # 현재 선택된 세션인지 확인
            is_current = (session_id == st.session_state.session_id)
            
            # Expander 제목 (현재 세션은 표시)
            expander_label = f"{'📌 ' if is_current else ''}{session_name[:30]}{'...' if len(session_name) > 30 else ''}"
            
            with st.expander(expander_label, expanded=False):
                st.markdown(f"**세션 이름**: {session_name}")
                st.markdown(f"**생성 시간**: {created_at}")
                st.markdown(f"**마지막 활동**: {updated_at}")
                st.markdown(f"**메시지 수**: {message_count}개")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 세션 선택 버튼
                    if not is_current:
                        button_key = f"select_session_btn_{session_id}"
                        flag_key = f"select_session_flag_{session_id}"
                        
                        # 이미 처리된 버튼인지 확인
                        if flag_key not in st.session_state:
                            st.session_state[flag_key] = False
                        
                        if st.button("선택", key=button_key, use_container_width=True):
                            # 한 번만 실행되도록 플래그 설정
                            if not st.session_state[flag_key]:
                                st.session_state[flag_key] = True
                                st.session_state.session_id = session_id
                                st.session_state.selected_session = session_name
                                
                                # DB에서 메시지 불러오기
                                db_messages = dbs['chat'].get_session_messages(session_id)
                                st.session_state.messages = [{"role": msg["role"], "content": msg["content"]} for msg in db_messages]
                                st.session_state.session_needs_rename = False
                                
                                # 다음 렌더링에서 플래그 리셋
                                st.rerun()
                        else:
                            # 버튼이 클릭되지 않았으면 플래그 리셋
                            st.session_state[flag_key] = False
                    else:
                        # 현재 세션 - 투명한 버튼 (클릭 불가)
                        st.markdown(
                            f"""
                            <style>
                            div[data-testid="stHorizontalBlock"] button[kind="primary"][disabled] {{
                                opacity: 0;
                                pointer-events: none;
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        st.button("선택", key=f"current_session_btn_{session_id}", 
                                 disabled=True, use_container_width=True, type="primary")
                
                with col2:
                    # 삭제 버튼
                    if st.button("삭제", key=f"delete_session_{session_id}", type="secondary", use_container_width=True):
                        st.session_state[f'confirm_delete_{session_id}'] = True
                
                # 삭제 확인 대화상자
                if st.session_state.get(f'confirm_delete_{session_id}', False):
                    st.warning(f"정말로 '{session_name}' 세션을 삭제하시겠습니까?")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("예", key=f"confirm_yes_{session_id}", type="primary", use_container_width=True):
                            if dbs['chat'].delete_session(session_id):
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
                                
                                st.session_state[f'confirm_delete_{session_id}'] = False
                                st.rerun()
                    with col_no:
                        if st.button("아니오", key=f"confirm_no_{session_id}", use_container_width=True):
                            st.session_state[f'confirm_delete_{session_id}'] = False
                            st.rerun()
        
        st.markdown("---")

    else:
        st.info("저장된 채팅 세션이 없습니다.")
    
    # 세션 통계
    st.subheader("채팅 통계")
    
    chat_stats = dbs['chat'].get_chat_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 대화 수", f"{chat_stats.get('total_sessions', 0)}")
        st.metric("활성 세션", f"{chat_stats.get('active_sessions', 0)}")
    with col2:
        st.metric("총 메시지", f"{chat_stats.get('total_messages', 0)}")
        
        # 평균 대화 길이 계산
        total_sessions = chat_stats.get('total_sessions', 0)
        total_messages = chat_stats.get('total_messages', 0)
        avg_length = total_messages / total_sessions if total_sessions > 0 else 0
        st.metric("평균 대화 길이", f"{avg_length:.1f}개")
    
    # 추가 통계
    st.markdown("**메시지 구성**")
    user_msg = chat_stats.get('user_messages', 0)
    assistant_msg = chat_stats.get('assistant_messages', 0)
    st.text(f"사용자: {user_msg} | AI: {assistant_msg}")

# ----- 2. 메인 영역 구현 -----

# 메인 영역 제목
st.title("문서 검색 시스템")

# 탭 생성 및 선택 추적
selected_tab = st.radio(
    "메뉴 선택",
    ["AI 채팅", "문서 검색"],
    horizontal=True,
    label_visibility="collapsed"
)

# 선택된 탭에 따라 사이드바 스크롤
if selected_tab == "AI 채팅":
    scroll_sidebar_for_tab("AI 채팅")

elif selected_tab == "문서 검색":
    scroll_sidebar_for_tab("문서 검색")

# 채팅 메시지 렌더링 함수
def render_chat_message(role, content):
    """HTML/CSS로 채팅 메시지 렌더링"""
    if role == "user":
        avatar = "🧑"
        align_class = "user"
        bg_color = "#E3F2FD"
        text_color = "#1f77b4"
    else:
        avatar = "🤖"
        align_class = "assistant"
        bg_color = "#F5F5F5"
        text_color = "#333"
    
    # HTML 이스케이프 처리 및 줄바꿈 변환
    content_html = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    
    message_html = f"""
    <div class="chat-message {align_class}">
        <div class="message-avatar">{avatar}</div>
        <div class="message-bubble" style="background-color: {bg_color}; color: {text_color};">
            {content_html}
        </div>
    </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)

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
        render_chat_message(message["role"], message["content"])

    # --------------------------------------------------------------------------------------------
    # 사용자 입력 텍스트 박스 & 전송 버튼 구현
    # --------------------------------------------------------------------------------------------
    if prompt := st.chat_input("여기에 메시지를 입력하세요...", disabled=not api_key_valid):
        
        query = prompt.strip()
        logger.debug(f"사용자 입력: {prompt}")

        st.session_state.messages.append({"role": "user", "content": prompt})
        render_chat_message("user", prompt)
        
        db_messages = dbs['chat'].get_session_messages(st.session_state.session_id)

        file_hash = None
        if not db_messages or len(db_messages) != 0:
            for msg in db_messages:
                str_retrieved_chunks = msg.get('retrieved_chunks', None)
                if str_retrieved_chunks and len(str_retrieved_chunks) > 3:
                    logger.debug(f"retrieved_chunks: {str_retrieved_chunks[:100]}")
                    retrieved_chunks = json.loads(str_retrieved_chunks)
                    if retrieved_chunks and len(retrieved_chunks) > 0:
                        try:
                            file_hash = retrieved_chunks['best_page']['file_hash']
                            logger.debug(f"file_hash 추출됨: {file_hash}")
                        except (KeyError, TypeError) as e:
                            logger.error(f"retrieved_chunks 처리 중 오류 발생: {e}")
                            file_hash = None
                if file_hash is not None:
                    break
        
        metadata = None
        if file_hash is not None and len(file_hash) == 64:
            metadata = {
                'file_hash': file_hash,
            }            
        
        # 벡터 검색
        embedding_result = llm_retrieval.search_page(query, sort_by='page', filter_metadata=metadata)
        print_dic_tree(embedding_result)
        
        # LLM 프로세서 초기화 (선택된 모델 전달)
        current_model = st.session_state.get('current_model', 'gpt-5')
        llm_processor = LLMProcessor(
            session_id=st.session_state.session_id, 
            model=current_model,
            config=config
        )
        logger.info(f"LLM 요청: model={current_model}, query={prompt[:50]}...")
        
        # 스트리밍 응답을 받을 빈 컨테이너 생성
        message_placeholder = st.empty()
        
        # 초기 로딩 메시지 표시
        loading_html = """
        <div class="chat-message assistant">
            <div class="message-avatar">🤖</div>
            <div class="message-bubble" style="background-color: #F5F5F5; color: #999;">
                답변을 준비중입니다...
            </div>
        </div>
        """
        message_placeholder.markdown(loading_html, unsafe_allow_html=True)
        
        full_response = ""
        
        # 스트리밍 응답 처리 (첫 청크부터 즉시 표시)
        for response_chunk in llm_processor.generate_response_stream(query, retrieved_chunks=embedding_result):
            full_response = response_chunk
            # HTML로 실시간 렌더링
            content_html = full_response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            streaming_html = f"""
            <div class="chat-message assistant">
                <div class="message-avatar">🤖</div>
                <div class="message-bubble" style="background-color: #F5F5F5; color: #333;">
                    {content_html}
                </div>
            </div>
            """
            message_placeholder.markdown(streaming_html, unsafe_allow_html=True)
        
        logger.debug(f"result: {full_response[:100]}")
        
        # 스트리밍 완료 후 messages에 추가
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 첫 사용자 메시지로 세션 이름 변경 (rerun 전에 처리)
        if st.session_state.session_needs_rename:
            # 세션 이름을 사용자 메시지로 설정 (최대 50자)
            new_session_name = prompt[:50] + ("..." if len(prompt) > 50 else "")
            dbs['chat'].update_session_name(st.session_state.session_id, new_session_name)
            st.session_state.selected_session = new_session_name
            st.session_state.session_needs_rename = False
            logger.info(f"세션 이름 변경: {new_session_name}")
            st.rerun()  

    # ============================================================================================

# ===== 2번 탭: 문서 검색 =====
elif selected_tab == "문서 검색":
    st.subheader("문서 검색")
    top_k = st.number_input("결과 수", min_value=1, max_value=20, value=5, key="top_k_input")
            
    search_col1, search_col2 = st.columns([5, 1])
    with search_col1:
        search_query = st.text_input("검색어", key="doc_search_input", label_visibility="collapsed", placeholder="검색어를 입력하세요")
    with search_col2:
        search_button = st.button("검색", key="btn_search", use_container_width=True)
    
    if search_button:
        if search_query:
            st.info(f"'{search_query}' 검색 중...")
            embedding_result = llm_retrieval.search(query=search_query, top_k=top_k)
            print_dic_tree(embedding_result)
            
            st.success(f"검색 완료! {len(embedding_result)}개 결과")
            
            # 차트 시각화
            if embedding_result and len(embedding_result) > 0:
                st.subheader("검색 결과 시각화")
                
                # 데이터 준비
                chart_data = []
                for idx, result in enumerate(embedding_result, 1):
                    file_name = result.get('file_name', '파일명 없음')
                    distance = result.get('distance', 0)
                    similarity_pct = max(0, (1.5 - distance) / 1.5 * 100)
                    
                    # 파일명 축약 (너무 길면)
                    display_name = file_name[:50] + '...' if len(file_name) > 50 else file_name
                    
                    chart_data.append({
                        '순위': f"{idx}. {display_name}",
                        '유사도': similarity_pct,
                        '거리': distance
                    })
                
                df = pd.DataFrame(chart_data)
                
                # 문서 분포 계산
                doc_distribution = {}
                for result in embedding_result:
                    file_name = result.get('file_name', '파일명 없음')
                    doc_distribution[file_name] = doc_distribution.get(file_name, 0) + 1
                
                # 탭으로 차트 종류 선택
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["유사도 막대", "유사도 추이", "문서 분포"])
                
                with chart_tab1:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=df['순위'],
                            y=df['유사도'],
                            text=df['유사도'].apply(lambda x: f"<b>{x:.1f}%</b>"),
                            textposition='outside',
                            textfont=dict(size=13, family='Arial Black', weight='bold'),
                            marker=dict(
                                color=df['유사도'],
                                colorscale='Blues',
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text="유사도 (%)", font=dict(size=12, weight='bold')),
                                    tickfont=dict(size=11, weight='bold')
                                )
                            ),
                            hovertemplate='<b>%{x}</b><br>유사도: %{y:.2f}%<br>거리: %{customdata:.4f}<extra></extra>',
                            customdata=df['거리']
                        )
                    ])
                    
                    fig.update_layout(
                        title=dict(text='검색 결과 유사도 분포', font=dict(size=18, weight='bold')),
                        xaxis_title=dict(text='검색 순위', font=dict(size=14, weight='bold')),
                        yaxis_title=dict(text='유사도 (%)', font=dict(size=14, weight='bold')),
                        height=800,
                        hovermode='closest',
                        xaxis=dict(
                            tickangle=-45,
                            tickfont=dict(size=12, family='Arial Black', weight='bold')
                        ),
                        yaxis=dict(
                            tickfont=dict(size=12, weight='bold')
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_tab2:
                    fig = px.line(df, x='순위', y='유사도', markers=True, 
                                  title='검색 결과 유사도 추이',
                                  labels={'순위': '검색 순위', '유사도': '유사도 (%)'},
                                  line_shape='linear')
                    fig.update_traces(marker=dict(size=20, line=dict(width=2, color='white')),
                                     line=dict(width=3))
                    fig.update_layout(
                        height=600,
                        hovermode='x unified',
                        title=dict(text='검색 결과 유사도 추이', font=dict(size=12, weight='bold')),
                        xaxis_title=dict(text='검색 순위', font=dict(size=12, weight='bold')),
                        yaxis_title=dict(text='유사도 (%)', font=dict(size=12, weight='bold')),
                        xaxis=dict(tickfont=dict(size=12, weight='bold')),
                        yaxis=dict(tickfont=dict(size=12, weight='bold'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_tab3:
                    # 문서별 청크 수 차트
                    doc_df = pd.DataFrame(list(doc_distribution.items()), columns=['문서명', '청크 수'])
                    
                    # 문서명 축약
                    doc_df['문서명 (축약)'] = doc_df['문서명'].apply(
                        lambda x: x[:25] + '...' if len(x) > 25 else x
                    )
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=doc_df['문서명 (축약)'],
                            y=doc_df['청크 수'],
                            text=doc_df['청크 수'].apply(lambda x: f"<b>{x}</b>"),
                            textposition='outside',
                            textfont=dict(size=13, family='Arial Black', weight='bold'),
                            marker=dict(
                                color=doc_df['청크 수'],
                                colorscale='Greens',
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text="청크 수", font=dict(size=12, weight='bold')),
                                    tickfont=dict(size=11, weight='bold')
                                )
                            ),
                            hovertemplate='<b>%{x}</b><br>청크 수: %{y}개<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        title=dict(text='문서별 청크 분포', font=dict(size=18, weight='bold')),
                        xaxis_title=dict(text='문서명', font=dict(size=14, weight='bold')),
                        yaxis_title=dict(text='청크 수', font=dict(size=14, weight='bold')),
                        height=800,
                        hovermode='closest',
                        xaxis=dict(
                            tickangle=-45,
                            tickfont=dict(size=11, family='Arial Black', weight='bold')
                        ),
                        yaxis=dict(
                            tickfont=dict(size=12, weight='bold')
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"총 {len(doc_distribution)}개 문서에서 {len(embedding_result)}개 청크 검색됨")
            
            st.subheader("검색 결과")
            # 검색 결과 표시
            for idx, result in enumerate(embedding_result, 1):
                file_name = result.get('file_name', '파일명 없음')
                distance = result.get('distance', 0)
                similarity_pct = max(0, (1.5 - distance) / 1.5 * 100)  # 거리 기반 유사도 변환
                start_page = result.get('start_page', '?')
                end_page = result.get('end_page', '?')
                text_snippet = result.get('text', '')[:200]  # 텍스트 미리보기 200자
                
                # 유사도에 따른 색상 결정
                if similarity_pct >= 70:
                    color = "🟢"  # 높은 유사도 - 초록색
                    quality = "높음"
                elif similarity_pct >= 40:
                    color = "🟡"  # 중간 유사도 - 노란색
                    quality = "중간"
                else:
                    color = "🔴"  # 낮은 유사도 - 빨간색
                    quality = "낮음"
                
                with st.expander(f"{color} [{idx}] {file_name} (페이지 {start_page}-{end_page}) - 관련도: {quality}"):
                    # 유사도 바 시각화
                    bar_length = int(similarity_pct / 2)  # 0-50 범위로 변환
                    bar_color = "🟩" if similarity_pct >= 70 else "🟨" if similarity_pct >= 40 else "🟥"
                    similarity_bar = bar_color * bar_length + "⬜" * (50 - bar_length)
                    
                    st.markdown(f"**유사도**: {similarity_pct:.1f}%")
                    st.markdown(f"{similarity_bar}")
                    st.markdown(f"**거리 값**: {distance:.4f}")
                    st.markdown(f"**내용 미리보기**:")
                    
                    # 검색어 하이라이트 (간단한 구현)
                    highlighted_text = text_snippet
                    if search_query and len(search_query) > 2:
                        # 검색어를 볼드체로 강조
                        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
                        highlighted_text = pattern.sub(f"**{search_query}**", text_snippet)
                    
                    st.markdown(highlighted_text)
        else:
            st.warning("검색어를 입력해주세요.")
# ------- 사이드바  끝 구간 -------

