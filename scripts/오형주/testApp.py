# -*- coding: utf-8 -*-
"""
문서 검색 시스템 (PDF, HWP, DOCX 등)
Streamlit UI 초안 구현 및 테스트
"""

import streamlit as st
import os
from openai import OpenAI
from pathlib import Path
import sys
# sys.path.insert(0, str(Path(__file__).parent))
# 프로젝트 루트를 sys.path에 추가 (scripts/오형주 → 프로젝트 루트로 이동)
project_root = Path(__file__).resolve().parents[2]  # scripts/오형주 → 2단계 상위 = 프로젝트 루트
sys.path.insert(0, str(project_root))

from src.db import DocumentsDB, ChatHistoryDB
from src.config import get_config

from styles.streamlit_styling import load_css, apply_default_styling

# Streamlit 페이지 설정 - 반드시 첫 번째 Streamlit 명령
st.set_page_config(
    page_title="문서 검색 시스템",
    layout="wide",
)

# CSS 로드 및 스타일 적용 - set_page_config 다음
load_css("scripts/오형주/styles/styles.css")  # CSS 파일 로드
apply_default_styling()  # 기본 Streamlit 오버라이드

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
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"}]
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

# ----- 사이드바 구현 구간 ----- (1번 ~ 6번)

# 1. st.sidebar 컨텍스트를 사용하여 사이드바 내용 정의
with st.sidebar:
    st.title("설정 및 세션")
    
    # 2. OpenAI API Key 입력 위젯
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    
    # API 키가 유효하게 입력되었는지 확인하는 플래그
    api_key_valid = False 
    if openai_api_key:
        st.success("API Key 입력 완료!")
        api_key_valid = True 
    else:
        st.warning("API Key를 입력해주세요.")
    
    st.markdown("---") 


    # 데이터 통계
    st.subheader("데이터 통계")
    try:
        doc_stats = dbs['docs'].get_document_stats()
        embedding_stats = dbs['embeddings'].get_embedding_stats()
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

    st.title("업로드할 파일 선택 ") # 지금은 PDF파일만 업로드하고 추후 다양한 포맷 지원 예정
    # 파일 업로드 버튼 추가
    uploaded_file = st.file_uploader(
        "여기에 파일을 업로드하세요", # 사용자에게 보여줄 텍스트
        type=['pdf'] # 허용할 파일 확장자 목록 (선택 사항) ['csv', 'txt', 'pdf', 'png'...]
    )

    # 파일이 성공적으로 업로드되었는지 확인하고 처리
    if uploaded_file is not None:
        st.success(f"파일 '{uploaded_file.name}'이(가) 성공적으로 업로드되었습니다.")
    
        # 예시: 업로드된 파일의 타입과 크기 표시
        file_details = {
            "파일 이름": uploaded_file.name,
            "파일 타입": uploaded_file.type,
            "파일 크기 (바이트)": uploaded_file.size
        }
        st.write("---")
        st.subheader("업로드된 파일 상세 정보")
        st.json(file_details)
    
        # 파일을 읽고 싶다면 (예: CSV 파일)
        # import pandas as pd
        # if uploaded_file.type == "text/csv":
        #     df = pd.read_csv(uploaded_file)
        #     st.dataframe(df.head())
    else:
        st.info("파일을 기다리고 있습니다...")

    # 3 & 4. 데이터/임베딩 업데이트 버튼
    st.title("데이터 및 임베딩 업데이트")
    if st.button("데이터 업데이트 (A API)", use_container_width=True, key="btn_data_update", disabled=not api_key_valid):
        # 실제 로직에서는 API 키 유효성 검사를 통과해야 버튼 동작
        st.info("데이터 업데이트 시작...")
        st.success("데이터 업데이트 완료!")
        
    if st.button("임베딩 업데이트 (B API)", use_container_width=True, key="btn_embedding_update", disabled=not api_key_valid):
        st.info("새 데이터를 기반으로 임베딩 벡터를 갱신하고 있습니다...")
        st.success("임베딩 업데이트 완료!")

    # 5 & 6. 세션 관리
    st.title("채팅 세션 관리")
    
    # 새로운 세션 생성
    if st.button("새 세션 생성", use_container_width=True, key="btn_new_session"):
        session_name = "새 채팅 세션"
        new_session_id = dbs['chat'].create_session(session_name)
        st.session_state.session_id = new_session_id
        st.session_state.messages = []
        st.session_state.selected_session = session_name
        st.rerun()

# ----- 2. 메인 영역 구현 -----

# 7. 메인 영역 제목
st.title("AI 채팅 도우미")

st.subheader(f"현재 세션: {st.session_state.selected_session}")

# 세션이 없으면 생성
if st.session_state.session_id is None:
    st.session_state.session_id = dbs['chat'].create_session()

# 8. 채팅 메시지 표시 컨테이너
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 9. 사용자 입력 텍스트 박스 & 10. 전송 버튼 구현
if prompt := st.chat_input("여기에 메시지를 입력하세요...", disabled=not api_key_valid):
# 에러시 api 확인하는 코드 추가

    # 1. 사용자 입력 저장 및 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # DB에 사용자 메시지 저장
    dbs['chat'].add_message(st.session_state.session_id, "user", prompt)

    # 2. API 키 유효성을 다시 확인하고, 유효한 경우에만 API 호출
    if api_key_valid:
        try:
            # 2-1. OpenAI 클라이언트 초기화
            client = OpenAI(api_key=openai_api_key)

            # 2-2. API 호출을 위한 메시지 리스트 준비
            # Streamlit 세션 상태의 messages를 OpenAI API 형식에 맞게 사용합니다.
            messages_for_api = st.session_state.messages

            # 2-3. AI 응답 생성 (스트리밍 사용)
            with st.chat_message("assistant"):
                # chat.completions.create 호출
                stream = client.chat.completions.create(
                    model="gpt-3.5-turbo", # 이 버전이 개발 초기 단계나 소규모 앱에서 비용 효율성으로 사용됨
                    messages=messages_for_api,
                    stream=True,
                )
                
                # Streamlit의 st.write_stream을 사용하여 응답을 실시간으로 화면에 출력
                response = st.write_stream(stream)
            
            # 2-4. AI 응답 저장
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"OpenAI API 호출 중 오류가 발생했습니다: {e}")
            st.session_state.messages.pop() # 오류 발생 시 마지막 사용자 메시지 제거

    else:
        st.error("OpenAI API Key를 먼저 입력해주세요.")
    
# 실행 예시: Streamlit을 실행하면 왼쪽에 "설정 및 세션" 제목이 있는 사이드바가 보입니다.
# ------- 사이드바  끝 구간 -------

