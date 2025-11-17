# -*- coding: utf-8 -*-
"""
문서 검색 시스템 (PDF, HWP, DOCX 등)
Streamlit UI 초안 구현 및 테스트
"""

import streamlit as st
import os
import time
import random
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
""" # 더미데이터 연동 끝나면 주석제거
# 세션 상태 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"}]
if 'api_key' not in st.session_state:
    st.session_state.api_key = config.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY', '')
"""

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

# 더미 세션 목록
session_list = ["새 채팅 세션", "2025년 11월 10일: 데이터 분석 요청", "2025년 11월 11일: Streamlit 문의",
                "2025년 11월 12일: 프로젝트 계획 논의", "2025년 11월 13일: 기술 지원 요청",
                "2025년 11월 14일: 제품 피드백"]

# ----------------- [1] 더미 데이터 및 함수 정의 -----------------

# 더미 채팅 히스토리 로드 함수
def load_dummy_history(session_name):
    """선택된 세션에 따른 더미 채팅 기록을 반환합니다."""
    if "Streamlit 문의" in session_name:
        return [
            {"role": "assistant", "content": "안녕하세요! Streamlit 레이아웃 구성에 대한 문의를 해주셨네요."},
            {"role": "user", "content": "사이드바와 메인 영역을 나누는 방법이 궁금해요."},
            {"role": "assistant", "content": "사이드바는 `st.sidebar`로, 메인 영역은 일반적인 `st.` 명령어로 구현합니다."}
        ]
    elif "데이터 분석 요청" in session_name:
        return [
            {"role": "assistant", "content": "이전 세션에서 요청하신 데이터 분석 결과입니다. 주요 지표는 다음과 같습니다: `매출 증가율 15%`, `고객 이탈율 5%`"},
            {"role": "user", "content": "이탈율 5%는 좋은 수치인가요?"}
        ]
    else: # 새 채팅 세션
        return [{"role": "assistant", "content": "안녕하세요! 저는 AI 채팅 어시스턴트입니다. 무엇을 도와드릴까요?"}]

# 더미 LLM 응답 스트리밍 시뮬레이션 함수
def stream_dummy_response(prompt):
    """GPT API 스트리밍을 시뮬레이션하여 응답을 실시간으로 반환하는 제너레이터."""
    
    dummy_text = (
        f"사용자님이 '{prompt[:30]}...'라고 질문하셨습니다. 이에 대한 더미 LLM 응답을 스트리밍합니다." 
        "이것은 OpenAI API 연동 없이 UI/UX 테스트를 위해 만들어진 응답입니다. "
        "Streamlit의 `st.write_stream` 기능을 사용하면 실제 API 응답처럼 타이핑되는 효과를 연출할 수 있습니다."
        "예를 들어, 이 응답은 여러 문장으로 나누어져 순차적으로 출력될 것입니다."
)    
    # 문장 단위로 분리하여 스트리밍 효과 시뮬레이션
    sentences = dummy_text.split('. ')
    
    for sentence in sentences:
        yield sentence + ('.' if not sentence.endswith('.') else '') + " "
        time.sleep(random.uniform(0.05, 0.1)) # 실제 네트워크 지연처럼 보이게 약간의 딜레이 추가


if "messages" not in st.session_state:
    st.session_state.messages = load_dummy_history("새 채팅 세션")
    st.session_state.selected_session = "새 채팅 세션" # 세션 상태에 현재 선택된 세션 이름 저장

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

# ----- 사이드바 구현 구간 ----- (1번 ~ 6번)

# 1. st.sidebar 컨텍스트를 사용하여 사이드바 내용 정의
with st.sidebar:
    st.title("설정 및 세션")
    
    # 2. OpenAI API Key 입력 위젯
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    
    # API 키가 유효하게 입력되었는지 확인하는 플래그
    #api_key_valid = False # 주석제거
    api_key_valid = True # 더미테스트용 임시 설정
    if openai_api_key:
        st.success("API Key 입력 완료!")
        #api_key_valid = True # 주석제거
    else:
        st.warning("API Key를 입력해주세요.")
        st.info("더미 데이터 모드 동작") # 더미테스트 후 삭제
    
    st.markdown("---") 


    st.title("업로드할 파일 선택 ")
    # 파일 업로드 버튼 추가
    uploaded_file = st.file_uploader(
        "여기에 파일을 업로드하세요", # 사용자에게 보여줄 텍스트
        type=['csv', 'txt', 'pdf', 'png'] # 허용할 파일 확장자 목록 (선택 사항)
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
        #st.info("데이터 업데이트 시작...")
        #st.success("데이터 업데이트 완료!")
        with st.spinner("데이터베이스에서 최신 데이터를 가져오는 중..."):
            time.sleep(1.5)
        st.success("데이터 업데이트 완료! (더미 데이터)")
        
    if st.button("임베딩 업데이트 (B API)", use_container_width=True, key="btn_embedding_update", disabled=not api_key_valid):
        st.info("새 데이터를 기반으로 임베딩 벡터를 갱신하고 있습니다...")
        with st.spinner("임베딩 모델(B API) 호출 및 벡터 DB 저장 중..."):
            time.sleep(2)
        st.success("임베딩 업데이트 완료! (더미 벡터 100개 생성)")
    
        # 더미 검색 결과 표시 (RAG 테스트용)
        st.markdown("**--- 더미 검색 결과 ---**")
        st.markdown("문서 A:** Streamlit은 Python 기반의 웹 앱 프레임워크입니다.")
        st.markdown("문서 B:** 사이드바는 st.sidebar를 사용하여 쉽게 구현할 수 있습니다.")
        st.markdown("---")

    # 5 & 6. 세션 관리
    st.title("채팅 세션 관리")
    selected_session = st.selectbox("채팅 세션 선택", options=session_list, index=0)
    
    if st.button("새 세션 생성", use_container_width=True, key="btn_new_session"):
        st.session_state.messages = [{"role": "assistant", "content": "새로운 대화를 시작합니다."}]
        st.info("새로운 채팅 세션을 시작합니다.")
        # 세션 상태를 초기화 후 앱을 새로고침
        st.rerun()

# ----- 2. 메인 영역 구현 -----

# 7. 메인 영역 제목
st.title("AI 채팅 어시스턴트")
st.subheader(f"현재 세션: {st.session_state.selected_session}")

# 8. 채팅 메시지 표시 컨테이너
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 9. 사용자 입력 텍스트 박스 & 10. 전송 버튼 구현
if prompt := st.chat_input("여기에 메시지를 입력하세요...", disabled=not api_key_valid):
    
    # A. 사용자 입력 저장 및 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 응답 생성 (더미 스트리밍 시뮬레이션)
    with st.chat_message("assistant"):
        # 더미 LLM 응답 스트리밍 시뮬레이션 함수 사용
        response = st.write_stream(stream_dummy_response(prompt))
    
    # 3. AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
""" # 더미데이터 삭제 후 주석제거
    # B. API 키 유효성을 다시 확인하고, 유효한 경우에만 API 호출
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
"""
    
# 실행 예시: Streamlit을 실행하면 왼쪽에 "설정 및 세션" 제목이 있는 사이드바가 보입니다.
# ------- 사이드바  끝 구간 -------

