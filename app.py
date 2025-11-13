# -*- coding: utf-8 -*-
"""
RAG 기반 PEP 문서 검색 시스템
Streamlit UI
"""

import streamlit as st
import os
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from src.db import DocumentsDB, ChatHistoryDB
from src.llm.retrieval import Retrieval
from src.llm.llm_processor import LLMProcessor
from src.config import get_config

# 페이지 설정
st.set_page_config(
    page_title="RAG 기반 PEP 문서 검색 시스템",
    page_icon="📚",
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

# === 사이드바 ===
with st.sidebar:
    st.title("⚙️ 설정")

    # API 키 입력
    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="OpenAI API 키를 입력하세요"
    )
    if api_key:
        st.session_state.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key

    st.divider()

    # 데이터 통계
    # !!!DocumentsDB 클래스가 없기 때문에 ImportError 또는 AttributeError 가 납니다. (11/11 추가) ***까지!!!
    st.subheader("📊 데이터 통계")
    #doc_stats = dbs['docs'].get_document_stats()
    #embedding_stats = dbs['embeddings'].get_embedding_stats()

    #해결 방법 (UI만 테스트하고 싶을 때) 만약 UI만 보고 싶다면, dbs['docs'] 관련 부분을 더미로 바꾸면 됩니다.
    # 더미 데이터로 교체
    doc_stats = {'total_files': 0, 'total_pages': 0}
    embedding_stats = {'total_embeddings': 0, 'total_chunks': 0}
    #                   *** 여기까지 UI 테스트용 더미 데이터 추가 (추후삭제) ***

    col1, col2 = st.columns(2)
    with col1:
        st.metric("문서 수", f"{doc_stats['total_files']}")
        st.metric("페이지 수", f"{doc_stats['total_pages']}")
    with col2:
        st.metric("임베딩 수", f"{embedding_stats['total_embeddings']}")
        st.metric("청크 수", f"{embedding_stats['total_chunks']}")

    st.divider()

    # 세션 관리
    st.subheader("💬 채팅 세션")

    # 새 세션 생성
    if st.button("➕ 새 채팅 시작", use_container_width=True):
        session_name = f"채팅 {len(dbs['chat'].get_all_sessions()) + 1}"
        new_session_id = dbs['chat'].create_session(session_name)
        st.session_state.session_id = new_session_id
        st.session_state.messages = []
        st.rerun()

    # 기존 세션 목록
    sessions = dbs['chat'].get_recent_sessions(limit=10)
    if sessions:
        st.write("최근 세션:")
        for session in sessions:
            is_current = session['session_id'] == st.session_state.session_id
            label = f"{'🟢' if is_current else '⚪'} {session['session_name']}"
            if st.button(label, key=session['session_id'], use_container_width=True):
                st.session_state.session_id = session['session_id']
                # 세션 메시지 로드
                messages = dbs['chat'].get_session_messages(session['session_id'])
                st.session_state.messages = [
                    {"role": msg['role'], "content": msg['content']}
                    for msg in messages
                ]
                st.rerun()

# === 메인 영역 ===
st.title("📚 RAG 기반 PEP 문서 검색 시스템")
st.markdown("공공데이터 및 AI 관련 문서를 검색하고 질문하세요.")

# 세션이 없으면 생성
if st.session_state.session_id is None:
    st.session_state.session_id = dbs['chat'].create_session()

# 채팅 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # API 키 확인
    if not st.session_state.api_key:
        st.error("⚠️ OpenAI API 키를 입력해주세요!")
        st.stop()

    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # DB에 사용자 메시지 저장
    dbs['chat'].add_message(st.session_state.session_id, "user", prompt)

    # 어시스턴트 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            # 임베딩 가져오기 (첫 번째 임베딩 사용)
            all_embeddings = dbs['embeddings'].get_all_embeddings()

            if not all_embeddings:
                response = "임베딩된 문서가 없습니다. 먼저 문서를 업로드하고 임베딩을 생성해주세요."
            else:
                embedding_hash = all_embeddings[0]['embedding_hash']

                # 검색 수행 (Config의 TOP_K_FINAL 사용)
                retrieval = Retrieval(config=config)
                retrieved_chunks = retrieval.search(
                    query=prompt,
                    embedding_hash=embedding_hash,
                    top_k=config.TOP_K_FINAL,
                    api_key=st.session_state.api_key
                )

                # LLM 응답 생성
                llm = LLMProcessor(config=config)
                response = llm.generate_response(
                    query=prompt,
                    retrieved_chunks=retrieved_chunks,
                    api_key=st.session_state.api_key
                )

                # 출처 표시
                if retrieved_chunks:
                    response += "\n\n---\n**📄 참고 문서:**\n"
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        file_name = chunk.get('file_name', 'unknown')
                        similarity = chunk.get('similarity', 0)
                        response += f"\n{i}. {file_name} (유사도: {similarity:.2%})"

        st.markdown(response)

    # 어시스턴트 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

    # DB에 어시스턴트 메시지 저장
    retrieved_info = [
        {
            "file_name": chunk.get('file_name'),
            "similarity": float(chunk.get('similarity', 0))
        }
        for chunk in retrieved_chunks
    ] if 'retrieved_chunks' in locals() and retrieved_chunks else None

    dbs['chat'].add_message(
        st.session_state.session_id,
        "assistant",
        response,
        retrieved_chunks=retrieved_info
    )

# 푸터
st.divider()
st.caption("💡 Tip: 왼쪽 사이드바에서 OpenAI API 키를 입력하고, 새 채팅을 시작하거나 기존 채팅을 선택할 수 있습니다.")
