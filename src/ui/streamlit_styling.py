# -*- coding: utf-8 -*-
"""
Streamlit 스타일링 및 UI 유틸리티 함수
styles.css를 Streamlit 앱에 적용하기 위한 헬퍼 모듈
"""

import streamlit as st
from pathlib import Path


def load_css(css_file_path: str) -> None:
    """
    CSS 파일을 로드하고 Streamlit 앱에 적용합니다.
    
    Args:
        css_file_path (str): CSS 파일의 경로 (절대경로 또는 상대경로)
    
    Example:
        >>> load_css("scripts/오형주/styles/styles.css")
    """
    try:
        css_path = Path(css_file_path)
        
        # 파일이 존재하는지 확인
        if not css_path.exists():
            st.warning(f"⚠️ CSS 파일을 찾을 수 없습니다: {css_file_path}")
            return
        
        # CSS 파일 읽기
        with open(css_path, encoding='utf-8') as f:
            css_content = f.read()
        
        # Streamlit markdown을 사용해 CSS 주입
        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"❌ CSS 로드 중 오류 발생: {str(e)}")


def load_css_from_string(css_content: str) -> None:
    """
    문자열로 된 CSS를 Streamlit 앱에 적용합니다.
    
    Args:
        css_content (str): CSS 코드 문자열
    """
    try:
        st.markdown(
            f"<style>{css_content}</style>",
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"❌ CSS 적용 중 오류 발생: {str(e)}")


def render_metric_card(label: str, value: str, color: str = "primary") -> None:
    """
    커스텀 메트릭 카드를 렌더링합니다.
    
    Args:
        label (str): 메트릭 라벨
        value (str): 메트릭 값
        color (str): 색상 ('primary', 'success', 'warning', 'danger')
    """
    html_content = f"""
    <div class="metric-card" style="border-color: var(--{color}-color);">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_alert(message: str, alert_type: str = "info") -> None:
    """
    얼럿 메시지를 렌더링합니다.
    
    Args:
        message (str): 얼럿 메시지
        alert_type (str): 얼럿 타입 ('info', 'success', 'warning', 'danger')
    """
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "danger": "❌"
    }
    icon = icons.get(alert_type, "ℹ️")
    
    html_content = f"""
    <div class="alert alert-{alert_type}">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_badge(text: str, badge_type: str = "primary") -> None:
    """
    배지를 렌더링합니다.
    
    Args:
        text (str): 배지 텍스트
        badge_type (str): 배지 타입 ('primary', 'success', 'warning', 'danger')
    """
    html_content = f"""
    <span class="badge badge-{badge_type}">{text}</span>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_message_bubble(message: str, is_user: bool = False) -> None:
    """
    채팅 메시지 버블을 렌더링합니다.
    
    Args:
        message (str): 메시지 내용
        is_user (bool): 사용자 메시지 여부 (True: 우측, False: 좌측)
    """
    bubble_class = "message-bubble-user" if is_user else "message-bubble-assistant"
    direction = "message-user" if is_user else "message-assistant"
    
    html_content = f"""
    <div class="message-container {direction}">
        <div class="message-bubble {bubble_class}">
            {message}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_spinner(text: str = "로딩 중...") -> None:
    """
    로더 스피너를 렌더링합니다.
    
    Args:
        text (str): 로더 아래 표시될 텍스트
    """
    html_content = f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="spinner"></div>
        <p style="margin-top: 1rem; color: var(--text-light);">{text}</p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_card_header(title: str, subtitle: str = "") -> None:
    """
    카드 헤더를 렌더링합니다.
    
    Args:
        title (str): 헤더 제목
        subtitle (str): 헤더 부제목 (선택사항)
    """
    subtitle_html = f"<p style='color: var(--text-light); font-size: 0.9rem;'>{subtitle}</p>" if subtitle else ""
    
    html_content = f"""
    <div class="card-header">
        <div>
            <h3 class="card-title">{title}</h3>
            {subtitle_html}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def apply_default_styling() -> None:
    """
    기본 Streamlit 스타일링을 적용합니다.
    
    Example:
        >>> apply_default_styling()
    """
    default_css = """
    /* Streamlit 기본 오버라이드 */
    .stButton > button {
        width: 100%;
    }
    
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > select {
        border-radius: 8px;
    }
    """
    load_css_from_string(default_css)


if __name__ == "__main__":
    # 테스트용 코드
    st.set_page_config(page_title="스타일 테스트", layout="wide")
    
    # CSS 로드 (새 경로로 업데이트)
    load_css("scripts/오형주/ui/styles.css")
    apply_default_styling()
    
    # 테스트 UI
    st.title("📚 Streamlit 스타일 테스트")
    
    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("전체 문서", "245", "primary")
        render_alert("이것은 정보 메시지입니다.", "info")
    
    with col2:
        render_metric_card("처리됨", "198", "success")
        render_alert("이것은 경고 메시지입니다.", "warning")
    
    st.divider()
    
    st.subheader("메시지 버블 테스트")
    render_message_bubble("안녕하세요! 질문이 있습니다.", is_user=True)
    render_message_bubble("안녕하세요! 무엇을 도와드릴까요?", is_user=False)
