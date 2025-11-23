# -*- coding: utf-8 -*-
"""
사이드바 자동 스크롤 기능 모듈
탭 전환 시 사이드바를 특정 섹션으로 자동 스크롤
"""

import streamlit as st
import streamlit.components.v1 as components


def scroll_to_section(section_id: str, unique_key: str = None):
    """
    사이드바를 특정 섹션으로 스크롤
    
    Args:
        section_id: 스크롤할 대상 섹션의 HTML ID
        unique_key: 각 탭을 구분하기 위한 고유 키
    """
    # 고유 ID 생성
    scroll_id = f"scroll_{section_id}_{unique_key}" if unique_key else f"scroll_{section_id}"
    
    # HTML iframe을 사용하여 JavaScript 실행
    # 고유성을 위해 HTML에 데이터 속성 추가
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script>
            window.onload = function() {{
                console.log('[{scroll_id}] Script loaded, attempting to scroll to: {section_id}');
                
                function findAndScroll() {{
                    try {{
                        const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                        if (!sidebar) {{
                            return false;
                        }}
                        
                        const target = sidebar.querySelector('#{section_id}');
                        if (!target) {{
                            return false;
                        }}
                        
                        console.log('[{scroll_id}] Found target, scrolling...');
                        target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                        return true;
                    }} catch (e) {{
                        console.error('[{scroll_id}] Error:', e);
                        return false;
                    }}
                }}
                
                // 즉시 1번만 실행
                setTimeout(function() {{
                    if (findAndScroll()) {{
                        console.log('[{scroll_id}] Scroll successful');
                    }}
                }}, 100);
            }};
        </script>
    </head>
    <body data-scroll-id="{scroll_id}"></body>
    </html>
    """
    
    components.html(html_code, height=0)


def add_section_anchor(section_id: str, title: str = None):
    """
    사이드바 섹션에 HTML 앵커 추가
    
    Args:
        section_id: 섹션 식별용 HTML ID
        title: 섹션 제목 (있으면 st.title로 표시)
    """
    st.markdown(f'<div id="{section_id}"></div>', unsafe_allow_html=True)
    if title:
        st.title(title)


# 탭별 사이드바 섹션 매핑
TAB_SECTION_MAP = {
    "AI 채팅": "chat-session-section",
    "문서 검색": "document-search-section",
    "분석 및 통계": "analytics-section"
}


def scroll_sidebar_for_tab(tab_name: str):
    """
    탭 이름에 따라 사이드바를 해당 섹션으로 스크롤
    
    Args:
        tab_name: 탭 이름 ("AI 채팅", "문서 검색", "분석 및 통계")
    """
    section_id = TAB_SECTION_MAP.get(tab_name)
    if section_id:
        # 탭 이름을 고유 키로 사용
        scroll_to_section(section_id, unique_key=tab_name.replace(" ", "_"))
