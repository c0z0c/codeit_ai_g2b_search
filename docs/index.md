---
layout: default
title: "코드잇 AI 4기 3팀 중급 프로젝트 - doc"
description: "코드잇 AI 4기 3팀 중급 프로젝트 - doc"
date: 2025-11-08
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# 🏥 코드잇 AI 4기 3팀 중급 프로젝트

### 📱 프로젝트 개요
**목표**: RAG(Retrieval-Augmented Generation) 시스템을 구축하여 복잡한 형태의 기업 및 정부 제안요청서(Request For Proposal, RFP) 내용을 효과적으로 추출하고 요약하여 필요한 정보를 제공하는 서비스를 개발하는 것을 목표로 합니다.

### 👥 팀원

| 역할          | 담당자       | 핵심 업무                                      |
|---------------|--------------|-----------------------------------------------|
| 데이터 엔지니어 |        | 문서 수집 및 원본 전처리 (PDF/HWP → Markdown 변환 및 DB 저장) |
| 머신러닝 엔지니어 |        | 임베딩 처리 (Markdown → 벡터 임베딩 및 FAISS 저장)         |
| AI 리서처      |        | LLM 기반 정보 추출 및 요약 시스템                      |
| 프론트엔드 엔지니어 |        | Streamlit UI 개발 및 통합                           |

### 📅 프로젝트 기간
**2025년 11월 10일 ~ 2025년 11월 28일**

## 📝 협업일지

팀원별 개발 과정 및 학습 내용을 기록한 협업일지입니다.

- [김명환 협업일지 (Project Manager)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/김명환/)
- [신승목 협업일지 (Data Engineer)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/신승일/)
- [오형주 협업일지 (Model Architect)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/오형주/)
- [이민규 협업일지 (Experimentation Lead)](https://c0z0c.github.io/codeit_ai_g2b_search/협업일지/이민규/)
- 이현재 협업일지
- [팀 회의록](https://c0z0c.github.io/codeit_ai_g2b_search/회의록/)

<script>

// 폴더 정보 가져오기 함수
function getFolderInfo(folderName) {
    folderName = (folderName || '').toString().replace(/^\/+|\/+$/g, '');
    // 폴더명에 따른 아이콘과 설명 (가나다순 정렬)
    const folderMappings = {
        '감성데이타': { icon: '📊', desc: 'AI HUB 감성 데이타셋' },
        '경구약제 이미지 데이터(데이터 설명서, 경구약제 리스트)': { icon: '📊', desc: '데이터 설명서' },
        '경구약제이미지데이터': { icon: '💊', desc: '약물 데이터' },
        '멘토': { icon: '👨‍🏫', desc: '멘토 관련 자료' },
        '백업': { icon: '💾', desc: '백업 파일들' },
        '발표자료': { icon: '📊', desc: '발표 자료' },
        '셈플': { icon: '📂', desc: '샘플 파일들' },
        '스터디': { icon: '📒', desc: '학습 자료' },
        '스프린트미션_완료': { icon: '✅', desc: '완료된 스프린트 미션들' },
        '스프린트미션_작업중': { icon: '🚧', desc: '진행 중인 미션들' },
        '실습': { icon: '🔬', desc: '실습 자료' },
        '위클리페이퍼': { icon: '📰', desc: '주간 학습 리포트' },
        '테스트': { icon: '🧪', desc: '테스트 파일들' },
        '협업일지': { icon: '📓', desc: '협업일지' },
        'doc': { icon: '📋', desc: '팀 doc' },
        'AI 모델 환경 설치가이드': { icon: '⚙️', desc: '설치 가이드' },
        'assets': { icon: '🎨', desc: '정적 자원' },
        'image': { icon: '🖼️', desc: '이미지 파일들' },
        'Learning': { icon: '📚', desc: '학습 자료' },
        'Learning Daily': { icon: '📅', desc: '일일 학습 기록' },
        'md': { icon: '📝', desc: 'Markdown 문서' }
    };
    return folderMappings[folderName] || { icon: '📁', desc: '폴더' };
}

function getFileInfo(extname) {
  switch(extname.toLowerCase()) {
    case '.ipynb':
      return { icon: '📓', type: 'Colab' };
    case '.py':
      return { icon: '🐍', type: 'Python' };
    case '.md':
      return { icon: '📝', type: 'Markdown' };
    case '.json':
      return { icon: '⚙️', type: 'JSON' };
    case '.zip':
      return { icon: '📦', type: '압축' };
    case '.png':
    case '.jpg':
    case '.jpeg':
      return { icon: '🖼️', type: '이미지' };
    case '.csv':
      return { icon: '📊', type: '데이터' };
    case '.pdf':
      return { icon: '📄', type: 'PDF' };
    case '.docx':
      return { icon: '�', type: 'Word' };
    case '.pptx':
      return { icon: '📊', type: 'PowerPoint' };
    case '.xlsx':
      return { icon: '📈', type: 'Excel' };
    case '.hwp':
      return { icon: '📄', type: 'HWP' };
    case '.txt':
      return { icon: '📄', type: 'Text' };
    case '.html':
      return { icon: '🌐', type: 'HTML' };
    default:
      return { icon: '📄', type: '파일' };
  }
}

{% assign cur_dir = "/" %}
{% include cur_files.liquid %}
{% include page_values.html %}
{% include page_files_table.html %}
{% include page_folders_tree.html %}

</script>

<div class="file-grid">
  <!-- 파일 목록이 JavaScript로 동적 생성됩니다 -->
</div>

---

## 폴더목록

<div class="folder-grid">
  <!-- 폴더 목록이 JavaScript로 동적 생성됩니다 -->
</div>


---

<div class="navigation-footer">
  <a href="{{- site.baseurl -}}/" class="nav-button home">
    <span class="nav-icon">🏠</span> 홈으로
  </a>
  <a href="https://github.com/c0z0c/codeit_ai_g2b_search" target="_blank">
    <span class="link-icon">📱</span> GitHub 저장소
  </a>
</div>