"""
더미 데이터 생성 스크립트

3개의 가상 PEP 문서 데이터를 생성하여 데이터베이스에 삽입합니다.
실제 한국 행정문서 스타일의 마크다운 콘텐츠를 생성합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hashlib
from datetime import datetime

from src.db import DocumentsDB, EmbeddingsDB, ChatHistoryDB
from src.utils import get_logger
import tiktoken

logger = get_logger(__name__)


class DummyDataGenerator:
    """더미 데이터 생성 클래스"""

    def __init__(self):
        self.docs_db = DocumentsDB()
        self.embeddings_db = EmbeddingsDB()
        self.chat_db = ChatHistoryDB()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.tokenizer.encode(text))

    def generate_sample_documents(self) -> list:
        """
        3개의 샘플 PEP 문서 생성

        Returns:
            문서 정보 리스트
        """
        documents = [
            {
                "file_name": "공공데이터_품질관리_가이드라인_2024.pdf",
                "pages": [
                    {
                        "page_number": 1,
                        "content": """# 공공데이터 품질관리 가이드라인

## 제1장 총칙

### 제1조 (목적)
이 가이드라인은 「공공데이터의 제공 및 이용 활성화에 관한 법률」 제21조에 따라 공공데이터의 품질관리에 필요한 사항을 규정함을 목적으로 한다.

### 제2조 (적용범위)
이 가이드라인은 공공기관이 보유·관리하는 모든 공공데이터에 적용한다.

### 제3조 (정의)
이 가이드라인에서 사용하는 용어의 뜻은 다음과 같다:
1. "공공데이터"란 공공기관이 법령 등에서 정하는 목적을 위하여 생성 또는 취득하여 관리하고 있는 데이터를 말한다.
2. "품질관리"란 공공데이터의 정확성, 완전성, 일관성, 유효성을 확보하기 위한 활동을 말한다.
3. "메타데이터"란 공공데이터를 효율적으로 관리하고 이용하기 위한 데이터에 대한 데이터를 말한다.""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 2,
                        "content": """## 제2장 품질관리 체계

### 제4조 (품질관리 조직)
① 공공기관의 장은 공공데이터 품질관리를 위한 전담조직을 구성·운영하여야 한다.
② 전담조직은 다음 각 호의 업무를 수행한다:
1. 공공데이터 품질관리 계획 수립
2. 공공데이터 품질 진단 및 개선
3. 공공데이터 품질 교육 및 컨설팅

### 제5조 (품질관리 절차)
공공데이터 품질관리는 다음의 절차에 따라 수행한다:
1. **계획 수립**: 품질관리 목표 및 범위 설정
2. **데이터 진단**: 품질 지표에 따른 현황 분석
3. **개선 실행**: 발견된 품질 문제 해결
4. **모니터링**: 품질 개선 효과 측정 및 피드백

### 제6조 (품질 지표)
공공데이터의 품질은 다음의 지표로 측정한다:
- 정확성(Accuracy): 데이터가 실제 값과 일치하는 정도
- 완전성(Completeness): 필수 항목이 누락 없이 기록된 정도
- 일관성(Consistency): 데이터가 규칙과 형식에 부합하는 정도
- 적시성(Timeliness): 데이터가 최신 상태로 유지되는 정도""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 3,
                        "content": """## 제3장 품질 진단

### 제7조 (진단 대상)
① 모든 공공데이터는 연 1회 이상 품질 진단을 실시하여야 한다.
② 국민 활용도가 높은 데이터는 분기별 진단을 실시할 수 있다.

### 제8조 (진단 방법)
품질 진단은 다음의 방법으로 수행한다:
1. **자동 진단**: 시스템을 활용한 데이터 검증
2. **수동 진단**: 전문가의 직접 검토
3. **이용자 피드백**: 데이터 이용자 의견 수렴

### 제9조 (진단 결과 활용)
① 진단 결과는 등급으로 분류하여 관리한다.
② 진단 결과에 따라 개선 계획을 수립하고 이행하여야 한다.

| 등급 | 점수 범위 | 조치사항 |
|------|-----------|----------|
| 우수 | 90점 이상 | 현행 유지 |
| 양호 | 80~89점 | 경미한 개선 |
| 보통 | 70~79점 | 개선 권고 |
| 미흡 | 70점 미만 | 즉시 개선 |""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 4,
                        "content": """## 제4장 데이터 표준화

### 제10조 (표준화 원칙)
공공데이터는 다음의 원칙에 따라 표준화되어야 한다:
1. 상호운용성 확보
2. 재사용성 증진
3. 일관성 유지

### 제11조 (메타데이터 표준)
① 모든 공공데이터는 표준 메타데이터를 포함하여야 한다.
② 필수 메타데이터 항목은 다음과 같다:
- 데이터명, 제공기관, 생성일자, 갱신주기
- 데이터 형식, 라이선스, 담당자 정보

### 제12조 (부칙)
이 가이드라인은 2024년 1월 1일부터 시행한다.

---

**[문서 정보]**
- 발행일: 2024년 1월
- 발행기관: 행정안전부
- 문의: 공공데이터정책과 (02-2100-1234)""",
                        "is_empty": False,
                    },
                ],
            },
            {
                "file_name": "AI_학습용_데이터_구축_지침서_v2.pdf",
                "pages": [
                    {
                        "page_number": 1,
                        "content": """# AI 학습용 데이터 구축 지침서

## 개요

본 지침서는 인공지능(AI) 학습용 데이터를 구축함에 있어 품질 높은 데이터셋을 확보하기 위한 절차와 기준을 제시한다.

## 1. 데이터 기획

### 1.1 목적 정의
- AI 모델의 학습 목표 명확화
- 활용 분야 및 예상 사용자 정의
- 성능 목표 수립 (정확도, F1 스코어 등)

### 1.2 데이터 설계
구축할 데이터의 유형과 규모를 결정한다:
- **이미지 데이터**: 해상도, 포맷, 수량
- **텍스트 데이터**: 언어, 도메인, 문장 수
- **음성 데이터**: 샘플링 레이트, 화자 수, 시간
- **영상 데이터**: 프레임 레이트, 길이, 해상도""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 2,
                        "content": """## 2. 데이터 수집

### 2.1 수집 방법
1. **크롤링**: 웹사이트에서 자동 수집
2. **API 활용**: 공개 API를 통한 수집
3. **직접 생성**: 자체 제작 또는 촬영
4. **구매/제휴**: 외부 데이터 확보

### 2.2 수집 시 주의사항
- 저작권 및 개인정보보호법 준수
- 데이터 출처 명확히 기록
- 수집 일자 및 버전 관리

### 2.3 데이터 다양성 확보
AI 모델의 일반화 성능을 위해 다양한 데이터를 수집한다:
- 연령, 성별, 지역 등 인구통계학적 다양성
- 시간대, 계절, 환경 등 상황적 다양성
- Edge case 및 예외 상황 포함

| 수집 방법 | 장점 | 단점 |
|----------|------|------|
| 크롤링 | 대량 수집 가능 | 품질 불균일 |
| API | 안정적 수집 | 제약사항 존재 |
| 직접 생성 | 품질 통제 용이 | 시간/비용 소요 |""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 3,
                        "content": """--- [빈페이지] ---""",
                        "is_empty": True,
                    },
                    {
                        "page_number": 4,
                        "content": """## 3. 데이터 가공

### 3.1 전처리
수집된 원시 데이터를 AI 학습에 적합한 형태로 변환한다:
1. **정제(Cleaning)**: 노이즈 제거, 결측치 처리
2. **정규화(Normalization)**: 값의 범위 통일
3. **변환(Transformation)**: 형식 변환, 인코딩

### 3.2 라벨링(Annotation)
데이터에 정답 레이블을 부여하는 과정:
- **분류 라벨**: 카테고리 할당
- **객체 탐지**: 바운딩 박스 지정
- **세그멘테이션**: 픽셀 단위 라벨링
- **개체명 인식**: 텍스트 내 엔티티 태깅

### 3.3 라벨링 품질 관리
① 라벨링 가이드라인 작성 및 교육
② 작업자 간 일치도(Inter-annotator Agreement) 측정
③ 검수 프로세스 운영 (Double Annotation)
④ 주기적인 샘플링 검증""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 5,
                        "content": """## 4. 데이터 검증

### 4.1 품질 검증 지표
- **정확성**: 라벨이 실제와 일치하는 비율
- **일관성**: 동일 기준 적용 여부
- **완전성**: 필수 속성 누락 여부
- **유효성**: 데이터 형식 준수 여부

### 4.2 데이터셋 분할
학습, 검증, 테스트 세트로 분할:
```
Train Set: 70-80%
Validation Set: 10-15%
Test Set: 10-15%
```

### 4.3 편향성 검사
데이터셋의 편향(Bias)을 분석하고 완화:
- 클래스 불균형 확인
- 인구통계학적 편향 점검
- 샘플링 편향 분석

## 5. 데이터 배포

### 5.1 메타데이터 작성
- 데이터 명세서(Data Sheet)
- 라벨링 가이드라인
- 사용 라이선스
- 구축 과정 기록

### 5.2 데이터 포맷
표준 포맷으로 제공:
- 이미지: JPEG, PNG
- 라벨: JSON, XML, COCO format
- 메타데이터: CSV, JSON

---

**[참고 문헌]**
1. AI Hub 데이터 구축 가이드라인
2. ISO/IEC 25024 데이터 품질 표준
3. FAIR 데이터 원칙""",
                        "is_empty": False,
                    },
                ],
            },
            {
                "file_name": "개인정보_비식별화_조치_가이드라인.pdf",
                "pages": [
                    {
                        "page_number": 1,
                        "content": """# 개인정보 비식별화 조치 가이드라인

## I. 목적 및 배경

본 가이드라인은 개인정보의 안전한 활용을 위한 비식별 조치 기준을 제시하고, 빅데이터 분석 등에 활용할 수 있도록 지원하는 것을 목적으로 한다.

### 1. 필요성
- 개인정보 활용과 보호의 균형
- 빅데이터 산업 활성화 지원
- 개인정보 유출 리스크 최소화

### 2. 법적 근거
- 「개인정보 보호법」 제18조(개인정보의 목적 외 이용·제공 제한)
- 「신용정보의 이용 및 보호에 관한 법률」
- 「정보통신망 이용촉진 및 정보보호 등에 관한 법률」

## II. 비식별화의 개념

### 1. 정의
비식별화란 개인정보의 일부 또는 전부를 삭제하거나 대체함으로써 다른 정보와 결합하여도 특정 개인을 알아볼 수 없도록 하는 조치를 말한다.""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 2,
                        "content": """### 2. 비식별화와 익명화의 구분

| 구분 | 비식별화 | 익명화 |
|------|----------|--------|
| 정의 | 추가 정보 없이 특정 개인을 식별할 수 없는 상태 | 어떠한 방법으로도 개인 식별이 불가능한 상태 |
| 재식별 가능성 | 재식별 가능성 존재 | 재식별 불가능 |
| 개인정보 해당 여부 | 개인정보에 해당 가능 | 개인정보 아님 |
| 적용 법규 | 개인정보보호법 적용 | 개인정보보호법 미적용 |

### 3. 비식별화 대상 정보
① 직접 식별자: 이름, 주민등록번호, 연락처 등
② 준식별자: 나이, 성별, 주소, 직업 등
③ 민감정보: 건강정보, 금융정보 등

## III. 비식별화 기법

### 1. 가명처리(Pseudonymization)
개인정보의 일부를 다른 값으로 대체하는 기법

**적용 예시:**
- 이름 → 익명ID (홍길동 → USER_001)
- 주민등록번호 → 해시값 변환
- 전화번호 뒷자리 → 난수 치환""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 3,
                        "content": """### 2. 총계처리(Aggregation)
개인 단위 데이터를 그룹 단위 통계값으로 변환

**예시:**
```
개인 소득: 3,500만원, 4,200만원, 3,800만원
↓
평균 소득: 3,833만원
```

### 3. 데이터 삭제(Data Reduction)
식별 위험이 높은 속성을 삭제

**삭제 대상:**
- 고유 식별자 (주민등록번호, 여권번호)
- 불필요한 민감정보
- 재식별 위험이 높은 속성

### 4. 데이터 범주화(Data Suppression)
값의 정확도를 낮추어 일반화

**예시:**
- 상세 주소 → 시/도 단위
- 정확한 나이 → 연령대 (25세 → 20대)
- 구체적 직업 → 직업군 (소프트웨어 개발자 → IT 종사자)

### 5. 데이터 마스킹(Data Masking)
데이터 일부를 * 등으로 치환

**예시:**
- 이름: 홍길동 → 홍*동
- 전화번호: 010-1234-5678 → 010-****-5678
- 이메일: hong@example.com → h***@example.com""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 4,
                        "content": """## IV. 비식별 조치 절차

### 1단계: 사전 검토
- 비식별 목적 및 활용 계획 수립
- 처리 대상 개인정보 현황 파악
- 법적 요구사항 검토

### 2단계: 비식별 조치
- 적절한 비식별 기법 선택
- 비식별 조치 실행
- 비식별 결과 검증

### 3단계: 적정성 평가
다음 기준으로 적정성을 평가한다:
① **k-익명성**: 동일한 속성 조합을 가진 레코드가 k개 이상
② **l-다양성**: 민감 속성이 l개 이상의 다양한 값 보유
③ **t-근접성**: 민감 속성 분포가 전체 분포와 유사

### 4단계: 사후 관리
- 주기적인 재식별 위험 모니터링
- 추가 정보 공개 시 재평가
- 비식별 조치 이력 기록 및 보관

## V. 재식별 금지 의무

비식별 정보를 처리하는 자는 재식별을 시도해서는 안 되며, 재식별된 정보는 즉시 파기하여야 한다.""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 5,
                        "content": """## VI. 비식별 조치 사례

### 사례 1: 의료 데이터
**원본 데이터:**
- 이름: 김철수
- 주민등록번호: 850101-1234567
- 주소: 서울시 강남구 테헤란로 123
- 진단명: 고혈압

**비식별 조치 후:**
- 환자ID: PT_A00123 (가명처리)
- 생년월일: 1985년 (연도만 유지)
- 지역: 서울시 강남구 (상세주소 삭제)
- 진단명: 심혈관계 질환 (범주화)

### 사례 2: 모바일 이용 데이터
**원본 데이터:**
```json
{
  "user_id": "user@email.com",
  "age": 28,
  "location": "서울시 서초구 반포동",
  "purchase_amount": 45000
}
```

**비식별 조치 후:**
```json
{
  "user_id": "USR_8A3F92",
  "age_group": "20-29",
  "location": "서울시",
  "purchase_range": "40000-50000"
}
```

## VII. 부칙

본 가이드라인은 2023년 7월 1일부터 시행한다.

---

**[참고자료]**
- 개인정보 비식별 조치 가이드라인 (개인정보보호위원회, 2016)
- GDPR Article 4 (Definitions)
- ISO/IEC 29100 Privacy Framework""",
                        "is_empty": False,
                    },
                    {
                        "page_number": 6,
                        "content": """--- [빈페이지] ---""",
                        "is_empty": True,
                    },
                ],
            },
        ]

        return documents

    def calculate_file_hash(self, file_name: str, content: str) -> str:
        """파일 해시 계산 (파일명 + 콘텐츠 기반)"""
        data = f"{file_name}_{content}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    def insert_documents(self):
        """문서 데이터 삽입"""
        documents = self.generate_sample_documents()

        for doc in documents:
            # 전체 콘텐츠 결합
            full_content = "\n\n".join(
                page["content"] for page in doc["pages"]
            )

            # 파일 해시 계산
            file_hash = self.calculate_file_hash(doc["file_name"], full_content)

            # 파일 정보 계산
            total_pages = len(doc["pages"])
            total_chars = len(full_content)
            total_tokens = self.count_tokens(full_content)
            file_size = len(full_content.encode("utf-8"))

            # file_info 삽입
            self.docs_db.insert_file_info(
                file_hash=file_hash,
                file_name=doc["file_name"],
                total_pages=total_pages,
                file_size=file_size,
                total_chars=total_chars,
                total_tokens=total_tokens,
            )

            # page_data 삽입
            for page in doc["pages"]:
                page_content = page["content"]
                page_tokens = self.count_tokens(page_content)

                self.docs_db.insert_page_data(
                    file_hash=file_hash,
                    page_number=page["page_number"],
                    markdown_content=page_content,
                    token_count=page_tokens,
                    is_empty=page["is_empty"],
                )

            logger.info(
                f"문서 삽입 완료: {doc['file_name']} "
                f"({total_pages} 페이지, {total_tokens} 토큰)"
            )

    def insert_sample_chat_sessions(self):
        """샘플 채팅 세션 생성"""
        # 세션 1: 공공데이터 품질관리 관련 질문
        session1_id = self.chat_db.create_session("공공데이터 품질관리 문의")
        self.chat_db.add_message(
            session1_id,
            "user",
            "공공데이터 품질관리에서 '완전성' 지표는 무엇을 의미하나요?",
        )
        self.chat_db.add_message(
            session1_id,
            "assistant",
            "완전성(Completeness)은 필수 항목이 누락 없이 기록된 정도를 의미합니다. "
            "공공데이터 품질관리 가이드라인 제6조에 따르면, 완전성은 데이터 품질을 측정하는 "
            "주요 지표 중 하나로, 모든 필수 데이터 항목이 빠짐없이 입력되었는지를 평가합니다.",
            retrieved_chunks=[
                {
                    "chunk_id": 1,
                    "file_name": "공공데이터_품질관리_가이드라인_2024.pdf",
                    "page_range": "2",
                    "similarity": 0.92,
                }
            ],
        )

        # 세션 2: AI 데이터 구축 관련 질문
        session2_id = self.chat_db.create_session("AI 학습 데이터 라벨링")
        self.chat_db.add_message(
            session2_id, "user", "AI 학습용 데이터의 라벨링 품질을 어떻게 관리하나요?"
        )
        self.chat_db.add_message(
            session2_id,
            "assistant",
            "라벨링 품질 관리는 다음 4가지 방법으로 수행됩니다:\n"
            "1. 라벨링 가이드라인 작성 및 교육\n"
            "2. 작업자 간 일치도(Inter-annotator Agreement) 측정\n"
            "3. 검수 프로세스 운영 (Double Annotation)\n"
            "4. 주기적인 샘플링 검증\n\n"
            "이를 통해 정확성, 일관성, 완전성을 확보할 수 있습니다.",
            retrieved_chunks=[
                {
                    "chunk_id": 5,
                    "file_name": "AI_학습용_데이터_구축_지침서_v2.pdf",
                    "page_range": "4",
                    "similarity": 0.89,
                }
            ],
        )

        # 세션 3: 비식별화 기법 질문
        session3_id = self.chat_db.create_session("개인정보 비식별화 방법")
        self.chat_db.add_message(
            session3_id, "user", "데이터 마스킹과 가명처리의 차이는 무엇인가요?"
        )
        self.chat_db.add_message(
            session3_id,
            "assistant",
            "데이터 마스킹과 가명처리는 다음과 같이 구분됩니다:\n\n"
            "**데이터 마스킹**: 데이터 일부를 '*' 등으로 치환하는 방법입니다. "
            "예: '홍길동' → '홍*동', '010-1234-5678' → '010-****-5678'\n\n"
            "**가명처리**: 개인정보의 일부를 다른 값으로 완전히 대체하는 방법입니다. "
            "예: '홍길동' → 'USER_001', 주민등록번호 → 해시값\n\n"
            "가명처리가 더 강력한 비식별화 기법이며, 재식별 위험이 낮습니다.",
            retrieved_chunks=[
                {
                    "chunk_id": 8,
                    "file_name": "개인정보_비식별화_조치_가이드라인.pdf",
                    "page_range": "2-3",
                    "similarity": 0.94,
                }
            ],
        )

        logger.info("샘플 채팅 세션 3개 생성 완료")

    def generate_all(self):
        """모든 더미 데이터 생성"""
        logger.info("=== 더미 데이터 생성 시작 ===")

        # 1. 문서 데이터 삽입
        logger.info("[1/2] 문서 데이터 삽입 중...")
        self.insert_documents()

        # 2. 채팅 세션 삽입
        logger.info("[2/2] 채팅 세션 삽입 중...")
        self.insert_sample_chat_sessions()

        # 통계 출력
        logger.info("\n=== 데이터 삽입 완료 ===")
        doc_stats = self.docs_db.get_document_stats()
        chat_stats = self.chat_db.get_chat_stats()

        logger.info(f"\n[문서 통계]")
        logger.info(f"- 총 파일 수: {doc_stats['total_files']}개")
        logger.info(f"- 총 페이지 수: {doc_stats['total_pages']}페이지")
        logger.info(f"- 총 토큰 수: {doc_stats['total_tokens']:,}개")
        logger.info(f"- 총 크기: {doc_stats['total_size_mb']}MB")

        logger.info(f"\n[채팅 통계]")
        logger.info(f"- 총 세션 수: {chat_stats['total_sessions']}개")
        logger.info(f"- 총 메시지 수: {chat_stats['total_messages']}개")
        logger.info(f"- 사용자 메시지: {chat_stats['user_messages']}개")
        logger.info(f"- 어시스턴트 메시지: {chat_stats['assistant_messages']}개")


if __name__ == "__main__":
    generator = DummyDataGenerator()
    generator.generate_all()
