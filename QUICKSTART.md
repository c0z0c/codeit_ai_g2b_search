# RAG 시스템 빠른 시작 가이드

## 🎯 5분 안에 시작하기

### 1단계: 패키지 설치

```
gcloud compute firewall-rules list --project=sprint-ai-chunk2-03
gcloud compute ssh spai0433@codeit-ai-g2b-search --zone=us-central1-c --project=sprint-ai-chunk2-03
```

```bash
pip install -r requirements.txt
```

필수 패키지:
- `langchain` - LLM 체인
- `langchain-openai` - OpenAI 연동
- `openai` - OpenAI API
- `faiss-cpu` - 벡터 검색
- `tiktoken` - 토큰 카운팅
- `streamlit` - 웹 UI
- `pymupdf` - PDF 처리

### 2단계: API 키 설정

`.env` 파일을 생성하고 OpenAI API 키를 추가:

```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### 3단계: 더미 데이터 생성 (테스트용)

```bash
python scripts/generate_dummy_simple.py
```

출력 예시:
```
더미 데이터 생성 시작...
문서 1 삽입 완료: 공공데이터_품질관리_가이드라인_2024.pdf
문서 2 삽입 완료: AI_학습용_데이터_구축_지침서_v2.pdf
채팅 세션 생성 완료

=== 데이터 삽입 완료 ===
- 총 파일 수: 2개
- 총 페이지 수: 6페이지
- 총 토큰 수: 678개
- 총 세션 수: 2개
- 총 메시지 수: 4개
```

### 4단계: Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501`이 열립니다.

## 📚 더미 데이터로 테스트하기

더미 데이터에는 다음 문서가 포함되어 있습니다:

### 문서 1: 공공데이터 품질관리 가이드라인
- 총 3페이지
- 내용: 품질관리 체계, 품질 지표, 진단 방법 등

**테스트 질문:**
- "공공데이터 품질관리에서 완전성이란 무엇인가요?"
- "품질 진단은 어떻게 수행하나요?"

### 문서 2: AI 학습용 데이터 구축 지침서
- 총 3페이지
- 내용: 데이터 기획, 수집, 가공, 라벨링 등

**테스트 질문:**
- "AI 학습용 데이터의 라벨링은 어떻게 하나요?"
- "데이터 다양성을 확보하려면 어떻게 해야 하나요?"

## 🔧 실제 PDF 파일 처리하기

### Python 스크립트로 처리

```python
import os
os.environ['OPENAI_API_KEY'] = 'your-api-key'

from src.processors.document_processor import DocumentProcessor
from src.processors.embedding_processor import EmbeddingProcessor

# 1. PDF 처리
processor = DocumentProcessor()
file_hash = processor.process_pdf("path/to/your/document.pdf")

# 2. 임베딩 생성
embedder = EmbeddingProcessor(chunk_size=1000, chunk_overlap=200)
embedding_hash = embedder.process_document(file_hash)

print(f"완료! 이제 Streamlit 앱에서 검색할 수 있습니다.")
```

### 처리 과정 설명

1. **PDF → Markdown 변환**
   - PyMuPDF로 텍스트 추출
   - 페이지별로 구분하여 저장
   - 토큰 수 계산 및 메타데이터 저장

2. **Markdown → 임베딩**
   - LangChain으로 청킹 (기본 1000자, 오버랩 200자)
   - OpenAI API로 임베딩 생성
   - FAISS 인덱스 구축 및 저장

3. **검색 및 답변**
   - 사용자 질의를 임베딩으로 변환
   - FAISS로 유사한 청크 검색 (기본 top-5)
   - 검색 결과를 컨텍스트로 GPT에 전달
   - 답변 생성 및 출처 표시

## 🎨 UI 사용 방법

### 왼쪽 사이드바

1. **OpenAI API Key 입력**
   - `.env`에 설정하지 않았다면 여기서 입력

2. **데이터 통계**
   - 현재 시스템에 저장된 문서, 임베딩, 청크 통계 확인

3. **채팅 세션 관리**
   - `➕ 새 채팅 시작`: 새로운 대화 세션 생성
   - 기존 세션 목록에서 선택하여 이전 대화 재개

### 메인 영역

1. **채팅 인터페이스**
   - 질문 입력창에 질의 작성
   - Enter 또는 전송 버튼 클릭

2. **응답 확인**
   - AI 응답 확인
   - 하단에 참고한 문서 출처 표시

## 📊 데이터 확인하기

SQLite 데이터베이스를 직접 확인하려면:

```bash
# documents.db 확인
sqlite3 data/documents.db "SELECT file_name, total_pages FROM file_info;"

# embeddings.db 확인
sqlite3 data/embeddings.db "SELECT total_chunks, embedding_model FROM embedding_meta;"

# chat_history.db 확인
sqlite3 data/chat_history.db "SELECT session_name, COUNT(*) FROM chat_sessions JOIN chat_messages ON chat_sessions.session_id = chat_messages.session_id GROUP BY session_name;"
```

## 🐛 문제 해결

### "OpenAI API 키를 입력해주세요" 오류
- `.env` 파일에 `OPENAI_API_KEY` 설정 확인
- 또는 Streamlit 사이드바에서 직접 입력

### "임베딩된 문서가 없습니다" 메시지
- `python scripts/generate_dummy_simple.py` 실행하여 더미 데이터 생성
- 또는 실제 PDF 파일 처리

### 패키지 import 오류
```bash
pip install -r requirements.txt --upgrade
```

### FAISS 설치 오류 (Windows)
```bash
pip install faiss-cpu --no-cache-dir
```

## 💡 팁 & 트릭

### 청킹 파라미터 조정
```python
# 짧은 청크 (빠른 검색, 낮은 정확도)
embedder = EmbeddingProcessor(chunk_size=500, chunk_overlap=100)

# 긴 청크 (느린 검색, 높은 정확도)
embedder = EmbeddingProcessor(chunk_size=2000, chunk_overlap=400)
```

### 검색 결과 수 조정
```python
# retrieval.py에서 top_k 파라미터 변경
results = retrieval.search(query="...", embedding_hash="...", top_k=10)
```

### LLM 모델 변경
```python
# gpt-4 사용 (더 정확하지만 느림)
llm = LLMProcessor(model="gpt-4", temperature=0.5)

# gpt-3.5-turbo 사용 (빠르지만 덜 정확)
llm = LLMProcessor(model="gpt-3.5-turbo", temperature=0.7)
```

## 🚀 다음 단계

1. **실제 문서 추가**
   - PDF 파일을 `data/raw/`에 복사
   - Python 스크립트로 처리

2. **UI 커스터마이징**
   - `app.py` 수정하여 원하는 기능 추가
   - Streamlit 테마 변경

3. **성능 최적화**
   - 청킹 파라미터 튜닝
   - 임베딩 모델 변경
   - 캐싱 추가

4. **배포**
   - Streamlit Cloud에 배포
   - Docker 컨테이너화
   - 클라우드 서버에 호스팅

## 📖 더 자세한 정보

- [README.md](README.md) - 전체 프로젝트 개요
- [설계서](docs/RAG_기반_PEP_문서_처리_시스템_설계서.md) - 상세 설계 문서
- [개발 가이드](docs/RAG_기반_PEP_문서_처리_시스템_개발.md) - 개발 가이드라인

---

**문의사항이 있으시면 Issue를 생성해주세요!** 🙏
