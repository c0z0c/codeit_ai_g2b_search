# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from getpass import getpass
import os
import json # JSON 직렬화를 위한 임포트 (튜플 변환 시 내부적으로 필요)

try:
    # 1. LangChain Core 및 통합 모듈 (v0.2.x 환경)
    from langchain_openai import ChatOpenAI
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate # ChatPromptTemplate 추가
    from langchain.chains import ConversationChain 
    from langchain.memory import ConversationSummaryMemory 

    LANGCHAIN_AVAILABLE_LLM = True
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print("LangChain import error:", e)

# 프로젝트 내부 모듈 임포트
from src.llm.prompts.prompt_loader import PromptLoader
# from src.db.chat_history_db import ChatHistoryDB # 이 모듈이 외부에서 임포트된다고 가정
from src.config import get_config
from src.utils.logging_config import get_logger

# ChatHistoryDB 클래스는 이 파일에 없으므로, 해당 파일에서 임포트되어야 합니다.
# 예시:
class ChatHistoryDB:
    # ... (실제 ChatHistoryDB 클래스 내용)
    def create_session(self): return "test_session_123"
    def add_message(self, session_id, role, content, retrieved_chunks): pass


class LLMProcessor:
    """
    LLMProcessor + ChatHistoryDB 저장 버전
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config=None,
        api_key: Optional[str] = None,
        prompt_file: Optional[str] = None,
        session_id: Optional[str] = None,
        use_db: bool = True,
    ):
        # Config & Logger
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        self.model_name = model or self.config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else self.config.OPENAI_TEMPERATURE

        # ⭐️ Temperature 강제 변경 로직 (GPT-5-mini 대응)
        if self.model_name == "gpt-5-mini" and self.temperature == 0.0:
            self.logger.warning("gpt-5-mini는 temperature=0.0을 지원하지 않습니다. 1.0으로 강제 변경합니다.")
            self.temperature = 1.0

        # API Key 입력
        if api_key is None or not api_key.strip():
            api_key_input = getpass("OpenAI API Key를 입력하세요: ").strip()
            api_key = api_key_input or None
        if not api_key:
            raise ValueError("API Key가 필요합니다.")
        os.environ["OPENAI_API_KEY"] = api_key

        # LangChain 초기화
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        else:
            raise ImportError("LangChain이 설치되지 않음")

        # Prompt 로더
        self.prompts = PromptLoader()

        # ChatHistoryDB 초기화
        self.use_db = use_db
        if self.use_db:
            # self.db = ChatHistoryDB() # 실제 코드에서는 이 라인을 사용해야 함
            self.db = ChatHistoryDB() # 임시로 클래스 인스턴스 생성
            if session_id:
                self.session_id = session_id
            else:
                self.session_id = self.db.create_session()
        else:
            self.db = None
            self.session_id = "no-db-session"

        # 한국어 요약용 프롬프트
        KOREAN_SUMMARY_PROMPT = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="""
기존 대화 요약:
{summary}

새로운 대화:
{new_lines}

위 내용을 바탕으로 **한국어로** 업데이트된 대화 요약을 작성하세요.
최종 요약:
""",
        )

        # 메모리 + 히스토리
        self.memory = ConversationSummaryMemory(
            llm=self.llm, max_token_limit=1500, prompt=KOREAN_SUMMARY_PROMPT
        )
        self.chat_history = InMemoryChatMessageHistory()

        base_chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
        self.conversation_chain = RunnableWithMessageHistory(
            base_chain,
            # lambda session_id: self.db.get_history(session_id) if self.use_db else self.chat_history, # DB 사용 시 실제 로직
            lambda session_id: self.chat_history, # 현재는 InMemory를 사용하도록 유지
            input_messages_key="input",
            history_messages_key="history",
        )
        self.logger.info(f"LLMProcessor 초기화 완료 (model={self.model_name}, temperature={self.temperature})")
        print(f"LLMProcessor (DB 저장 활성화: {self.use_db}) initialised.")

    # -----------------------------------------------
    # ⭐ JSON 직렬화 오류 방지 헬퍼 함수 추가
    # -----------------------------------------------
    def _convert_tuple_keys_to_str(self, data: Any) -> Any:
        """JSON 직렬화를 위해 튜플 키를 문자열로 변환하는 재귀 함수."""
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if isinstance(k, tuple):
                    new_key = str(k)
                else:
                    new_key = k
                new_dict[new_key] = self._convert_tuple_keys_to_str(v)
            return new_dict
        elif isinstance(data, list):
            return [self._convert_tuple_keys_to_str(item) for item in data]
        else:
            return data

    # -----------------------------------------------
    # ⭐ 검색 컨텍스트 구성 (기존 그대로)
    # -----------------------------------------------
    def _build_context_from_chunks(self, chunks, max_chunks=None):
        if max_chunks:
            chunks = chunks[:max_chunks]
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                self.config.CONTEXT_FORMAT.format(
                    index=i,
                    file_name=chunk.get("file_name", ""),
                    chunk_text=chunk.get("chunk_text") or chunk.get("text") or "",
                )
            )
        return "\n\n".join(parts)

    def _build_context_from_pages(self, pages, max_chunks=None):
        if max_chunks:
            pages = pages[:max_chunks]
        parts = []
        for i, page in enumerate(pages, 1):
            parts.append(
                f"[문서 {i}] {page.get('file_name')} (페이지 {page.get('page_number')}, score={page.get('score'):.4f})\n{page.get('text')}"
            )
        return "\n\n".join(parts)

    # -----------------------------------------------
    # ⭐ DB 저장 기능 + LLM 응답 생성 (수정된 로직)
    # -----------------------------------------------
    def generate_response(self, query: str, retrieved_chunks: Any) -> str:

        # ⭐️ 1. DB 저장 전 튜플 키를 문자열로 변환하여 JSON 오류 방지
        processed_chunks = self._convert_tuple_keys_to_str(retrieved_chunks)

        # 2. User 메시지 DB 저장
        if self.use_db:
            self.db.add_message(
                session_id=self.session_id,
                role="user",
                content=query,
                retrieved_chunks=processed_chunks, # 변환된 청크 전달
            )

        # 3. 컨텍스트 구성
        max_chunks = 5
        if isinstance(processed_chunks, dict) and "pages" in processed_chunks:
            context = self._build_context_from_pages(processed_chunks["pages"], max_chunks)
        elif isinstance(processed_chunks, list):
            context = self._build_context_from_chunks(processed_chunks, max_chunks)
        else:
            context = self.config.NO_CONTEXT_MESSAGE

        # 4. 프롬프트 로딩 (기존 코드에서 복원된 로직)
        prompt_text = None
        # 1) PromptLoader.templates 딕셔너리 우선 검사
        if hasattr(self.prompts, "templates") and isinstance(self.prompts.templates, dict):
            for k in ("rag_prompt_template", "template", "default"):
                v = self.prompts.templates.get(k)
                if v: prompt_text = v; break
            if not prompt_text:
                first = next(iter(self.prompts.templates.values()), None)
                if isinstance(first, dict):
                    prompt_text = first.get("rag_prompt_template") or first.get("template") or first.get("default")
                elif isinstance(first, str):
                    prompt_text = first
        # 2) load_template 메서드가 있으면 파일명으로 로드 시도
        if not prompt_text and hasattr(self.prompts, "load_template"):
            try:
                tpl = self.prompts.load_template("prompt_temp_v1")
                if isinstance(tpl, dict):
                    prompt_text = tpl.get("rag_prompt_template") or tpl.get("template") or tpl.get("default")
            except Exception: pass
        # 3) 설정 파일(fallback)
        prompt_text = prompt_text or getattr(self.config, "RAG_PROMPT_TEMPLATE", None)

        if not prompt_text:
            self.logger.error("RAG 프롬프트 텍스트를 찾을 수 없습니다.")
            raise KeyError("'rag_prompt_template' 또는 사용 가능한 프롬프트 텍스트를 찾을 수 없습니다.")

        # 5. LLM 호출 입력 및 실행
        full_input = f"Context:\n{context}\n\nQuery:\n{query}"
        
        try:
            response = self.conversation_chain.invoke(
                {"input": full_input},
                config={"configurable": {"session_id": self.session_id or "default"}},
            )
            self.logger.info("LLM 응답 생성 완료")
            answer = (
                response["response"] if isinstance(response, dict) and "response" in response else str(response)
            )
        except Exception as e:
            self.logger.error(f"응답 생성 중 오류 발생: {e}")
            answer = f"오류 발생: {e}"

        # 6. Assistant 메시지 DB 저장
        if self.use_db:
            self.db.add_message(
                session_id=self.session_id,
                role="assistant",
                content=answer,
                retrieved_chunks=None,
            )

        return answer

    # -----------------------------------------------
    def get_memory_summary(self) -> str:
        """현재까지의 대화 요약 내용 반환"""
        try:
            vars = self.memory.load_memory_variables({})
            return vars.get("history", "(요약 없음)")
        except Exception:
            self.logger.error("메모리 요약 조회 중 오류 발생.")
            return "(요약 조회 실패)"