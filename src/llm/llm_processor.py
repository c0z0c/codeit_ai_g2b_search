# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from getpass import getpass
import os
import json
import hashlib  # ⭐ 파일 해시를 위한 추가!

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationSummaryMemory

    LANGCHAIN_AVAILABLE_LLM = True
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print("LangChain import error:", e)

from src.llm.prompts.prompt_loader import PromptLoader
from src.config import get_config
from src.utils.logging_config import get_logger

try:
    from src.db.chat_history_db import ChatHistoryDB
except ImportError:
    # 모듈이 없는 경우를 대비하여 더미 클래스를 유지할 수도 있지만,
    # 여기서는 정상 경로를 가정하고 기존 더미 클래스는 제거합니다.
    print("경고: src.db.chat_history_db를 임포트할 수 없습니다.")
    class ChatHistoryDB:
        def create_session(self): return "test_session_123"
        def add_message(self, session_id, role, content, retrieved_chunks): pass


class LLMProcessor:
    """
    LLMProcessor + ChatHistoryDB 저장 버전 (파일 해시 패치 적용)
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
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        self.model_name = model or self.config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else self.config.OPENAI_TEMPERATURE

        # gpt-5-mini 에서 temperature=0 금지
        if self.model_name == "gpt-5-mini" and self.temperature == 0.0:
            self.logger.warning("gpt-5-mini는 temperature=0을 지원하지 않음 → 1.0으로 자동 변경")
            self.temperature = 1.0

        # # API KEY 등록
        if api_key is not None:
            os.environ["OPENAI_API_KEY"] = api_key
            
        if os.environ["OPENAI_API_KEY"].strip() == "":
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않음")

        # LangChain LLM
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        else:
            raise ImportError("LangChain이 설치되지 않음")

        # Prompt loader
        self.prompts = PromptLoader()

        # DB
        self.use_db = use_db
        if self.use_db:
            self.db = ChatHistoryDB()
            self.session_id = session_id or self.db.create_session()
        else:
            self.db = None
            self.session_id = "no-db-session"

        # 한국어 요약 프롬프트
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

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=1500,
            prompt=KOREAN_SUMMARY_PROMPT,
        )
        self.chat_history = InMemoryChatMessageHistory()

        base_chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
        self.conversation_chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        self.logger.info(f"LLMProcessor 초기화 완료 (model={self.model_name})")
        print(f"LLMProcessor (DB 저장={self.use_db}) ready.")

    # -----------------------------------------------------------------------
    # ⭐ JSON 직렬화 오류 해결 (튜플 키 변환)
    # -----------------------------------------------------------------------
    def _convert_tuple_keys_to_str(self, data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_key = str(k) if isinstance(k, tuple) else k
                new_dict[new_key] = self._convert_tuple_keys_to_str(v)
            return new_dict
        elif isinstance(data, list):
            return [self._convert_tuple_keys_to_str(x) for x in data]
        return data

    # -----------------------------------------------------------------------
    # 컨텍스트 구성
    # -----------------------------------------------------------------------
    def _build_context_from_chunks(self, chunks, max_chunks=None):
        if max_chunks:
            chunks = chunks[:max_chunks]
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                self.config.CONTEXT_FORMAT.format(
                    index=i,
                    file_name=c.get("file_name", ""),
                    chunk_text=c.get("chunk_text") or c.get("text") or "",
                )
            )
        return "\n\n".join(parts)

    def _build_context_from_pages(self, pages, max_chunks=None):
        if max_chunks:
            pages = pages[:max_chunks]
        parts = []
        for i, p in enumerate(pages, 1):
            parts.append(
                f"[문서 {i}] {p.get('file_name')} "
                f"(페이지 {p.get('page_number')}, score={p.get('score'):.4f})\n{p.get('text')}"
            )
        return "\n\n".join(parts)

    # -----------------------------------------------------------------------
    # ⭐ main: 응답 생성 + DB 저장 (해시 적용)
    # -----------------------------------------------------------------------
    def generate_response(self, query: str, retrieved_chunks: Any) -> str:

        # 1) 튜플 키 → 문자열 변환
        processed = self._convert_tuple_keys_to_str(retrieved_chunks)

        # ⭐ 2) 파일 해시 패치 적용
        processed = self._add_file_hash(processed)

        # 3) DB 저장
        if self.use_db:
            self.db.add_message(
                session_id=self.session_id,
                role="user",
                content=query,
                retrieved_chunks=processed,
            )

        # 4) 컨텍스트 만들기
        max_chunks = 5
        if isinstance(processed, dict) and "pages" in processed:
            context = self._build_context_from_pages(processed["pages"], max_chunks)
        elif isinstance(processed, list):
            context = self._build_context_from_chunks(processed, max_chunks)
        else:
            context = self.config.NO_CONTEXT_MESSAGE

        # 5) 프롬프트 읽기
        prompt_text = None
        if hasattr(self.prompts, "templates"):
            for k in ("rag_prompt_template", "template", "default"):
                v = self.prompts.templates.get(k)
                if v: prompt_text = v; break

            if not prompt_text:
                first = next(iter(self.prompts.templates.values()), None)
                if isinstance(first, dict):
                    prompt_text = (
                        first.get("rag_prompt_template")
                        or first.get("template")
                        or first.get("default")
                    )
                elif isinstance(first, str):
                    prompt_text = first

        if not prompt_text and hasattr(self.prompts, "load_template"):
            try:
                tpl = self.prompts.load_template("prompt_temp_value")
                if isinstance(tpl, dict):
                    prompt_text = (
                        tpl.get("rag_prompt_template")
                        or tpl.get("template")
                        or tpl.get("default")
                    )
            except Exception:
                pass

        prompt_text = prompt_text or getattr(self.config, "RAG_PROMPT_TEMPLATE", None)
        if not prompt_text:
            raise KeyError("RAG 프롬프트 template을 찾을 수 없습니다.")

        # 6) LLM 호출
        full_input = f"Context:\n{context}\n\nQuery:\n{query}"

        try:
            response = self.conversation_chain.invoke(
                {"input": full_input},
                config={"configurable": {"session_id": self.session_id}},
            )
            answer = response["response"] if isinstance(response, dict) else str(response)
        except Exception as e:
            self.logger.error(f"응답 오류: {e}")
            answer = f"오류 발생: {e}"

        # 7) assistant 메시지 DB 저장
        if self.use_db:
            self.db.add_message(
                session_id=self.session_id,
                role="assistant",
                content=answer,
                retrieved_chunks=None,
            )

        return answer

    # -----------------------------------------------------------------------
    def get_memory_summary(self) -> str:
        try:
            data = self.memory.load_memory_variables({})
            return data.get("history", "(요약 없음)")
        except Exception:
            return "(요약 조회 실패)"