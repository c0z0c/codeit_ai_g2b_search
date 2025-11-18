# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional
from getpass import getpass
import os

try:
    # 1. OpenAI 통합 모듈
    from langchain_openai import ChatOpenAI
    
    # 2. Core 모듈
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    
    # 3. Memory 및 Chain 모듈 (0.2.x 버전의 표준 경로)
    # ⭐️ .memory 대신 .chains에서 임포트 시도 (v0.2.x 구조)
    from langchain.chains import ConversationChain 
    
    # ⭐️ Memory는 v0.2.x에서 langchain 패키지에 위치하지만, 경로 충돌 방지 위해 유지
    # Note: v0.2.x에서 ConversationSummaryMemory는 langchain.memory에 있습니다.
    from langchain.memory import ConversationSummaryMemory 
    
    # 4. 기타 Chain (LLMChain은 사용하지 않으므로 제거)
    
    LANGCHAIN_AVAILABLE_LLM = True
    # LLMProcessor에서 사용하는 변수 명 통일
    LANGCHAIN_AVAILABLE = LANGCHAIN_AVAILABLE_LLM 
    
except ImportError as e:
    LANGCHAIN_AVAILABLE_LLM = False
    LANGCHAIN_AVAILABLE = LANGCHAIN_AVAILABLE_LLM 
    print("\n=============================================")
    print(f"🚨 LangChain 임포트 실패! (LANGCHAIN_AVAILABLE_LLM = False)")
    print(f"🚨 오류 메시지: {e}")
    print("=============================================\n")

from src.llm.prompts.prompt_loader import PromptLoader

# from openai import OpenAI
from src.config import get_config
from src.utils.logging_config import get_logger

class LLMProcessor:
    """
    LLMProcessor 클래스는 대규모 언어 모델(LLM)을 활용하여 사용자 입력과 검색된 데이터를 기반으로
    응답을 생성하는 역할을 합니다. LangChain 라이브러리를 사용하여 프롬프트 템플릿과 LLM 호출을 처리합니다.

    주요 기능:
    - 설정(config) 기반으로 모델 및 온도(temperature) 초기화
    - 검색된 청크 데이터를 컨텍스트로 변환
    - LangChain을 통해 LLM 응답 생성
    """

    def __init__(    
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        config=None,
        api_key: Optional[str] = None,
        prompt_file: Optional[str] = None
    ):
        """
        LLMProcessor 초기화

        Args:
            model: LLM 모델명
            temperature: 생성 온도
            config: 설정 객체
        """
         
        # Config & Logger
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        self.model_name = model or self.config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else self.config.OPENAI_TEMPERATURE

        # 특정 모델의 temperature 제한 회피
        if self.model_name == 'gpt-5-mini' and self.temperature == 0.0:
            self.logger.warning("gpt-5-mini는 temperature=0.0을 지원하지 않습니다. 1.0으로 강제 변경합니다.")
            self.temperature = 1.0

        # API 키 입력 처리
        if api_key is None or not api_key.strip():
            api_key_input = getpass("OpenAI API Key를 입력하세요: ").strip()
            api_key = api_key_input if api_key_input else None

        if not api_key:
            raise ValueError("OpenAI API Key가 제공되지 않았습니다.")

        # 환경 변수에 등록 (LangChain에서 자동 사용)
        os.environ["OPENAI_API_KEY"] = api_key

        # LangChain 초기화
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
            self.logger.info(f"LLMProcessor 초기화 완료 (model={self.model_name}, temperature={self.temperature})")
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")
            raise ImportError("LangChain이 설치되지 않았습니다.")
        
        # 프롬프트 로더 초기화
        self.prompts = PromptLoader()
        print(f"LLMProcessor 초기화 완료 (model={self.model_name})")

        # 1. 한국어 요약을 강제하는 PromptTemplate 정의
        KOREAN_SUMMARY_PROMPT = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="""기존의 대화 요약 정보는 다음과 같습니다:
            {summary}

            최근의 새로운 대화 내용은 다음과 같습니다:
            {new_lines}

            새로운 대화 내용을 바탕으로 요약 정보를 **반드시 한국어**로 업데이트 해주세요.
            최종 요약 정보:"""
        )

        # 2. 대화 메모리 초기화 (요약 기반)
        self.memory = ConversationSummaryMemory(
            llm=self.llm, 
            max_token_limit=1500,
            prompt=KOREAN_SUMMARY_PROMPT  # 정의한 한국어 프롬프트 적용
        )
        self.chat_history = InMemoryChatMessageHistory()
        self.logger.info("ConversationSummaryMemory (한국어 프롬프트 적용) 초기화 완료")
        # -------------------------------------------------------------

        # ConversationChain 생성 (Memory + LLM)
        base_chain = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
        self.conversation_chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def generate_response(self, query: str, retrieved_chunks: Dict[str, Any]) -> str:
        """
        검색된 청크를 컨텍스트로 활용하여 LLM 응답 생성

        Args:
            query: 사용자 질문
            retrieved_chunks: search() 또는 search_page() 결과 (Dict[str, Any] 또는 List[Dict[str, Any]])
        
        Returns:
            str: LLM 응답
        """
        self.logger.info(f"LLM 응답 생성 시작: query='{query[:50]}...'")

        # max_chunks는 일반적으로 config에서 가져오거나 파라미터로 받아야 하지만, 
        # 현재 코드에서는 정의되어 있지 않으므로 임시로 5를 사용하거나, 
        # generate_response의 정의에서 max_chunks 파라미터를 제거합니다.
        # 현재 호출 코드를 보니 max_chunks가 전달되지 않고 있으므로,
        # self._build_context_from_... 메서드에서 None을 허용하도록 처리합니다.
        max_chunks = 5  # 기본값 설정

        # ⭐️⭐️⭐️ 컨텍스트 구성 로직 수정 ⭐️⭐️⭐️
        
        # 1. search_page() 결과인지 확인: 'pages' 키가 있고 딕셔너리 형태일 때
        if isinstance(retrieved_chunks, dict) and 'pages' in retrieved_chunks:
            # retrieved_chunks['pages']를 사용하여 _build_context_from_pages 호출
            pages = retrieved_chunks.get('pages', [])
            context = self._build_context_from_pages(pages, max_chunks)
        
        # 2. search() 결과인지 확인: 청크 리스트 형태일 때
        elif isinstance(retrieved_chunks, list):
            # retrieved_chunks 자체를 사용하여 _build_context_from_chunks 호출
            context = self._build_context_from_chunks(retrieved_chunks, max_chunks)
        
        else:
            self.logger.warning("검색 결과가 유효한 딕셔너리(pages 포함) 또는 리스트 형태가 아닙니다.")
            context = self.config.NO_CONTEXT_MESSAGE
        
        # ⭐️⭐️⭐️ 컨텍스트 구성 로직 수정 끝 ⭐️⭐️⭐️


        if not context or context == self.config.NO_CONTEXT_MESSAGE:
            self.logger.warning("검색 결과 없음")
            context = self.config.NO_CONTEXT_MESSAGE
        
        self.logger.debug(f"context size = {len(context)}")

        # YAML 기반 프롬프트 템플릿 로드 (기존 로직 유지)
        prompt_text = None

        # 1) PromptLoader.templates 딕셔너리 우선 검사 (기존 로직 유지)
        if hasattr(self.prompts, "templates") and isinstance(self.prompts.templates, dict):
            for k in ("rag_prompt_template", "template", "default"):
                v = self.prompts.templates.get(k)
                if v:
                    prompt_text = v
                    break
            if not prompt_text:
                first = next(iter(self.prompts.templates.values()), None)
                if isinstance(first, dict):
                    prompt_text = first.get("rag_prompt_template") or first.get("template") or first.get("default")
                elif isinstance(first, str):
                    prompt_text = first

        # 2) load_template 메서드가 있으면 파일명으로 로드 시도 (기존 로직 유지)
        if not prompt_text and hasattr(self.prompts, "load_template"):
            try:
                tpl = self.prompts.load_template("prompt_temp_v1")
                if isinstance(tpl, dict):
                    prompt_text = tpl.get("rag_prompt_template") or tpl.get("template") or tpl.get("default")
            except Exception:
                pass

        # 3) 설정 파일(fallback) (기존 로직 유지)
        prompt_text = prompt_text or getattr(self.config, "RAG_PROMPT_TEMPLATE", None)

        if not prompt_text:
            raise KeyError("'rag_prompt_template' 또는 사용 가능한 프롬프트 텍스트를 찾을 수 없습니다.")

        # Note: ChatPromptTemplate.from_template을 사용하면 LangChain Chain을 통하지 않고 
        # 바로 LLM을 호출할 때 메모리 이력이 반영되지 않을 수 있습니다. 
        # 여기서는 ConversationChain을 사용하므로 그대로 둡니다.
        
        # ConversationChain을 통해 호출 (memory 반영)
        full_input = f"Context:\n{context}\n\nQuery:\n{query}"
        try:
            response = self.conversation_chain.invoke(
                {"input": full_input},
                config={"configurable": {"session_id": "default"}}
            )
            self.logger.info(f"LLM 응답 생성 완료")
            # LangChain RunnableWithMessageHistory의 결과는 일반적으로 딕셔너리 형태를 반환합니다.
            return response["response"] if isinstance(response, dict) and "response" in response else str(response)
        except Exception as e:
            self.logger.error(f"응답 생성 중 오류 발생: {e}")
            return f"응답 생성 중 오류 발생: {e}"
        
    def _build_context_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None
    ) -> str:
        """
        search() 결과에서 컨텍스트 생성
        
        Args:
            chunks: 청크 리스트
            max_chunks: 최대 포함 개수
        """
        if max_chunks:
            chunks = chunks[:max_chunks]
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            file_name = chunk.get('file_name', 'unknown')
            chunk_text = chunk.get('text', '')
            
            context_parts.append(
                self.config.CONTEXT_FORMAT.format(
                    index=i,
                    file_name=file_name,
                    chunk_text=chunk_text
                )
            )
        
        self.logger.debug(f"컨텍스트 구성 (search): {len(chunks)}개 청크")
        return "\n\n".join(context_parts)


    def _build_context_from_pages(
        self,
        pages: List[Dict[str, Any]],
        max_chunks: Optional[int] = None
    ) -> str:
        """
        search_page() 결과에서 컨텍스트 생성
        
        Args:
            pages: 페이지 리스트
            max_chunks: 최대 포함 개수
        """
        if max_chunks:
            pages = pages[:max_chunks]
        
        context_parts = []
        for i, page in enumerate(pages, 1):
            file_name = page.get('file_name', 'unknown')
            page_number = page.get('page_number', 0)
            page_text = page.get('text', '')
            score = page.get('score', 0.0)
            
            context_parts.append(
                f"[문서 {i}] {file_name} (페이지 {page_number}, 유사도: {score:.4f})\n{page_text}"
            )
        
        self.logger.debug(f"컨텍스트 구성 (search_page): {len(pages)}개 페이지")
        return "\n\n".join(context_parts)
        
    def get_memory_summary(self) -> str:
        """현재까지의 대화 요약 내용 반환"""
        try:
            vars = self.memory.load_memory_variables({})
            history = vars.get("history", "")
            if not history.strip():
                return "(아직 대화 기록이 없습니다.)"
            return history
        except Exception as e:
            self.logger.error(f"메모리 요약 조회 중 오류 발생: {e}")
            return f"메모리 조회 실패: {e}"