# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from openai import OpenAI
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

    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None, config=None):
        """
        LLMProcessor 초기화

        Args:
            model: LLM 모델명
            temperature: 생성 온도
            config: 설정 객체
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        self.model_name = model or self.config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else self.config.OPENAI_TEMPERATURE
        
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("LangChain 미설치")
            raise ImportError("LangChain 라이브러리 필요")
        
        self.logger.info(f"LLMProcessor 초기화 (model={self.model_name}, temperature={self.temperature})")

    def generate_response(
        self,
        question: str,
        retrieved_chunks: Any,
        api_key: Optional[str] = None,
        max_chunks: Optional[int] = None
    ) -> str:
        """
        검색된 청크를 컨텍스트로 활용하여 LLM 응답 생성 (OpenAI 직접 사용)

        Args:
            question: 사용자 질문
            retrieved_chunks: search() 또는 search_page() 결과
            api_key: OpenAI API 키 (옵션)
            max_chunks: 최대 청크/페이지 수

        Returns:
            str: LLM 응답
        """
        # API 키 설정
        effective_api_key = api_key or self.config.OPENAI_API_KEY
        if not effective_api_key:
            self.logger.error("OpenAI API 키 미설정")
            return "API 키가 설정되지 않았습니다."

        # 컨텍스트 구성
        context = self._build_context(retrieved_chunks, max_chunks)
        if not context:
            self.logger.warning("검색 결과 없음")
            context = self.config.NO_CONTEXT_MESSAGE
        
        self.logger.debug(f"context size = {len(context)}")

        # 프롬프트 템플릿 적용
        template = self.config.RAG_PROMPT_TEMPLATE
        user_message = template.format(context=context, question=question)

        # OpenAI 클라이언트 생성
        client = OpenAI(api_key=effective_api_key)

        # 모델별 파라미터 설정
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_message}]
        }

        if self.model_name.startswith(("gpt-5", "gpt-4.1", "o1")):
            params["max_completion_tokens"] = 50000
            self.logger.debug(f"모델 {self.model_name}: temperature 제외, max_completion_tokens 사용")
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = 50000
            self.logger.debug(f"모델 {self.model_name}: temperature={self.temperature} 사용")

        # API 호출
        try:
            response = client.chat.completions.create(**params)
            #answer = response.choices[0].message.content
            self.logger.debug(f"LLM 응답 생성 완료")
            return response
            #return answer
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 실패: {e}")
            raise


    def _build_context(
        self,
        retrieved_chunks: Any,
        max_chunks: Optional[int] = None
    ) -> str:
        """
        검색 결과 형태에 따라 컨텍스트를 자동 생성합니다.

        Args:
            retrieved_chunks: search() 또는 search_page() 결과
            max_chunks: 최대 청크/페이지 수

        Returns:
            str: 포맷된 컨텍스트 문자열
        """
        if isinstance(retrieved_chunks, dict) and 'pages' in retrieved_chunks:
            return self._build_context_from_pages(retrieved_chunks['pages'], max_chunks)
        
        if isinstance(retrieved_chunks, list) and retrieved_chunks:
            return self._build_context_from_chunks(retrieved_chunks, max_chunks)
        
        return ""


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