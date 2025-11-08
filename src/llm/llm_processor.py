# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

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
        LLMProcessor 초기화 메서드.

        Args:
            model: 사용할 LLM 모델 이름 (예: "gpt-4")
            temperature: 생성 온도 (값이 높을수록 출력이 다양해짐)
            config: 사용자 정의 설정 객체 (없을 경우 기본 설정 사용)
        """
        # Config 로드
        self.config = config or get_config()

        # 로거 초기화
        self.logger = get_logger(__name__)

        # 모델 및 온도 설정 (파라미터 우선, 없으면 Config 사용)
        self.model_name = model or self.config.OPENAI_MODEL
        self.temperature = temperature if temperature is not None else self.config.OPENAI_TEMPERATURE

        # LangChain 초기화
        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
            self.logger.info(
                f"LLMProcessor 초기화 완료 (model={self.model_name}, temperature={self.temperature})"
            )
        else:
            self.logger.error("LangChain이 설치되지 않았습니다.")

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        api_key: Optional[str] = None
    ) -> str:
        """
        검색된 청크를 바탕으로 LLM 응답 생성.

        Args:
            query: 사용자 질문
            retrieved_chunks: 검색된 청크 리스트 (각 청크는 파일명 및 텍스트 포함)
            api_key: OpenAI API 키 (필요 시 설정)

        Returns:
            str: LLM 응답 텍스트
        """
        if not LANGCHAIN_AVAILABLE:
            self.logger.error("LangChain이 설치되지 않았습니다.")
            return "LangChain이 설치되지 않았습니다."

        self.logger.info(f"LLM 응답 생성 시작: query='{query[:50]}...'")

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 컨텍스트 구성
        if not retrieved_chunks:
            # 검색된 청크가 없을 경우 기본 메시지 사용
            context = self.config.NO_CONTEXT_MESSAGE
            self.logger.warning("검색된 청크가 없습니다.")
        else:
            # 검색된 청크를 컨텍스트로 변환
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                file_name = chunk.get('file_name', 'unknown')
                chunk_text = chunk.get('chunk_text', '')
                context_parts.append(
                    self.config.CONTEXT_FORMAT.format(
                        index=i,
                        file_name=file_name,
                        chunk_text=chunk_text
                    )
                )
            context = "\n\n".join(context_parts)
            self.logger.debug(f"컨텍스트 구성 완료: {len(retrieved_chunks)}개 청크")

        # 프롬프트 템플릿 로드
        template = self.config.RAG_PROMPT_TEMPLATE

        # LangChain 프롬프트 생성
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            # LLM 호출 및 응답 생성
            self.logger.debug(f"LLM API 호출 중... (모델: {self.model_name})")
            response = chain.invoke({"context": context, "query": query})
            self.logger.info(f"LLM 응답 생성 완료: {len(response.content)} 문자")
            return response.content
        except Exception as e:
            # 오류 발생 시 메시지 반환
            self.logger.error(f"응답 생성 중 오류 발생: {e}")
            return f"응답 생성 중 오류 발생: {e}"