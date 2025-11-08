# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LLMProcessor:
    \"\"\"LLM 응답 생성 클래스\"\"\"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model_name = model
        self.temperature = temperature

        if LANGCHAIN_AVAILABLE:
            self.llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            print("LangChain이 설치되지 않았습니다.")

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        api_key: Optional[str] = None
    ) -> str:
        \"\"\"
        검색된 청크를 바탕으로 LLM 응답 생성

        Args:
            query: 사용자 질문
            retrieved_chunks: 검색된 청크 리스트
            api_key: OpenAI API 키

        Returns:
            LLM 응답
        \"\"\"
        if not LANGCHAIN_AVAILABLE:
            return "LangChain이 설치되지 않았습니다."

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 컨텍스트 구성
        if not retrieved_chunks:
            context = "관련 문서를 찾을 수 없습니다."
        else:
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                file_name = chunk.get('file_name', 'unknown')
                chunk_text = chunk.get('chunk_text', '')
                context_parts.append(f"[문서 {i}: {file_name}]\n{chunk_text}")
            context = "\n\n".join(context_parts)

        # 프롬프트 템플릿
        template = \"\"\"다음 문서를 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변:\"\"\"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"context": context, "query": query})
            return response.content
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"
