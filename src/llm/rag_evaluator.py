# -*- coding: utf-8 -*-

from openai import OpenAI
import json
from typing import Dict, Any, Optional

# OpenAI 클라이언트 초기화 (API 키는 환경 변수에서 로드한다고 가정)
# 또는 함수 인수로 받아야 합니다.
def initialize_openai_client(api_key: str) -> OpenAI:
    """OpenAI 클라이언트를 초기화합니다."""
    return OpenAI(api_key=api_key)

def evaluate_rag_performance(
    client: OpenAI,
    query: str,
    doc_text: str,
    query_result: str,
    model_id: str = "gpt-4o-mini", # 기본 모델 설정
  ) -> Dict[str, Any]:
    """
    GPT 모델을 사용하여 RAG 시스템의 성능을 평가하고 결과를 JSON 형식으로 반환합니다.
    """

    messages = [
        {
            "role": "system",
            "content": "당신은 RAG(Retrieval-Augmented Generation) 시스템의 품질을 평가하는 최고 전문가 'LLM Judge'입니다. 당신은 제공된 '문서 전체', '질의', '질의 답변'을 철저히 분석하여 4가지 핵심 평가 지표에 대해 1점(매우 나쁨)부터 5점(매우 우수)까지 점수를 매기고, 그 점수를 뒷받침하는 상세한 이유를 제공해야 합니다. 출력은 반드시 아래의 JSON 형식만을 따라야 합니다."
        },
        {
            "role": "user",
            "content": """
      ## 평가 데이터
      1.  **질의 (Query):**\n{query}
      2.  **문서 전체 (Doc Text / Retrieved Context):**\n{doc_text}
      3.  **질의 답변 (Query Result / Generated Answer):**\n{query_result}

      ---
      ## 평가 지표 및 기준
      [... (이전에 제공하신 평가 지표 기준 내용 전체) ...]

      ---
      ## 출력 형식
      결과는 반드시 다음의 **JSON 형식**으로만 출력해야 합니다.
      ```json
      {{
        "Query": "{query}",
        "Generated_Answer": "{query_result}",
        "Evaluation_Metrics": {{
          "Faithfulness": {{
            "Rating": 0,
            "Reasoning": "..."
          }},
          "Context_Relevance": {{
            "Rating": 0,
            "Reasoning": "..."
          }},
          "Answer_Accuracy": {{
            "Rating": 0,
            "Reasoning": "..."
          }},
          "Answer_Relevance": {{
            "Rating": 0,
            "Reasoning": "..."
          }}
        }},
        "Overall_Assessment": "모든 평가 지표를 종합하여 RAG 시스템의 전반적인 성능을 1-2문장으로 요약합니다."
      }}
      ```
      """.format(query=query, doc_text=doc_text, query_result=query_result)
      }
    ]

    # -----------------------------------------------------
    # ⭐️ 실제 평가(API 호출) 및 결과 처리 로직 ⭐️
    # -----------------------------------------------------
    try:
        # 1. OpenAI API 호출
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            # max_completion_tokens는 모델이 자동으로 결정하도록 하거나,
            # JSON 응답을 위해 충분히 크게 설정해야 합니다. (여기서는 생략)
            # JSON 포맷팅을 강제하는 것이 안정적일 수 있습니다.
            # response_format={"type": "json_object"} 
        )
        content = response.choices[0].message.content
        
        # 2. JSON 문자열을 딕셔너리로 파싱
        if content:
            try:
                # LLM이 반환한 JSON 문자열을 파이썬 딕셔너리로 변환
                return json.loads(content)
            except json.JSONDecodeError:
                # JSON 형식이 잘못되었을 경우 에러 처리
                print(f"경고: 모델({model_id}) 응답이 유효한 JSON 형식이 아닙니다.")
                return {"error": "JSON Decode Error", "raw_content": content}
        else:
            return {"error": "모델로부터 응답 내용이 비어 있습니다."}

    except Exception as e:
        # API 호출 중 발생한 예외 처리 (예: API 키 오류, 네트워크 오류 등)
        print(f"평가 중 API 오류 발생: {e}")
        return {"error": str(e)}