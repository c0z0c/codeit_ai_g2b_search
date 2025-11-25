# -*- coding: utf-8 -*-

import json
from typing import Dict, Any, Optional
from openai import OpenAI


class RAGEvaluator:
    """
    RAG 시스템(Large Language Model 기반 검색-증강 생성)의 출력 품질을 평가하는 LLM 평가기.
    기존 함수 evaluate_rag_performance()를 클래스 내부 메서드로 통합한 버전.
    """

    def __init__(self, api_key: str, model_id: str = "gpt-5"):
        """
        OpenAI 클라이언트 초기화
        """
        self.client = OpenAI(api_key=api_key)
        self.model_id = model_id

    def evaluate(
        self,
        query: str,
        doc_text: str,
        query_result: str
    ) -> Dict[str, Any]:
        """
        RAG 시스템의 응답 품질을 GPT 모델을 통해 평가하고 JSON 데이터를 반환합니다.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 RAG(Retrieval-Augmented Generation) 시스템의 품질을 평가하는 "
                    "최고 전문가 'LLM Judge'입니다. "
                    "당신은 제공된 '문서 전체', '질의', '질의 답변'을 철저히 분석하여 "
                    "4가지 핵심 평가 지표에 대해 1점(매우 나쁨)부터 5점(매우 우수)까지 "
                    "점수를 매기고, 그 점수를 뒷받침하는 상세한 이유를 제공해야 합니다. "
                    "출력은 반드시 아래의 JSON 형식만을 따라야 합니다."
                )
            },
            {
                "role": "user",
                "content": f"""
## 평가 데이터
1. **질의 (Query):**
{query}

2. **문서 전체 (Doc Text / Retrieved Context):**
{doc_text}

3. **질의 답변 (Query Result / Generated Answer):**
{query_result}

---
## 평가 기준
[이전 제공한 평가 기준 전체를 사용한다고 가정]

---
## 출력 형식(JSON)
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
  "Overall_Assessment": "..."
}}
            """
        }
        ]

        try:
            # 1. JSON 형식 강제 추가 (API 호출 수정)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                response_format={"type": "json_object"}, # <--- JSON 출력 강제
            )

            content = response.choices[0].message.content

            if content:
                # 2. Markdown 코드 블록 제거 로직 추가 (JSON 추출)
                # content가 '```json\n{...}\n```' 형태일 경우 내부 JSON만 추출
                if content.startswith("```json"):
                    content = content.replace("```json\n", "").replace("\n```", "").strip()

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    return {
                        "error": "JSON Decode Error (추출 후 재시도 실패)",
                        "raw_content": content,
                        "exception": str(e)
                    }

            else:
                return {"error": "빈 응답을 수신했습니다."}

        except Exception as e:
            return {"error": str(e)}