import sys
from pathlib import Path

# 프로젝트 루트 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
import importlib
import argparse
import logging
import src.utils.helper_c0z0c_dev as helper
from src.utils.helper_utils import *

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Colab 환경 체크
IS_COLAB = 'google.colab' in sys.modules

# OpenAI API 키 설정
openai_api_key = None
if IS_COLAB:
    from google.colab import userdata
    openai_api_key = userdata.get('OPENAI_API_KEY')
else:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    openai_api_key = openai_api_key.strip()
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    raise ValueError("OpenAI API 키 필요")

# 설정 로드
CONFIG_PATH = str(project_root / "configs" / "config.json")

from src import config
importlib.reload(config)
from src.config import get_config, Config

config = get_config(CONFIG_PATH)
config.DOCUMENTS_DB_PATH = str(project_root / "data" / "documents.db")
config.EMBEDDINGS_DB_PATH = str(project_root / "data" / "embeddings.db")
config.CHAT_HISTORY_DB_PATH = str(project_root / "data" / "chat_history.db")
config.VECTORSTORE_PATH = str(project_root / "data" / "vectorstore")
config.CONFIG_PATH = CONFIG_PATH


def process_query(query: str) -> str:
    """질문을 처리하여 LLM 응답 생성
    
    Args:
        query: 사용자 질문
        
    Returns:
        LLM 응답 텍스트
    """
    from src.llm import retrieval
    importlib.reload(retrieval)
    from src.llm.retrieval import Retrieval
    
    from src.llm import llm_processor
    importlib.reload(llm_processor)
    from src.llm.llm_processor import LLMProcessor
    
    logger.info(f"질문 처리 시작: {query}")
    
    retrieval_instance = Retrieval(config=config)
    llm_processor = LLMProcessor(config=config)

    result = retrieval_instance.search_page(query, sort_by='page')
    res = llm_processor.generate_response(query, retrieved_chunks=result)
    
    logger.info("질문 처리 완료")

    print("\n" + "-"*80)
    print_dic_tree(res.choices)
    print("-"*80 + "\n")
    
    return res.choices[0].message.content


def main() -> None:
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='질문을 받아 LLM 응답 생성')
    parser.add_argument('query', type=str, nargs='+', help='처리할 질문')
    
    args = parser.parse_args()
    query = ' '.join(args.query)
    
    response = process_query(query)
    print("\n" + "="*80)
    print("응답:")
    print("="*80)
    print(response)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
    
    # example
    # python scripts\pipeline_llm_question.py "중이온 가속기 극저온시스템 스팩은?"