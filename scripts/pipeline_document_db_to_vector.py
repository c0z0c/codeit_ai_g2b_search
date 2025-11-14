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
from src.utils.logging_config import (
    setup_logger,
    get_logger,
    reset_logger,
    reset_all_loggers,
    set_level,
    get_log_file_path,
    ShortLevelFormatter
)
# 로깅 설정

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

reset_all_loggers()
logger = setup_logger(config=config)

from src.processors import embedding_processor
importlib.reload(embedding_processor)
from src.processors.embedding_processor import EmbeddingProcessor

if __name__ == "__main__":
    proc_emb = EmbeddingProcessor(config=config)
    proc_emb.sync_with_docs_db(config.OPENAI_API_KEY)
    proc_emb.vector_manager.summary()

    # python scripts\pipeline_document_db_to_vector.py
