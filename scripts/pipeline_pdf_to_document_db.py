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

# print_dic_tree(config.to_dict())

from src.processors.document_processor import DocumentProcessor

def process_pdf(file_path: Path) -> None:
    """PDF 파일을 처리하여 DB에 저장
    
    Args:
        file_path: PDF 파일 경로
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {file_path}")
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f"PDF 파일이 아닙니다: {file_path}")
    
    logger.info(f"PDF 처리 시작: {file_path}")
    processor = DocumentProcessor(config=config)
    processor.process_pdf(file_path)
    logger.info(f"PDF 처리 완료: {file_path}")
    processor.docs_db.summary()

def main() -> None:
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='PDF를 DB에 저장하는 파이프라인')
    parser.add_argument('pdf_path', type=str, nargs='*', help='처리할 PDF 파일 경로')
    
    args = parser.parse_args()
    
    if not args.pdf_path:
        parser.error("pdf_path 인자가 필요합니다")
    
    # 백슬래시 이스케이프 처리된 토큰 재결합
    pdf_path_str = ' '.join(args.pdf_path).strip('"')
    pdf_path = Path(pdf_path_str)
    
    process_pdf(pdf_path)

if __name__ == "__main__":
    main()
    
    # example
    # python scripts\pipeline_pdf_to_document_db.py "d:\temp\codeit_ai_g2b_search_data.zip.unzip\files\기초과학연구원_2025년도 중이온가속기용 극저온시스템 운전 용역.pdf"