# -*- coding: utf-8 -*-
from pathlib import Path

project_root = Path(__file__).parent.parent

# document_processor.py 내용
document_processor_code = """# -*- coding: utf-8 -*-
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import tiktoken

try:
    import pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from src.db import DocumentsDB

class DocumentProcessor:
    \\\"\\\"\\\"문서 처리 클래스 - PDF를 Markdown으로 변환하고 DB에 저장\\\"\\\"\\\"

    def __init__(self, db_path: str = 'data/documents.db'):
        self.docs_db = DocumentsDB(db_path)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def calculate_file_hash(self, file_path: Path) -> str:
        \\\"\\\"\\\"파일 SHA-256 해시 계산\\\"\\\"\\\"
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def count_tokens(self, text: str) -> int:
        \\\"\\\"\\\"텍스트 토큰 수 계산\\\"\\\"\\\"
        return len(self.tokenizer.encode(text))

    def process_pdf(self, pdf_path: str) -> Optional[str]:
        \\\"\\\"\\\"
        PDF 파일을 처리하여 Markdown으로 변환하고 DB에 저장

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            파일 해시값 또는 None
        \\\"\\\"\\\"
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF가 설치되지 않았습니다. pip install pymupdf")
            return None

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            print(f"파일을 찾을 수 없습니다: {pdf_path}")
            return None

        # 파일 해시 계산
        file_hash = self.calculate_file_hash(pdf_file)

        # PDF 열기 및 텍스트 추출
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        all_content = []

        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text("text")

            # 빈 페이지 처리
            is_empty = len(text.strip()) < 10
            if is_empty:
                page_content = "--- [빈페이지] ---"
            else:
                page_content = f"--- 페이지 {page_num + 1} ---\\n\\n{text}"

            all_content.append(page_content)

            # 페이지 데이터 저장
            token_count = self.count_tokens(page_content)
            self.docs_db.insert_page_data(
                file_hash=file_hash,
                page_number=page_num + 1,
                markdown_content=page_content,
                token_count=token_count,
                is_empty=is_empty
            )

        # 전체 콘텐츠 결합
        full_content = "\\n\\n".join(all_content)
        total_tokens = self.count_tokens(full_content)

        # 파일 정보 저장
        self.docs_db.insert_file_info(
            file_hash=file_hash,
            file_name=pdf_file.name,
            total_pages=total_pages,
            file_size=pdf_file.stat().st_size,
            total_chars=len(full_content),
            total_tokens=total_tokens
        )

        doc.close()
        print(f"PDF 처리 완료: {pdf_file.name} ({total_pages} 페이지, {total_tokens} 토큰)")
        return file_hash
"""

# embedding_processor.py 내용
embedding_processor_code = """# -*- coding: utf-8 -*-
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db import DocumentsDB, EmbeddingsDB

class EmbeddingProcessor:
    \\\"\\\"\\\"임베딩 처리 클래스 - 문서를 청킹하고 벡터 임베딩 생성\\\"\\\"\\\"

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.docs_db = DocumentsDB()
        self.embeddings_db = EmbeddingsDB()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\\n\\n", "\\n", " ", ""]
            )
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            print("LangChain이 설치되지 않았습니다.")

    def calculate_embedding_hash(self, file_hash: str, config: Dict) -> str:
        \\\"\\\"\\\"임베딩 설정 해시 계산\\\"\\\"\\\"
        data = f"{file_hash}_{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()

    def process_document(self, file_hash: str, api_key: Optional[str] = None) -> Optional[str]:
        \\\"\\\"\\\"
        문서를 청킹하고 임베딩 생성

        Args:
            file_hash: 파일 해시값
            api_key: OpenAI API 키 (선택사항)

        Returns:
            임베딩 해시값 또는 None
        \\\"\\\"\\\"
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            print("필수 패키지가 설치되지 않았습니다.")
            return None

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 문서 내용 가져오기
        pages = self.docs_db.get_page_data(file_hash)
        if not pages:
            print(f"문서를 찾을 수 없습니다: {file_hash[:8]}...")
            return None

        # 모든 페이지 콘텐츠 결합
        full_text = "\\n\\n".join([p['markdown_content'] for p in pages if not p['is_empty']])

        # 청킹
        chunks = self.text_splitter.split_text(full_text)
        total_chunks = len(chunks)
        print(f"청킹 완료: {total_chunks}개 청크 생성")

        # 임베딩 생성
        try:
            embeddings = self.embeddings.embed_documents(chunks)
            embeddings_array = np.array(embeddings).astype('float32')
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return None

        # FAISS 인덱스 생성
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # 임베딩 해시 계산
        config = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'model': self.embedding_model
        }
        embedding_hash = self.calculate_embedding_hash(file_hash, config)

        # FAISS 인덱스 저장
        faiss_path = f"data/vectorstore/{embedding_hash}.faiss"
        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, faiss_path)

        # 임베딩 메타데이터 저장
        self.embeddings_db.insert_embedding_meta(
            embedding_hash=embedding_hash,
            file_hash=file_hash,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            preprocessing_option={},
            embedding_model=self.embedding_model,
            total_chunks=total_chunks,
            vector_path=faiss_path
        )

        # 청크 매핑 저장
        file_info = self.docs_db.get_file_info(file_hash)
        for idx, chunk_text in enumerate(chunks):
            self.embeddings_db.insert_chunk_mapping(
                embedding_hash=embedding_hash,
                file_hash=file_hash,
                file_name=file_info['file_name'] if file_info else 'unknown',
                chunk_text=chunk_text,
                vector_index=idx,
                estimated_tokens=len(chunk_text) // 4
            )

        print(f"임베딩 처리 완료: {embedding_hash[:8]}...")
        return embedding_hash
"""

# retrieval.py 내용
retrieval_code = """# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from src.db import EmbeddingsDB

class Retrieval:
    \\\"\\\"\\\"검색 클래스 - 쿼리에 대한 유사 청크 검색\\\"\\\"\\\"

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings_db = EmbeddingsDB()
        self.embedding_model = embedding_model

        if LANGCHAIN_AVAILABLE:
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            print("LangChain이 설치되지 않았습니다.")

    def search(
        self,
        query: str,
        embedding_hash: str,
        top_k: int = 5,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        \\\"\\\"\\\"
        쿼리에 대한 유사 청크 검색

        Args:
            query: 검색 쿼리
            embedding_hash: 임베딩 해시값
            top_k: 상위 k개 검색
            api_key: OpenAI API 키

        Returns:
            검색 결과 리스트
        \\\"\\\"\\\"
        if not LANGCHAIN_AVAILABLE or not FAISS_AVAILABLE:
            print("필수 패키지가 설치되지 않았습니다.")
            return []

        # API 키 설정
        if api_key:
            import os
            os.environ['OPENAI_API_KEY'] = api_key

        # 임베딩 메타데이터 가져오기
        meta = self.embeddings_db.get_embedding_meta(embedding_hash)
        if not meta or not meta.get('vector_path'):
            print(f"임베딩을 찾을 수 없습니다: {embedding_hash[:8]}...")
            return []

        # FAISS 인덱스 로드
        try:
            index = faiss.read_index(meta['vector_path'])
        except Exception as e:
            print(f"FAISS 인덱스 로드 실패: {e}")
            return []

        # 쿼리 임베딩 생성
        try:
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
        except Exception as e:
            print(f"쿼리 임베딩 생성 실패: {e}")
            return []

        # 유사도 검색
        distances, indices = index.search(query_vector, top_k)

        # 결과 조회
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.embeddings_db.get_chunk_by_vector_index(embedding_hash, int(idx))
            if chunk:
                chunk['similarity'] = 1 / (1 + float(dist))
                chunk['distance'] = float(dist)
                results.append(chunk)

        return results
"""

# llm_processor.py 내용
llm_processor_code = """# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

class LLMProcessor:
    \\\"\\\"\\\"LLM 응답 생성 클래스\\\"\\\"\\\"

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
        \\\"\\\"\\\"
        검색된 청크를 바탕으로 LLM 응답 생성

        Args:
            query: 사용자 질문
            retrieved_chunks: 검색된 청크 리스트
            api_key: OpenAI API 키

        Returns:
            LLM 응답
        \\\"\\\"\\\"
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
                context_parts.append(f"[문서 {i}: {file_name}]\\n{chunk_text}")
            context = "\\n\\n".join(context_parts)

        # 프롬프트 템플릿
        template = \\\"\\\"\\\"다음 문서를 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {query}

답변:\\\"\\\"\\\"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        try:
            response = chain.invoke({"context": context, "query": query})
            return response.content
        except Exception as e:
            return f"응답 생성 중 오류 발생: {e}"
"""

# 파일 생성
processors_dir = project_root / 'src' / 'processors'
llm_dir = project_root / 'src' / 'llm'

with open(processors_dir / 'document_processor.py', 'w', encoding='utf-8') as f:
    f.write(document_processor_code)
print('document_processor.py created')

with open(processors_dir / 'embedding_processor.py', 'w', encoding='utf-8') as f:
    f.write(embedding_processor_code)
print('embedding_processor.py created')

with open(llm_dir / 'retrieval.py', 'w', encoding='utf-8') as f:
    f.write(retrieval_code)
print('retrieval.py created')

with open(llm_dir / 'llm_processor.py', 'w', encoding='utf-8') as f:
    f.write(llm_processor_code)
print('llm_processor.py created')

print('All core modules created successfully!')
