"""Data pipeline for processing HR documents and updating Pinecone vector database"""

import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Union

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hr-policy")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    return _embeddings


def _get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is not set")
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists() -> None:
    """Create Pinecone index if it does not already exist"""
    pc = _get_pinecone_client()
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"Created Pinecone index '{PINECONE_INDEX_NAME}'")
    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'")


def clean_text(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\w)(LLP)', r'\1 LLP', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_by_policy(documents: List[Document]) -> List[str]:
    policies = []
    current_text = ""

    for doc in documents:
        text = doc.page_content
        if "POLICY" in text.upper():
            if current_text:
                policies.append(current_text)
            current_text = text
        else:
            current_text += " " + text

    if current_text:
        policies.append(current_text)

    return policies


def categorize_policy(text: str) -> str:
    text_upper = text.upper()
    if "ATTENDANCE" in text_upper:
        return "attendance"
    elif "LEAVE" in text_upper:
        return "leave"
    elif "SAFETY" in text_upper or "HSE" in text_upper:
        return "safety"
    elif "EQUAL" in text_upper:
        return "equal_opportunity"
    else:
        return "general"


def load_and_process_documents(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    policies = split_by_policy(documents)

    structured_docs = []
    for policy in policies:
        category = categorize_policy(policy)
        structured_docs.append({"text": policy, "category": category})

    return structured_docs


def chunk_documents(structured_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)

    final_chunks = []
    for doc in structured_docs:
        for chunk in splitter.split_text(doc["text"]):
            final_chunks.append({"text": chunk, "category": doc["category"]})

    return final_chunks


def upsert_to_pinecone(chunks: List[Dict[str, str]], source_file: str = "") -> PineconeVectorStore:
    """Embed chunks and upsert them into the Pinecone index"""
    lc_docs = [
        Document(
            page_content=chunk["text"],
            metadata={"category": chunk["category"], "source": source_file},
        )
        for chunk in chunks
    ]

    ids = [str(uuid.uuid4()) for _ in lc_docs]

    vector_store = PineconeVectorStore.from_documents(
        documents=lc_docs,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX_NAME,
        ids=ids,
    )

    print(f"Upserted {len(lc_docs)} vectors into Pinecone index '{PINECONE_INDEX_NAME}'")
    return vector_store


def load_vector_store() -> PineconeVectorStore:
    """Return a PineconeVectorStore connected to the existing index"""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
    )


def process_uploaded_file(file_path: Union[str, Path]) -> None:
    """Main pipeline: process uploaded file and upsert into Pinecone"""
    try:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        print(f"Processing file: {file_path}")

        ensure_index_exists()

        print("Loading and processing documents...")
        structured_docs = load_and_process_documents(file_path)
        print(f"Extracted {len(structured_docs)} policies")

        print("Chunking documents...")
        chunks = chunk_documents(structured_docs)
        print(f"Created {len(chunks)} chunks")

        print("Creating embeddings and upserting to Pinecone...")
        upsert_to_pinecone(chunks, source_file=file_path.name)

        print(f"Pipeline completed successfully for {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_uploaded_file(sys.argv[1])
    else:
        print("Usage: python data_pipeline.py <file_path>")
