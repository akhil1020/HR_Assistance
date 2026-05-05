"""Data pipeline for processing HR documents and updating vector database"""

import re
from pathlib import Path
from typing import List, Dict, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Configuration - Using pathlib for cross-platform path support
BASE_DIR = Path(__file__).parent.parent  # Project root
UPLOAD_DIR = BASE_DIR / "app" / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "app" / "data" / "hr_policy_vector_db"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Fix merged words like "workinghours" → "working hours"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix LLP / company names spacing
    text = re.sub(r'(\w)(LLP)', r'\1 LLP', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def split_by_policy(documents: List[Document]) -> List[str]:
    """Split documents into individual policies"""
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
    """Categorize policy based on content"""
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
    """Load PDF and process into structured documents"""
    # Convert to Path object if string
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    # Load PDF
    loader = PyPDFLoader(str(file_path))
    documents = loader.load()

    # Clean text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Split by policy
    policies = split_by_policy(documents)

    # Categorize and structure
    structured_docs = []
    for policy in policies:
        category = categorize_policy(policy)
        structured_docs.append({
            "text": policy,
            "category": category
        })

    return structured_docs


def chunk_documents(structured_docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Split documents into chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120
    )

    final_chunks = []

    for doc in structured_docs:
        chunks = splitter.split_text(doc["text"])

        for chunk in chunks:
            final_chunks.append({
                "text": chunk,
                "category": doc["category"]
            })

    return final_chunks


def create_vector_db(chunks: List[Dict[str, str]]) -> FAISS:
    """Create FAISS vector database from chunks"""
    # Convert to LangChain documents
    lc_docs = [
        Document(
            page_content=chunk["text"],
            metadata={"category": chunk["category"]}
        )
        for chunk in chunks
    ]

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL
    )

    # Create FAISS database
    db = FAISS.from_documents(lc_docs, embeddings)

    return db


def save_vector_db(db: FAISS, db_path: Union[str, Path] = None) -> None:
    """Save vector database to disk"""
    if db_path is None:
        db_path = VECTOR_DB_DIR
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    db_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(db_path))
    print(f"Vector database saved to {db_path}")


def load_vector_db(db_path: Union[str, Path] = None) -> FAISS:
    """Load existing vector database"""
    if db_path is None:
        db_path = VECTOR_DB_DIR
    db_path = Path(db_path) if isinstance(db_path, str) else db_path
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDINGS_MODEL
    )
    db = FAISS.load_local(str(db_path), embeddings, allow_dangerous_deserialization=True)
    return db


def process_uploaded_file(file_path: Union[str, Path]) -> None:
    """Main pipeline: process uploaded file and update vector database"""
    try:
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        print(f"Processing file: {file_path}")

        # Step 1: Load and process documents
        print("Loading and processing documents...")
        structured_docs = load_and_process_documents(file_path)
        print(f"Extracted {len(structured_docs)} policies")

        # Step 2: Chunk documents
        print("Chunking documents...")
        chunks = chunk_documents(structured_docs)
        print(f"Created {len(chunks)} chunks")

        # Step 3: Create vector database
        print("Creating embeddings and vector database...")
        db = create_vector_db(chunks)

        # Step 4: Save vector database
        print("Saving vector database...")
        save_vector_db(db)

        print(f"Pipeline completed successfully for {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_uploaded_file(file_path)
    else:
        print("Usage: python data_pipeline.py <file_path>")
