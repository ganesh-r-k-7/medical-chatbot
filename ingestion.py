import os
import hashlib
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.23.0.1:11434")


def get_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file to detect duplicates."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_pdf_documents(file_paths: List[str]) -> List[Document]:
    """Load PDF files and preserve page metadata."""
    documents = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["file_path"] = file_path
                doc.metadata["file_hash"] = get_file_hash(file_path)
            documents.extend(docs)
            print(f"Loaded {len(docs)} pages from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks while preserving metadata."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def get_vectorstore() -> Chroma:
    """Initialize or load persistent ChromaDB vector store."""
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def get_existing_hashes(vectorstore: Chroma) -> set:
    """Retrieve hashes of already-ingested documents."""
    try:
        results = vectorstore.get(include=["metadatas"])
        hashes = set()
        for metadata in results.get("metadatas", []):
            if metadata and "file_hash" in metadata:
                hashes.add(metadata["file_hash"])
        return hashes
    except Exception:
        return set()


def ingest_documents(file_paths: List[str], vectorstore: Optional[Chroma] = None) -> int:
    """Ingest new PDF documents into the vector store, skipping duplicates."""
    if vectorstore is None:
        vectorstore = get_vectorstore()

    existing_hashes = get_existing_hashes(vectorstore)
    print(f"Found {len(existing_hashes)} existing document(s) in vector store")

    new_files = []
    for fp in file_paths:
        if not os.path.exists(fp):
            print(f"Skipping missing file: {fp}")
            continue
        fhash = get_file_hash(fp)
        if fhash in existing_hashes:
            print(f"Skipping duplicate: {os.path.basename(fp)}")
        else:
            new_files.append(fp)

    if not new_files:
        print("No new documents to ingest.")
        return 0

    documents = load_pdf_documents(new_files)
    if not documents:
        print("No valid documents loaded.")
        return 0

    chunks = chunk_documents(documents)
    if not chunks:
        print("No chunks generated.")
        return 0

    try:
        vectorstore.add_documents(chunks)
        print(f"Successfully ingested {len(chunks)} chunks from {len(new_files)} document(s)")
        return len(chunks)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise


def clear_vectorstore(vectorstore: Optional[Chroma] = None) -> None:
    """Clear all documents from the vector store."""
    if vectorstore is None:
        vectorstore = get_vectorstore()
    try:
        vectorstore.delete_collection()
        print("Vector store cleared.")
    except Exception as e:
        print(f"Error clearing vector store: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <pdf_file1> [pdf_file2 ...]")
        sys.exit(1)
    ingest_documents(sys.argv[1:])