import os
from typing import List, Dict, Any, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv

from prompts import SYSTEM_PROMPT, RAG_PROMPT, format_docs

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.23.0.1:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical_docs")
TOP_K = int(os.getenv("TOP_K", "4"))


class RAGPipeline:
    def __init__(self, vectorstore: Optional[Chroma] = None):
        self.vectorstore = vectorstore or self._get_vectorstore()
        self.llm = self._get_llm()
        self.chain = self._build_chain()

    def _get_vectorstore(self) -> Chroma:
        try:
            embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL,
            )
            # Test embeddings connection
            embeddings.embed_query("test")
            return Chroma(
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {OLLAMA_BASE_URL} or embedding model '{EMBEDDING_MODEL}' not found. "
                f"Ensure Ollama is running and the model is pulled: ollama pull {EMBEDDING_MODEL}. "
                f"Original error: {e}"
            ) from e

    def _get_llm(self) -> ChatOllama:
        try:
            llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0,
                base_url=OLLAMA_BASE_URL,
            )
            # Test connection by checking if model is available
            llm.invoke("test")
            return llm
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {OLLAMA_BASE_URL} or model '{OLLAMA_MODEL}' not found. "
                f"Ensure Ollama is running and the model is pulled: ollama pull {OLLAMA_MODEL}. "
                f"Original error: {e}"
            ) from e

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )

        def retrieve_and_format(inputs: Dict[str, Any]) -> Dict[str, Any]:
            question = inputs["question"]
            docs = retriever.invoke(question)
            return {
                "context": format_docs(docs),
                "question": question,
                "source_documents": docs,
            }

        chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["question"])),
                source_documents=lambda x: retriever.invoke(x["question"]),
            )
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        return chain

    def query(self, question: str) -> Dict[str, Any]:
        """Process a question and return answer with sources."""
        try:
            docs = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K},
            ).invoke(question)

            if not docs:
                return {
                    "answer": "I don't have enough information in the knowledge base to answer this question.",
                    "sources": [],
                    "has_context": False,
                }

            answer = self.chain.invoke({"question": question})

            sources = []
            for doc in docs:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                })

            return {
                "answer": answer,
                "sources": sources,
                "has_context": True,
            }

        except Exception as e:
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "has_context": False,
                "error": str(e),
            }

    def check_knowledge_base(self) -> Dict[str, Any]:
        """Check if the knowledge base has any documents."""
        try:
            results = self.vectorstore.get(include=["metadatas"])
            count = len(results.get("ids", []))
            return {
                "document_count": count,
                "has_documents": count > 0,
            }
        except Exception as e:
            return {
                "document_count": 0,
                "has_documents": False,
                "error": str(e),
            }


def create_rag_pipeline() -> RAGPipeline:
    """Factory function to create a RAG pipeline instance."""
    return RAGPipeline()


if __name__ == "__main__":
    pipeline = create_rag_pipeline()
    kb = pipeline.check_knowledge_base()
    print(f"Knowledge base: {kb}")

    if kb["has_documents"]:
        test_query = "What is hypertension?"
        result = pipeline.query(test_query)
        print(f"\nQ: {test_query}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")