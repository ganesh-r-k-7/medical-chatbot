# Medical Chatbot

A RAG-powered medical chatbot built with Python, LangChain, Ollama, ChromaDB, and Streamlit.

## Features

- 📄 Upload medical PDF documents
- 🔍 Semantic search with ChromaDB vector storage
- 💬 Chat interface with source citations (filename + page numbers)
- ⚠️ Medical safety disclaimers and emergency guidance
- 🔒 Persistent knowledge base across restarts
- 🚫 Duplicate document detection
- 🛡️ Graceful error handling for missing context and API failures

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF Upload │────▶│ Ingestion   │────▶│  ChromaDB   │
│  (Streamlit)│     │ (PyPDF +    │     │  (Vector    │
└─────────────┘     │  Chunking)  │     │   Store)    │
                    └─────────────┘     └──────┬──────┘
                                               │
                     ┌─────────────┐     ┌──────▼──────┐
                     │   Streamlit │◀────│  RAG Chain  │
                     │   Frontend  │     │ (Retrieval  │
                     └─────────────┘     │  + LLM)     │
                           ▲             └─────────────┘
                           │
                     ┌─────┴─────┐
                     │  User     │
                     │  Query    │
                     └───────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and Configure Ollama

**On Windows (where Ollama runs):**
```bash
# Install Ollama from https://ollama.com/download
# Pull required models:
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Ensure Ollama listens on all interfaces (for WSL access):
# Set OLLAMA_HOST=0.0.0.0:11434 in Windows environment variables
# Or run: $env:OLLAMA_HOST="0.0.0.0:11434"; ollama serve
```

**On Ubuntu (WSL):**
```bash
# Configure environment to point to Windows Ollama
cp .env.example .env
# Edit .env if needed (default OLLAMA_BASE_URL=http://172.23.0.1:11434 works for WSL2)
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Ingest Documents (CLI Alternative)

```bash
python ingestion.py path/to/document1.pdf path/to/document2.pdf
```

## Project Structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit frontend |
| `ingestion.py` | PDF loading, chunking, embedding, storage |
| `rag_pipeline.py` | Retrieval + generation chain |
| `prompts.py` | System prompts and templates |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `.gitignore` | Git ignore rules |

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://172.23.0.1:11434` | Ollama server URL (Windows host from WSL2) |
| `OLLAMA_MODEL` | `llama3.2:3b` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store directory |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `4` | Retrieved chunks per query |

## Usage

1. **Upload Documents**: Use the sidebar to upload medical PDFs
2. **Ask Questions**: Type medical questions in the chat
3. **View Sources**: Expand "Sources" to see citations with page numbers
4. **Safety**: Emergency guidance appears for urgent symptoms

## Safety Notes

- This tool provides **information only**, not medical advice
- Always consult healthcare professionals for personal concerns
- Emergency symptoms → call 911 (or local emergency number) immediately

## Testing

```bash
# Test ingestion
python ingestion.py sample.pdf

# Test RAG pipeline
python rag_pipeline.py
```

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running on Windows: `ollama serve`
- Verify WSL can reach Windows host: `curl http://172.23.0.1:11434/api/tags`
- Check Windows firewall allows port 11434
- Set `OLLAMA_HOST=0.0.0.0:11434` on Windows before starting Ollama

### Model Not Found
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## License

MIT License - see LICENSE file for details.