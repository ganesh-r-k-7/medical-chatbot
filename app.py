import streamlit as st
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import create_rag_pipeline
from ingestion import ingest_documents, get_vectorstore, get_file_hash

st.set_page_config(
    page_title="MediAssist - AI Medical Information Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ICONS (inline SVG)
# ============================================================
ICON_MEDICAL_CROSS = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 2v20M2 12h20"/>
</svg>
"""

ICON_DOCUMENT = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/>
    <line x1="16" y1="17" x2="8" y2="17"/>
    <polyline points="10 9 9 9 8 9"/>
</svg>
"""

ICON_UPLOAD = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="17 8 12 3 7 8"/>
    <line x1="12" y1="3" x2="12" y2="15"/>
</svg>
"""

ICON_CHECK = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="20 6 9 17 4 12"/>
</svg>
"""

ICON_CLOCK = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <polyline points="12 6 12 12 16 14"/>
</svg>
"""

ICON_ALERT = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
</svg>
"""

ICON_TRASH = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
</svg>
"""

ICON_PLUS = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <line x1="12" y1="5" x2="12" y2="19"/>
    <line x1="5" y1="12" x2="19" y2="12"/>
</svg>
"""

ICON_SEND = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
</svg>
"""

ICON_LOADER = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
    <path d="M12 2a10 10 0 0 1 10 10" stroke-opacity="1"/>
</svg>
"""

ICON_CHEVRON_DOWN = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="6 9 12 15 18 9"/>
</svg>
"""

ICON_SOURCES = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
</svg>
"""

ICON_SHIELD = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
</svg>
"""

ICON_INFO = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="16" x2="12" y2="12"/>
    <line x1="12" y1="8" x2="12.01" y2="8"/>
</svg>
"""

# ============================================================
# CSS LOADING
# ============================================================
def load_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = None
    if "documents_indexed" not in st.session_state:
        st.session_state.documents_indexed = set()
    if "documents_processing" not in st.session_state:
        st.session_state.documents_processing = set()
    if "documents_failed" not in st.session_state:
        st.session_state.documents_failed = {}
    if "sidebar_open" not in st.session_state:
        st.session_state.sidebar_open = True
    if "ollama_connected" not in st.session_state:
        st.session_state.ollama_connected = None
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True

# ============================================================
# PIPELINE MANAGEMENT
# ============================================================
@st.cache_resource
def get_cached_pipeline():
    return create_rag_pipeline()

def get_pipeline():
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = get_cached_pipeline()
            st.session_state.pipeline_error = None
            st.session_state.ollama_connected = True
        except Exception as e:
            st.session_state.pipeline_error = str(e)
            st.session_state.ollama_connected = False
            st.session_state.pipeline = None
    return st.session_state.pipeline

def reset_pipeline():
    st.session_state.pipeline = None
    st.session_state.pipeline_error = None
    st.session_state.ollama_connected = None
    get_cached_pipeline.clear()
    get_pipeline()

# ============================================================
# FILE HANDLING
# ============================================================
def get_document_status(file_hash: str) -> str:
    if file_hash in st.session_state.documents_failed:
        return "error"
    if file_hash in st.session_state.documents_processing:
        return "processing"
    if file_hash in st.session_state.documents_indexed:
        return "indexed"
    return "unknown"

def handle_file_upload(uploaded_files):
    if not uploaded_files:
        return
    
    vectorstore = get_vectorstore()
    existing_hashes = set()
    try:
        results = vectorstore.get(include=["metadatas"])
        for metadata in results.get("metadatas", []):
            if metadata and "file_hash" in metadata:
                existing_hashes.add(metadata["file_hash"])
    except Exception:
        pass
    
    temp_paths = []
    new_files = []
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getbuffer()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        if file_hash in existing_hashes or file_hash in st.session_state.documents_indexed:
            st.toast(f"Skipped duplicate: {uploaded_file.name}", icon="⚠️")
            continue
        
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        temp_paths.append((temp_path, uploaded_file.name, file_hash))
        new_files.append((uploaded_file.name, file_hash))
        st.session_state.documents_processing.add(file_hash)
    
    if not temp_paths:
        return
    
    try:
        file_paths_only = [tp[0] for tp in temp_paths]
        chunks_added = ingest_documents(file_paths_only, vectorstore)
        
        if chunks_added > 0:
            for name, fhash in new_files:
                st.session_state.documents_indexed.add(fhash)
                st.session_state.documents_processing.discard(fhash)
                st.session_state.documents_failed.pop(fhash, None)
            st.toast(f"Indexed {chunks_added} chunks from {len(new_files)} document(s)", icon="✅")
            st.session_state.pipeline = get_cached_pipeline()
        else:
            for _, fhash in new_files:
                st.session_state.documents_processing.discard(fhash)
            st.toast("No new documents were added (duplicates skipped).", icon="ℹ️")
            
    except Exception as e:
        for _, fhash in new_files:
            st.session_state.documents_processing.discard(fhash)
            st.session_state.documents_failed[fhash] = str(e)
        st.error(f"Error processing documents: {e}")
    finally:
        for tp, _, _ in temp_paths:
            if os.path.exists(tp):
                os.remove(tp)

def remove_document(file_hash: str, filename: str):
    st.session_state.documents_indexed.discard(file_hash)
    st.session_state.documents_processing.discard(file_hash)
    st.session_state.documents_failed.pop(file_hash, None)
    st.toast(f"Removed {filename} from session tracking", icon="🗑️")

# ============================================================
# CHAT HANDLING
# ============================================================
def process_user_message(prompt: str):
    pipeline = get_pipeline()
    if pipeline is None:
        st.error("Cannot process message: Pipeline not available. Check Ollama connection.")
        return
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = pipeline.query(prompt)
        
        answer = result["answer"]
        st.markdown(answer)
        
        sources = result.get("sources", [])
        if sources:
            render_sources(sources)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "has_context": result.get("has_context", False),
        })
    
    st.session_state.show_welcome = False

def render_sources(sources):
    if not sources:
        return
    
    st.html(f"""
    <div class="mediassist-sources">
        <div class="mediassist-sources__header" onclick="this.parentElement.classList.toggle('expanded'); this.querySelector('.mediassist-sources__icon').classList.toggle('mediassist-sources__icon--expanded');">
            {ICON_SOURCES}
            <span>Sources ({len(sources)})</span>
            <span class="mediassist-sources__icon">{ICON_CHEVRON_DOWN}</span>
        </div>
        <div class="mediassist-sources__list">
    """)
    
    for i, src in enumerate(sources, 1):
        page_info = f'<span class="mediassist-source__page">Page {src["page"]}</span>' if src.get("page") and src["page"] != "N/A" else ""
        preview = src.get("content_preview", "")
        st.html(f"""
        <div class="mediassist-source">
            <div class="mediassist-source__header">
                <span class="mediassist-source__filename">Source {i}: {src["source"]}</span>
                {page_info}
            </div>
            <p class="mediassist-source__preview">{preview}</p>
        </div>
        """)
    
    st.html("</div></div>")

# ============================================================
# UI COMPONENTS
# ============================================================
def render_header():
    ollama_status = st.session_state.ollama_connected
    status_class = ""
    status_text = "Checking..."
    if ollama_status is True:
        status_class = "mediassist-header__status-dot--connected"
        status_text = "Local AI · Ollama"
    elif ollama_status is False:
        status_class = "mediassist-header__status-dot--error"
        status_text = "Ollama unavailable"
    
    st.html(f"""
    <div class="mediassist-header">
        <div class="mediassist-header__icon">{ICON_MEDICAL_CROSS}</div>
        <div class="mediassist-header__content">
            <h1 class="mediassist-header__title">MediAssist</h1>
            <p class="mediassist-header__subtitle">AI-powered medical information assistant</p>
            <p class="mediassist-header__hint">Responses grounded in your uploaded medical references when available</p>
        </div>
        <div class="mediassist-header__status">
            <span class="mediassist-header__status-dot {status_class}"></span>
            <span>{status_text}</span>
        </div>
    </div>
    """)

def render_sidebar():
    pipeline = get_pipeline()
    
    st.html("""
    <div class="mediassist-sidebar__section">
        <h2 class="mediassist-sidebar__heading">Medical Knowledge</h2>
        <p class="mediassist-sidebar__description">Upload trusted medical references to ground your answers.</p>
    </div>
    """)
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload medical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload medical documents to expand the knowledge base (max 50MB each)",
        label_visibility="collapsed",
        key="file_uploader"
    )
    
    # Custom uploader display
    if uploaded_files:
        st.html(f"""
        <div class="mediassist-uploader mediassist-uploader--active">
            <div class="mediassist-uploader__icon">{ICON_DOCUMENT}</div>
            <p class="mediassist-uploader__text">{len(uploaded_files)} file(s) selected</p>
            <p class="mediassist-uploader__hint">Click "Process Documents" to index</p>
            <div class="mediassist-uploader__files">
        """)
        
        for f in uploaded_files:
            size_kb = len(f.getbuffer()) / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"
            st.html(f"""
            <div class="mediassist-uploader__file">
                <span class="mediassist-uploader__file-name">{f.name}</span>
                <span class="mediassist-uploader__file-size">{size_str}</span>
            </div>
            """)
        
        st.html("</div></div>")
    
    if st.button("Process Documents", disabled=not uploaded_files, use_container_width=True, key="process_docs"):
        handle_file_upload(uploaded_files)
        st.rerun()
    
    st.html("<hr style='margin: 1.5rem 0; border-color: var(--color-border);'>")
    
    # Document List
    kb = pipeline.check_knowledge_base() if pipeline else {"document_count": 0, "has_documents": False}
    
    if kb.get("has_documents"):
        st.html(f"""
        <div class="mediassist-kb-status mediassist-kb-status--has-docs">
            {ICON_CHECK}
            <span>Knowledge Base: {kb['document_count']} chunks indexed</span>
        </div>
        """)
    else:
        st.html(f"""
        <div class="mediassist-kb-status mediassist-kb-status--empty">
            {ICON_INFO}
            <span>Knowledge base is empty. Upload PDFs to get started.</span>
        </div>
        """)
    
    # Session documents
    if st.session_state.documents_indexed or st.session_state.documents_processing or st.session_state.documents_failed:
        st.html('<h3 class="mediassist-sidebar__heading" style="margin-top: 1rem;">This Session</h3>')
        
        all_session_hashes = set(st.session_state.documents_indexed) | set(st.session_state.documents_processing) | set(st.session_state.documents_failed.keys())
        
        for fhash in all_session_hashes:
            status = get_document_status(fhash)
            status_label = {
                "indexed": ('<span class="mediassist-doc-item__status mediassist-doc-item__status--indexed">Indexed</span>', ICON_CHECK),
                "processing": ('<span class="mediassist-doc-item__status mediassist-doc-item__status--processing">Processing</span>', ICON_CLOCK),
                "error": (f'<span class="mediassist-doc-item__status mediassist-doc-item__status--error">Failed</span>', ICON_ALERT),
            }.get(status, ('<span class="mediassist-doc-item__status">Unknown</span>', ICON_DOCUMENT))
            
            status_html, icon = status_label
            error_msg = st.session_state.documents_failed.get(fhash, "")
            
            st.html(f"""
            <div class="mediassist-doc-item">
                <div class="mediassist-doc-item__icon">{icon}</div>
                <div class="mediassist-doc-item__content">
                    <p class="mediassist-doc-item__name">Document ({fhash[:8]}...)</p>
                    <div class="mediassist-doc-item__meta">
                        {status_html}
                        {f'<span style="color: var(--color-error); font-size: 0.65rem;">{error_msg[:50]}...</span>' if error_msg else ''}
                    </div>
                </div>
                <button class="mediassist-doc-item__remove" onclick="window.parent.postMessage({{type: \'remove_doc\', hash: \'{fhash}\'}}, \'*\')">{ICON_TRASH}</button>
            </div>
            """)
    
    st.html("<hr style='margin: 1.5rem 0; border-color: var(--color-border);'>")
    
    # Actions
    st.html('<div class="mediassist-sidebar__actions">')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Conversation", use_container_width=True, key="new_chat"):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.rerun()
    
    with col2:
        if st.button("Clear Knowledge Base", use_container_width=True, type="secondary", key="clear_kb"):
            try:
                vectorstore = get_vectorstore()
                vectorstore.delete_collection()
                st.session_state.documents_indexed.clear()
                st.session_state.documents_processing.clear()
                st.session_state.documents_failed.clear()
                st.session_state.pipeline = get_cached_pipeline()
                st.toast("Knowledge base cleared", icon="🗑️")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear knowledge base: {e}")
    
    if st.session_state.pipeline_error:
        if st.button("Retry Ollama Connection", use_container_width=True, key="retry_ollama"):
            reset_pipeline()
            st.rerun()
    
    st.html("</div>")
    
    st.caption("Built with LangChain, ChromaDB, Ollama & Streamlit")

def render_welcome():
    st.html(f"""
    <div class="mediassist-welcome">
        <div class="mediassist-welcome__icon">{ICON_MEDICAL_CROSS}</div>
        <h1 class="mediassist-welcome__title">How can I help you today?</h1>
        <p class="mediassist-welcome__description">
            MediAssist answers medical information questions using its available knowledge base.
            Upload medical documents to get grounded, cited answers from your trusted sources.
        </p>
        <p class="mediassist-welcome__privacy">
            🔒 Your documents are processed locally. No data leaves your infrastructure.
            For educational purposes only. Not a substitute for professional medical advice.
        </p>
        <div class="mediassist-suggestions">
    """)
    
    suggestions = [
        "What are the common symptoms of anemia?",
        "Explain how blood pressure is measured.",
        "What does this medical document say about diabetes?",
        "Summarize the key points in my uploaded document."
    ]
    
    for suggestion in suggestions:
        st.html(f"""
        <button class="mediassist-suggestion" onclick="window.parent.postMessage({{type: 'suggestion', text: '{suggestion.replace("'", "\\'")}'}}, '*')">
            {suggestion}
        </button>
        """)
    
    st.html("</div></div>")

def render_chat_messages():
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.html(f"""
            <div class="mediassist-message mediassist-message--user">
                <div class="mediassist-message__avatar">{ICON_MEDICAL_CROSS}</div>
                <div class="mediassist-message__bubble">
                    <div class="mediassist-message__content">{message["content"]}</div>
                </div>
            </div>
            """)
        else:
            st.html(f"""
            <div class="mediassist-message mediassist-message--assistant">
                <div class="mediassist-message__avatar">{ICON_MEDICAL_CROSS}</div>
                <div class="mediassist-message__bubble">
                    <div class="mediassist-message__content">{message["content"]}</div>
                </div>
            </div>
            """)
            
            if message.get("sources"):
                render_sources(message["sources"])

def render_loading():
    st.html(f"""
    <div class="mediassist-loading">
        <div class="mediassist-loading__avatar">{ICON_LOADER}</div>
        <div class="mediassist-loading__bubble">
            <div class="mediassist-loading__dots">
                <div class="mediassist-loading__dot"></div>
                <div class="mediassist-loading__dot"></div>
                <div class="mediassist-loading__dot"></div>
            </div>
        </div>
    </div>
    """)

def render_error(error_msg: str):
    st.html(f"""
    <div class="mediassist-error">
        {ICON_ALERT}
        <span>{error_msg}</span>
    </div>
    """)

def render_composer():
    disabled = st.session_state.pipeline is None or st.session_state.pipeline_error is not None
    
    # Use native Streamlit chat_input for proper functionality
    # The custom HTML composer is replaced by st.chat_input in main()
    # This function now only renders the disclaimer/hint
    st.markdown(f"""
    <div class="mediassist-composer">
        <div class="mediassist-composer__inner">
            <p class="mediassist-composer__hint">Press Enter to send · Shift+Enter for new line</p>
            <p class="mediassist-composer__disclaimer">For educational purposes only. Not a substitute for professional medical advice.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_safety_notice():
    st.html(f"""
    <div class="mediassist-safety">
        <div class="mediassist-safety__inner">
            {ICON_SHIELD}
            <div>
                <strong>Medical Safety Notice:</strong> This application provides educational information only, not medical diagnoses or treatment recommendations.
                It does not replace a licensed healthcare professional. For urgent symptoms (chest pain, difficulty breathing, severe bleeding, loss of consciousness, stroke signs),
                call emergency services immediately. Do not delay emergency care to consult this chatbot.
            </div>
        </div>
    </div>
    """)

# ============================================================
# JAVASCRIPT FOR INTERACTIONS
# ============================================================
def inject_javascript():
    st.html("""
    <script>
    (function() {
        // Handle suggestion clicks
        window.addEventListener('message', function(event) {
            if (event.data.type === 'suggestion') {
                const textarea = document.getElementById('chat-input');
                const sendBtn = document.getElementById('send-btn');
                if (textarea && sendBtn) {
                    textarea.value = event.data.text;
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    sendBtn.click();
                }
            }
        });
        
        // Auto-resize textarea
        const textarea = document.getElementById('chat-input');
        if (textarea) {
            textarea.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 160) + 'px';
            });
            
            // Handle Enter/Shift+Enter
            textarea.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const sendBtn = document.getElementById('send-btn');
                    if (sendBtn && !sendBtn.disabled) {
                        sendBtn.click();
                    }
                }
            });
        }
        
        // Scroll to bottom on new messages
        const observer = new MutationObserver(() => {
            const chatContainer = document.querySelector('.mediassist-chat__messages');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
        
        const target = document.querySelector('.main');
        if (target) {
            observer.observe(target, { childList: true, subtree: true });
        }
    })();
    </script>
    """)

# ============================================================
# MAIN
# ============================================================
def main():
    initialize_session_state()
    get_pipeline()  # Initialize pipeline
    
    inject_javascript()
    render_header()
    
    # Main layout
    st.html('<div class="mediassist-layout">')
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
    
    # Main chat area
    st.html('<div class="mediassist-main">')
    
    # Chat messages container
    st.html('<div class="mediassist-chat">')
    st.html('<div class="mediassist-chat__messages">')
    
    if st.session_state.show_welcome and not st.session_state.messages:
        render_welcome()
    else:
        render_chat_messages()
        
        # Show loading indicator if last message was user and we're processing
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            # Check if assistant hasn't responded yet
            if len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "user":
                render_loading()
    
    st.html('</div>')  # chat__messages
    st.html('</div>')  # chat
    
    # Composer
    render_composer()
    
    # Safety notice
    render_safety_notice()
    
    st.html('</div>')  # main
    st.html('</div>')  # layout
    
    # Handle chat input via Streamlit's chat_input (hidden but functional)
    prompt = st.chat_input("Ask a medical question...", disabled=st.session_state.pipeline is None, key="chat_input_main")
    if prompt:
        process_user_message(prompt)
        st.rerun()

if __name__ == "__main__":
    main()