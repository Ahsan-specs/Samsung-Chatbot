"""
Samsung Product Support AI — Streamlit Interface
Multi-modal AI assistant powered by Agentic RAG architecture.
"""

import streamlit as st
import os
import tempfile
import hashlib
from dotenv import load_dotenv

load_dotenv()

from src.document_processor import DocumentProcessor
from src.retriever import HybridRetriever
from src.multimodal import MultiModalProcessor
from src.agent import SupportAgent

# =====================================================================
#  PAGE CONFIG & CSS
# =====================================================================

st.set_page_config(
    page_title="Samsung AI Support",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --samsung-blue: #1428A0;
        --blue-light: #4169E1;
        --accent: #4169E1;
        --bg-dark: #0A0E1A;
        --bg-card: #111827;
        --text-primary: #F1F5F9;
        --text-muted: #94A3B8;
        --border: #1E293B;
        --green: #10B981;
        --yellow: #F59E0B;
        --red: #EF4444;
    }

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Header */
    .app-header {
        border-bottom: 1px solid var(--border);
        padding: 0.8rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    .app-header h1 {
        color: var(--text-primary);
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .app-header p {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin: 0.2rem 0 0 0;
        font-weight: 400;
    }

    /* Stats inline */
    .stats-bar {
        display: flex;
        gap: 1.5rem;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    .stats-bar .stat {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .stats-bar .stat-val {
        font-weight: 700;
        color: var(--text-primary);
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 8px !important;
        margin-bottom: 4px !important;
    }

    /* Response metadata */
    .response-meta {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-top: 0.6rem;
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .meta-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 3px;
    }
    .meta-tag {
        display: inline-block;
        background: rgba(65, 105, 225, 0.1);
        color: var(--blue-light);
        border: 1px solid rgba(65, 105, 225, 0.2);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.72rem;
        font-weight: 500;
    }

    /* Confidence bar */
    .conf-bar {
        width: 100%;
        height: 4px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
        margin-top: 4px;
    }
    .conf-fill {
        height: 100%;
        border-radius: 2px;
    }
    .conf-high { background: var(--green); }
    .conf-mid { background: var(--yellow); }
    .conf-low { background: var(--red); }

    /* Source tags */
    .src-tag {
        display: inline-block;
        background: rgba(65, 105, 225, 0.08);
        color: var(--blue-light);
        border: 1px solid rgba(65, 105, 225, 0.18);
        border-radius: 3px;
        padding: 1px 8px;
        font-size: 0.7rem;
        margin: 2px 3px 2px 0;
    }

    /* Sidebar overrides */
    .sidebar-heading {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .kb-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        font-size: 0.8rem;
    }
    .kb-stats .kv {
        color: var(--text-muted);
    }
    .kb-stats .kv strong {
        color: var(--text-primary);
    }

    /* Unified Chat Input Container */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) {
        position: fixed;
        bottom: 1.5rem;
        left: 50%;
        transform: translateX(-50%);
        width: 90%;
        max-width: 800px;
        background-color: #2D2D2D;
        border-radius: 12px;
        padding: 0.5rem 0.5rem 0.2rem 0.5rem;
        z-index: 999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Remove gaps injected by Streamlit in the container */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) > div {
        gap: 0 !important;
    }

    /* Style the Text Input row */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stTextInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-top: 0;
        padding-bottom: 0;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stTextInput"] input {
        color: #e0e0e0;
        font-size: 1rem;
    }

    /* Paperclip/Attach Button -> Styled as Plus icon on the left */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section {
        padding: 0 !important;
        background-color: transparent !important;
        border: none !important;
        min-height: auto !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section > div > div > span,
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section > div > div > small {
        display: none !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section button {
        background-color: transparent;
        color: #9e9e9e;
        border: none;
        box-shadow: none;
        padding: 0;
        min-width: unset;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section button:hover {
        background-color: #424242;
        color: #fff;
    }
    /* Hide the upload SVG icon */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] svg {
        display: none !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section button div p { font-size: 0; }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="stFileUploader"] section button div p::before {
        content: "＋";
        font-size: 1.5rem;
        font-weight: 300;
        display: block;
        line-height: 1;
    }

    /* Send Button */
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="baseButton-primary"] {
        background-color: transparent;
        border: none;
        border-radius: 50%;
        color: #e0e0e0;
        width: 38px;
        height: 38px;
        padding: 0;
        min-width: unset;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="baseButton-primary"]:hover {
        background-color: #424242;
        color: #fff;
    }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="baseButton-primary"] p { font-size: 0; }
    div[data-testid="stVerticalBlock"]:has(> div:first-child #chat-input-hook) [data-testid="baseButton-primary"] p::before {
        content: "➔";
        font-size: 1.2rem;
        display: block;
        line-height: 1;
    }


    /* Hide chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  SESSION STATE
# =====================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_system_processor():
    p = DocumentProcessor(output_dir="data/processed")
    loaded = p.load_kb()
    return p, loaded

if "processor" not in st.session_state:
    st.session_state.processor, st.session_state.kb_loaded = load_system_processor()

if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = None
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None
if "attached_image" not in st.session_state:
    st.session_state.attached_image = None
if "image_analysis" not in st.session_state:
    st.session_state.image_analysis = None
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# =====================================================================
#  HEADER
# =====================================================================

st.markdown("""
<div class="app-header">
    <h1>Samsung AI Support</h1>
    <p>Agentic RAG &middot; Multi-Modal &middot; 14 Product Categories</p>
</div>
""", unsafe_allow_html=True)


# =====================================================================
#  SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown('<div class="sidebar-heading">API Configuration</div>', unsafe_allow_html=True)
    cerebras_api_key = st.text_input(
        "Cerebras API Key",
        type="password",
        help="Leave blank if CEREBRAS_API_KEY is set in .env",
        placeholder="csk_..."
    )
    groq_api_key = st.text_input(
        "Groq API Key (for Voice/Vision)",
        type="password",
        help="Leave blank if GROQ_API_KEY is set in .env",
        placeholder="gsk_..."
    )

    st.divider()

    st.markdown('<div class="sidebar-heading">Knowledge Base</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Build KB", use_container_width=True,
                      help="Ingest JSON product files from data/ folder"):
            with st.spinner("Building knowledge base..."):
                count = st.session_state.processor.ingest_json_folder("data")
                if count > 0:
                    st.session_state.kb_loaded = True
                    st.session_state.kb_stats = st.session_state.processor.get_stats()
                    st.success(f"{count} products ingested")
                else:
                    st.warning("No JSON files found in data/ folder")
    with col_b:
        if st.button("Clear KB", use_container_width=True):
            import shutil
            if os.path.exists("data/processed"):
                shutil.rmtree("data/processed")
                os.makedirs("data/processed", exist_ok=True)
            st.session_state.processor = DocumentProcessor(output_dir="data/processed")
            st.session_state.kb_loaded = False
            st.session_state.kb_stats = None
            st.success("Cleared")
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=['pdf'],
        label_visibility="collapsed"
    )

    if st.button("Process PDFs", disabled=not uploaded_files, use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing..."):
                for uf in uploaded_files:
                    path = os.path.join("data", "raw_pdfs", uf.name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(uf.getbuffer())
                    st.session_state.processor.ingest_single_pdf(path)
                st.session_state.kb_loaded = True
                st.session_state.kb_stats = st.session_state.processor.get_stats()
            st.success(f"{len(uploaded_files)} PDFs processed")

    st.divider()

    if st.session_state.kb_loaded:
        if st.session_state.kb_stats is None:
            st.session_state.kb_stats = st.session_state.processor.get_stats()

        stats = st.session_state.kb_stats
        st.markdown('<div class="sidebar-heading">Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="kb-stats">
            <div class="kv"><strong>{stats['total_products']}</strong> products</div>
            <div class="kv"><strong>{stats['total_categories']}</strong> categories</div>
            <div class="kv"><strong>{stats['total_chunks']}</strong> chunks</div>
            <div class="kv"><strong>{stats['graph_nodes']}</strong> graph nodes</div>
        </div>
        """, unsafe_allow_html=True)

        if stats.get("categories"):
            with st.expander("Categories", expanded=False):
                for cat in stats["categories"]:
                    st.caption(cat)
    else:
        st.info("Knowledge base empty. Build from data or upload PDFs.")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.voice_text = None
        st.session_state.attached_image = None
        st.session_state.image_analysis = None
        st.rerun()

    st.divider()
    st.caption("Samsung AI Support v2.0")
    st.caption("Groq LLaMA 3.3 70B")


# =====================================================================
#  SYSTEM INIT
# =====================================================================

active_cerebras_key = cerebras_api_key or os.environ.get("CEREBRAS_API_KEY")
active_groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")

if not active_cerebras_key:
    st.error("Provide a Cerebras API Key in the sidebar or .env file to continue.")
    st.stop()

try:
    retriever = HybridRetriever(
        st.session_state.processor, top_k=5, similarity_threshold=0.18
    ) if st.session_state.kb_loaded else None
    agent = SupportAgent(api_key=active_cerebras_key, retriever=retriever)
    multimodal = MultiModalProcessor(api_key=active_groq_key)
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()


# =====================================================================
#  STATS BAR
# =====================================================================

if st.session_state.kb_loaded and st.session_state.kb_stats:
    s = st.session_state.kb_stats
    st.markdown(f"""
    <div class="stats-bar">
        <div class="stat"><span class="stat-val">{s['total_products']}</span> products</div>
        <div class="stat"><span class="stat-val">{s['total_chunks']}</span> chunks</div>
        <div class="stat"><span class="stat-val">{s['graph_nodes']}</span> nodes</div>
        <div class="stat"><span class="stat-val">{s['graph_edges']}</span> edges</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
#  CHAT HISTORY
# =====================================================================

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:var(--text-muted);">
        <p style="font-size:1.1rem; font-weight:500; color:var(--text-primary); margin-bottom:0.3rem;">
            How can I help you?
        </p>
        <p style="font-size:0.82rem;">
            Ask about Samsung products — specs, troubleshooting, comparisons.
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "image" in msg:
            st.image(msg["image"], width=300)
        if "metadata" in msg and msg["role"] == "assistant":
            meta = msg["metadata"]
            conf_value = 0
            try:
                conf_value = int(str(meta.get("confidence", "0")).replace("%", ""))
            except (ValueError, AttributeError):
                conf_value = 0
            conf_class = "conf-high" if conf_value >= 70 else ("conf-mid" if conf_value >= 40 else "conf-low")
            meta_html = f"""
            <div class="response-meta">
                <div class="meta-row">
                    <span class="meta-tag">{meta.get('intent', 'N/A')}</span>
                    <span class="meta-tag">{meta.get('tool_used', 'none')}</span>
                </div>
                <div class="meta-row">Confidence: <strong>{meta.get('confidence', 'N/A')}</strong></div>
                <div class="conf-bar"><div class="conf-fill {conf_class}" style="width:{conf_value}%"></div></div>
            """
            sources = meta.get("sources", [])
            if sources:
                meta_html += '<div style="margin-top:6px;">'
                for src in sources[:5]:
                    meta_html += f'<span class="src-tag">{src}</span>'
                meta_html += '</div>'
            meta_html += "</div>"
            st.markdown(meta_html, unsafe_allow_html=True)

# Spacer so content isn't hidden behind the input bar
st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)


# =====================================================================
#  UNIFIED INPUT BAR — Voice | Image | Text Input | Send — all one row
# =====================================================================

image_query_text = None
image_analysis_result = st.session_state.image_analysis
audio_query_text = st.session_state.voice_text

# Show pending attachments above the input bar
if st.session_state.voice_text or st.session_state.attached_image:
    pending_parts = []
    if st.session_state.voice_text:
        vt = st.session_state.voice_text
        display_vt = vt if len(vt) <= 50 else vt[:47] + "..."
        pending_parts.append(f'<span class="pending-tag pending-voice">Voice: "{display_vt}"</span>')
    if st.session_state.attached_image:
        pending_parts.append('<span class="pending-tag pending-img">Image attached</span>')
    st.markdown(' '.join(pending_parts), unsafe_allow_html=True)

with st.container():
    st.markdown('<div id="chat-input-hook"></div>', unsafe_allow_html=True)
    
    # 1. Text input spans the top of this container
    query = st.text_input(
        "Message",
        placeholder="Ask anything, @ to mention, / for workflows",
        key="msg_input",
        label_visibility="collapsed"
    )

    # 2. Controls on the bottom row: [Attach (+)] [Space] [Mic] [Send]
    col_attach, col_space, col_mic, col_send = st.columns([1, 12, 1.2, 1.2])

    with col_attach:
        uploaded_image = st.file_uploader(
            "Attach",
            type=['png', 'jpg', 'jpeg', 'webp'],
            key="image_upload",
            label_visibility="collapsed"
        )
        if uploaded_image:
            st.session_state.attached_image = uploaded_image

    with col_space:
        st.empty() # Placeholder to push icons to the right

    with col_mic:
        try:
            from audio_recorder_streamlit import audio_recorder
            # Neutral color = standard gray, recording color = red
            audio_bytes = audio_recorder(
                text="",
                recording_color="#EF4444",
                neutral_color="#9e9e9e",
                icon_size="1x",
                key="voice_recorder",
                pause_threshold=2.0,
                sample_rate=44100
            )
            if audio_bytes:
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                if audio_hash != st.session_state.last_audio_id and len(audio_bytes) > 4000:
                    st.session_state.last_audio_id = audio_hash
                    with st.spinner("Transcribing..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                            tmp.write(audio_bytes)
                            tmp_wav_path = tmp.name

                        transcribed = multimodal.transcribe_audio(tmp_wav_path)

                        try:
                            os.unlink(tmp_wav_path)
                        except OSError:
                            pass

                        if transcribed and not transcribed.startswith("Audio transcription failed"):
                            st.session_state.voice_text = transcribed
                            audio_query_text = transcribed
                            st.rerun()
                        else:
                            error_msg = transcribed.replace("Audio transcription failed: ", "") if transcribed else "Unknown error"
                            st.error(error_msg)
                            st.session_state.voice_text = None
                elif len(audio_bytes) <= 4000 and audio_hash != st.session_state.last_audio_id:
                    st.session_state.last_audio_id = audio_hash
                    st.warning("Too short")
        except ImportError:
            st.caption("No mic")
        except Exception:
            st.caption("--")

    with col_send:
        send_clicked = st.button("Send", use_container_width=True, type="primary")

# =====================================================================
#  QUERY PROCESSING
# =====================================================================

# Determine if we should process: Send clicked or Enter pressed with text
should_process = (send_clicked and query) or (query and not send_clicked) or audio_query_text

if st.session_state.attached_image and query:
    with st.spinner("Analyzing image..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(st.session_state.attached_image.getbuffer())
            tmp_img_path = tmp.name
        image_analysis_result = multimodal.process_image(tmp_img_path)
        image_query_text = f"[Image Analysis: {image_analysis_result}] {query}"
elif st.session_state.attached_image and not query and not audio_query_text:
    image_query_text = None

final_input = None
if query and image_query_text:
    final_input = image_query_text
elif query:
    final_input = query
elif audio_query_text:
    final_input = audio_query_text

if final_input:
    input_type = "text"
    if image_query_text:
        input_type = "image"
    elif audio_query_text:
        input_type = "voice"

    # Add user message
    if input_type == "image":
        display_text = query if query else "Image analysis"
        st.session_state.messages.append({
            "role": "user",
            "content": display_text,
            "image": st.session_state.attached_image
        })
        with st.chat_message("user"):
            st.markdown(display_text)
            if st.session_state.attached_image:
                st.image(st.session_state.attached_image, width=300)
    elif input_type == "voice":
        display_text = audio_query_text
        st.session_state.messages.append({"role": "user", "content": display_text})
        with st.chat_message("user"):
            st.markdown(display_text)
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

    # Agent response
    with st.chat_message("assistant"):
        full_response = ""
        meta_data = {}
        placeholder = st.empty()
        
        generator = agent.generate_response(
            final_input,
            st.session_state.messages,
            input_type=input_type,
            image_analysis=image_analysis_result
        )

        for item in generator:
            if item["type"] == "chunk":
                full_response += item["text"]
                placeholder.markdown(full_response + "▌")
            elif item["type"] == "metadata":
                meta_data = item
            elif item["type"] == "error":
                st.error(item["text"])
                full_response = item["text"]
        
        placeholder.markdown(full_response)

    intent = meta_data.get("intent", "Unknown")
    tool_used = meta_data.get("tool_used", "none")
    sources = meta_data.get("sources", [])
    is_relevant = meta_data.get("is_relevant", False)
    
    st.markdown("---")

    confidence = meta_data.get("confidence", "N/A")
    conf_value = 0
    try:
        if "%" in confidence:
            conf_value = int(confidence.replace("%", ""))
    except (ValueError, AttributeError):
        conf_value = 0

    conf_class = "conf-high" if conf_value >= 70 else ("conf-mid" if conf_value >= 40 else "conf-low")

    meta_html = f"""
    <div class="response-meta">
        <div class="meta-row">
            <span class="meta-tag">{intent}</span>
            <span class="meta-tag">{tool_used}</span>
        </div>
        <div class="meta-row" style="margin-top:4px;">
            Confidence: <strong>{confidence}</strong>
        </div>
        <div class="conf-bar">
            <div class="conf-fill {conf_class}" style="width:{conf_value}%"></div>
        </div>
    """
    
    if sources:
        meta_html += '<div style="margin-top:6px;">'
        for src in sources[:5]:
            meta_html += f'<span class="src-tag">{src}</span>'
        meta_html += '</div>'

    if not is_relevant and intent not in ["GREETING", "OUT_OF_SCOPE"]:
        meta_html += '<div class="meta-row" style="color:var(--red);margin-top:4px;">No matching content in knowledge base</div>'

    meta_html += "</div>"
    st.markdown(meta_html, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "metadata": meta_data
    })

    # Clear transient state
    st.session_state.voice_text = None
    st.session_state.attached_image = None
    st.session_state.image_analysis = None
