import streamlit as st
from utils import index_file, run_chat, preload, load_registry

st.set_page_config(page_title="CDASH–SDTM Assistant", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .card-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 4px; color: #1a1a2e; }
    .card-subtitle { font-size: 0.85rem; color: #888; margin-bottom: 20px; }
    .indexed-badge {
        display: inline-block; background: #e8f5e9; color: #2e7d32;
        border-radius: 20px; padding: 4px 12px; font-size: 0.78rem; margin: 3px 2px;
    }
    div[data-testid="stChatMessage"] { border-radius: 12px; margin-bottom: 8px; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    h1 { color: #1a1a2e !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🧬 CDASH → SDTM Assistant")
st.caption("Powered by RAG | Ollama | FAISS")
st.divider()

with st.spinner("⚙️ Loading models..."):
    preload()

left, right = st.columns([1, 1], gap="large")

# ── LEFT: KNOWLEDGE BASE ──────────────────────
with left:
    st.markdown("""
        <div class="card-title">📚 Knowledge Base</div>
        <div class="card-subtitle">Upload documents to build the knowledge base</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF, CSV or Excel",
        type=["pdf", "csv", "xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        existing = {r["name"] for r in load_registry()}
        for uploaded_file in uploaded_files:
            if uploaded_file.name in existing:
                continue

            st.markdown(f"**Processing:** `{uploaded_file.name}`")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct, text):
                progress_bar.progress(pct)
                status_text.markdown(f"**{pct}%** — {text}")

            update_progress(0, "Starting...")
            result = index_file(uploaded_file, update_progress)

            if result["success"]:
                progress_bar.progress(100)
                status_text.markdown("**100%** — ✅ Done!")
                st.success(f"✅ {uploaded_file.name} — {result['chunks']} chunks indexed.")
                st.rerun()
            else:
                status_text.markdown("**Failed** ❌")
                st.error(f"❌ {result['error']}")

    st.markdown("---")
    st.markdown("**📂 Knowledge Base Documents**")
    records = load_registry()
    if records:
        for rec in records:
            col_a, col_b, col_c = st.columns([3, 1, 2])
            col_a.markdown(f'<span class="indexed-badge">✅ {rec["name"]}</span>', unsafe_allow_html=True)
            col_b.caption(f'{rec["chunks"]} chunks')
            col_c.caption(f'🕒 {rec["indexed_at"]}')
    else:
        st.info("No documents indexed yet. Upload a file to get started.")

# ── RIGHT: CHATBOT ────────────────────────────
with right:
    st.markdown("""
        <div class="card-title">💬 Chat Assistant</div>
        <div class="card-subtitle">Ask questions about CDASH/SDTM variables and domains</div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.container(height=420):
        if not st.session_state.chat_history:
            st.markdown("""
                <div style='text-align:center; color:#aaa; margin-top:80px;'>
                    💬 Ask something like:<br><br>
                    <i>"What domain does AEDECOD belong to?"</i><br>
                    <i>"Explain AE domain variables"</i><br>
                    <i>"Difference between AETERM and AEDECOD"</i>
                </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📚 Sources"):
                            for src in msg["sources"]:
                                st.caption(f"• {src}")

    if st.button("🗑️ Clear Chat", help="Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    user_input = st.chat_input("Ask about CDASH/SDTM...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = run_chat(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.get("answer", "Sorry, I could not find an answer."),
            "sources": response.get("sources", []),
        })
        st.rerun()
