import os, json, uuid
from pathlib import Path
import streamlit as st

SESS_DIR = Path("sessions")
SESS_DIR.mkdir(exist_ok=True)

def _get_sid():
    if "session_id" not in st.session_state:
        q = dict(st.query_params)
        st.session_state.session_id = q.get("sid", str(uuid.uuid4()))
    return st.session_state.session_id

def _history_path():
    return SESS_DIR / f"{_get_sid()}.json"

def load_history():
    fp = _history_path()
    if fp.exists():
        try:
            st.session_state.messages = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            st.session_state.messages = []
    else:
        st.session_state.messages = []

def save_history():
    fp = _history_path()
    fp.write_text(
        json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

if "messages" not in st.session_state:
    load_history()

st.title("🤖 PM Bot (Notebook RAG)")
st.caption("Docs: data/*.md • Index: indexes/pm_faiss_index • Embeddings: all-MiniLM-L6-v2 • LLM: Ollama local or OpenAI")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_q = st.chat_input("Ask about your PM docs…")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    answer_text = f"(demo answer) You asked: {user_q}"
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    save_history()
    st.rerun()

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        save_history()
        st.rerun()

    if st.session_state.messages:
        md_text = "\n\n".join(
            f"**{m['role']}**: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "Download chat as .md",
            data=md_text,
            file_name="chat_history.md",
            mime="text/markdown",
        )
