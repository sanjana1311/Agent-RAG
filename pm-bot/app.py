import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# LangChain / RAG imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory



# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "PM Bot — Local RAG (FAISS + Ollama)"
DB_DIR = Path("./vectorstore")
DOCS_DIR = Path("./docs")  # put your existing PM notes here
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3:mini"  # e.g., "phi3:mini", "llama3:8b", "mistral"
TOP_K = 4  # retrieved chunks

# ---------------------------
# Helpers
# ---------------------------
def pii_mask(text: str) -> str:
    """Very simple PII masking to avoid echoing emails/phones/addresses in UI/logs."""
    # emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    # phone numbers (loose)
    text = re.sub(r"\+?\d[\d\-\s()]{7,}\d", "[PHONE]", text)
    # VIN last 6 (common in auto contexts) — tweak as needed
    text = re.sub(r"\b[A-HJ-NPR-Z0-9]{6}\b", "[VIN6]", text)
    return text

def load_docs_from_folder(folder: Path) -> List[Document]:
    docs: List[Document] = []
    if not folder.exists():
        return docs

    for p in folder.rglob("*"):
        if p.is_dir():
            continue
        try:
            if p.suffix.lower() in [".pdf"]:
                loader = PyPDFLoader(str(p))
                docs.extend(loader.load())
            elif p.suffix.lower() in [".md"]:
                loader = UnstructuredMarkdownLoader(str(p))
                docs.extend(loader.load())
            elif p.suffix.lower() in [".txt", ".mdown", ".rtf"]:
                loader = TextLoader(str(p), encoding="utf-8")
                docs.extend(loader.load())
            # add .docx support if you like:
            # elif p.suffix.lower() in [".docx"]:
            #     from langchain_community.document_loaders import Docx2txtLoader
            #     loader = Docx2txtLoader(str(p))
            #     docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Skipping {p.name}: {e}")
    return docs

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_or_load_vectorstore(
    embedder: HuggingFaceEmbeddings,
    base_docs: List[Document],
    persist_dir: Path,
) -> FAISS:
    persist_dir.mkdir(parents=True, exist_ok=True)

    if any(persist_dir.iterdir()):
        # Load existing
        vs = FAISS.load_local(str(persist_dir), embedder, allow_dangerous_deserialization=True)
    else:
        # Build from scratch
        chunks = split_docs(base_docs)
        vs = FAISS.from_documents(chunks, embedder)
        vs.save_local(str(persist_dir))
    return vs

def upsert_uploaded_files(files, embedder, vs: FAISS):
    """Embed & upsert uploaded files into existing FAISS index (and save)."""
    tmp_docs = []
    for f in files:
        suffix = Path(f.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp.flush()
            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp.name)
                elif suffix in [".md"]:
                    loader = UnstructuredMarkdownLoader(tmp.name)
                else:
                    loader = TextLoader(tmp.name, encoding="utf-8")
                tmp_docs.extend(loader.load())
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")

    if tmp_docs:
        chunks = split_docs(tmp_docs)
        vs.add_documents(chunks)
        vs.save_local(str(DB_DIR))

def ensure_local_ollama_running() -> None:
    # Simple hint. If it fails at runtime, Streamlit will show the exception.
    # You can also check OLLAMA_HOST env var if non-default.
    pass

def make_chain(vs: FAISS, model_name: str) -> ConversationalRetrievalChain:
    # LLM used for both answering and summarizing the chat history
    llm = Ollama(model=model_name, temperature=0.2, num_ctx=4096)



    # Summarized memory keeps context compact; output_key tells memory what to save
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain



def render_sources(source_docs: List[Document]):
    if not source_docs:
        st.info("No sources retrieved.")
        return
    with st.expander("Show sources"):
        for i, d in enumerate(source_docs, 1):
            meta = d.metadata or {}
            source_label = meta.get("source", meta.get("file_path", ""))
            st.markdown(f"**{i}. {Path(str(source_label)).name}**")
            st.write(pii_mask(d.page_content[:1000]))

# ---------------------------

# ---------------------------
# Streamlit App (final theme + header)
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧋", layout="wide")

# ---- Global theme overrides (force full white + beige accents) ----
st.markdown("""
<style>
:root {
  --latte: #b1845b;         /* main accent */
  --latte-dark: #9a6f48;    /* hover */
  --ink: #222222;           /* primary text */
  --text: #2b2b2b;          /* body text */
  --muted: #7b6f65;         /* secondary text */
  --border: #ede4da;        /* dividers */
  --card: #fcfbf9;          /* card bg */
  --chip: #f9f7f4;          /* expander/chips */
}

/* 1) FORCE FULL WHITE APP (works even if Streamlit dark theme is on) */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"], .stApp, .main, .block-container {
  background: #ffffff !important;
  color: var(--text) !important;
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* 2) Header + footer bars */
header[data-testid="stHeader"] {
  background: #ffffff !important;
  color: var(--text) !important;
  border-bottom: 1px solid #f0e8e0;
}
footer, .st-emotion-cache-1y4p8pa, .st-emotion-cache-12fmjuu, .viewerBadge_container__1QSob {
  background: #ffffff !important;
  color: var(--muted) !important;
  border-top: 1px solid #f0e8e0;
}

/* 3) Sidebar */
[data-testid="stSidebar"] {
  background: #ffffff !important;
  color: var(--text) !important;
  border-right: 1px solid #f5f0ea;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
  color: var(--text) !important;
}

/* 4) Title + tagline */
h1 { color: var(--ink) !important; font-weight: 700 !important; }
h4 { color: var(--latte) !important; font-style: italic; }

/* 5) Inputs */
input, textarea { border-radius: 10px !important; border: 1px solid #ddd !important; }

/* 6) Primary button (Ask) */
button[kind="primary"] {
  background: var(--latte) !important;
  color: #fff !important;
  border-radius: 12px !important;
  padding: 0.6rem 1.2rem !important;
  font-weight: 500 !important;
  border: none !important;
}
button[kind="primary"]:hover {
  background: var(--latte-dark) !important;
  transition: all .2s ease-in-out;
}

/* 7) Expanders, cards, chat bubbles */
[data-testid="stExpander"] {
  background: var(--chip);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.answer-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem 1.2rem;
  box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}
.msg { padding: .9rem 1.1rem; border-radius: 12px; margin: .45rem 0; }
.human { background:#f5ebe0; color: var(--text); }
.bot   { background:#faf9f7; color: var(--text); border: 1px solid var(--border); }

/* 8) Links + divider */
a { color: var(--latte) !important; text-decoration: none !important; }
a:hover { text-decoration: underline !important; }
hr { border: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ---- SINGLE title + tagline (remove any other duplicates above/below) ----
st.title(APP_TITLE)
st.markdown("""
<div style='margin-top:-10px; margin-bottom:20px;'>
  <h4>
    Your PM assistant — sip 🧋, ask, and plan.
  </h4>
  <p style='color:#7b6f65; font-size:14px; margin-top:-8px;'>
    All your PM guide, helping you navigate your career.
  </p>
  <hr style='border: 1px solid #ede4da; margin-top:15px;'>
</div>
""", unsafe_allow_html=True)


# Sidebar: Model & Data Controls
with st.sidebar:
    st.header("Settings")
    model_choice = st.text_input("Ollama model", value=OLLAMA_MODEL, help="e.g., phi3:mini, llama3:8b, mistral")
    st.write("**Vector DB location:**", str(DB_DIR.resolve()))
    # Vectorstore status badge
    try:
        _has_index = any(DB_DIR.iterdir())
    except Exception:
        _has_index = False
    st.markdown(
        ("<span class='badge ok'>Vectorstore ready</span>"
         if _has_index else
         "<span class='badge warn'>No index yet</span>"),
        unsafe_allow_html=True
    )

    if st.button("Rebuild index from ./docs"):
        st.session_state.pop("chain", None)
        # rebuild vectorstore
        st.info("Rebuilding vectorstore...")
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        base_docs = load_docs_from_folder(DOCS_DIR)
        if not base_docs:
            st.warning(f"No documents found in {DOCS_DIR}. Upload some below or add files to the folder.")
        vs = build_or_load_vectorstore(embedder, base_docs, DB_DIR)
        st.success("Vectorstore ready.")

    st.markdown("---")
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "Add PDFs / Markdown / TXT to your knowledge base",
        type=["pdf", "md", "txt", "rtf", "mdown"],
        accept_multiple_files=True,
    )
    if uploads:
        embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        try:
            vs = FAISS.load_local(str(DB_DIR), embedder, allow_dangerous_deserialization=True)
        except Exception:
            vs = build_or_load_vectorstore(embedder, load_docs_from_folder(DOCS_DIR), DB_DIR)
        upsert_uploaded_files(uploads, embedder, vs)
        st.success(f"Indexed {len(uploads)} file(s).")

# Init vectorstore & chain once
if "chain" not in st.session_state:
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    base_docs = load_docs_from_folder(DOCS_DIR)
    if not base_docs:
        st.warning(f"No documents found in {DOCS_DIR}. Upload some or click 'Browse files' in the sidebar.")
    else:
        vs = build_or_load_vectorstore(embedder, base_docs, DB_DIR)
        st.session_state["chain"] = make_chain(vs, model_choice)


# Chat UI
user_q = st.text_input("Ask your PM Bot a question…", placeholder="e.g., Draft a STAR story from my Tesla NLP project")
ask = st.button("Ask 💬", type="primary")
with st.expander("Try these"):
    st.code("Summarize my Tesla chatbot project in STAR format.")
    st.code("Create 3 resume bullets in XYZ style for the AI summarizer project.")
    st.code("List the KPIs we tracked and why they mattered.")


if ask and user_q.strip():
    try:
        chain = st.session_state["chain"]
        result: Dict[str, Any] = chain({"question": user_q.strip() + "\n\nPlease answer in English."})
        answer = pii_mask(result.get("answer", "")).strip()
        source_docs = result.get("source_documents", [])

        # ───────────────── Conversation Memory ─────────────────
        if "chain" in st.session_state and hasattr(st.session_state["chain"], "memory"):
            hist = st.session_state["chain"].memory.chat_memory.messages
            if hist:
                with st.expander("💬 Conversation Memory (last few turns)"):
                    for m in hist[-6:]:
                        who = "🧍‍♀️ You" if getattr(m, "type", "") == "human" else "🤖 PM Bot"
                        st.markdown(f"**{who}:** {m.content}")

        # ───────────── Chat bubble + Answer card (outside expander) ─────────────
        st.markdown(f"<div class='msg human'><b>You:</b> {pii_mask(user_q)}</div>", unsafe_allow_html=True)

        st.markdown("<div class='answer-card'><h3>Answer</h3>", unsafe_allow_html=True)
        st.markdown(pii_mask(answer).replace("\n","<br>"), unsafe_allow_html=True)

        if source_docs:
            chips = []
            for d in source_docs:
                meta = d.metadata or {}
                label = Path(str(meta.get("source", meta.get("file_path", "")))).name
                chips.append(f"<span class='src-chip'>{label}</span>")
            st.markdown("<div style='margin-top:10px;'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='opacity:0.7;font-size:13px;'>No sources retrieved.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # close answer-card

    except Exception as e:
        st.error(
            "There was an error generating an answer. "
            "Confirm Ollama is running and the model is pulled (e.g., `ollama pull phi3:mini`)."
        )
        st.exception(e)

st.markdown("---")

st.caption(
    "Tips: put your PM notes in `./docs`, click **Rebuild index**, and try: "
    "`Create resume bullets in XYZ for my Tesla intent taxonomy project.`"
)
