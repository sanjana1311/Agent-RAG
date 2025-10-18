import os, time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
INDEX_PATH = ROOT / "indexes" / "pm_faiss_index"
SYSTEM = "You are a precise PM coach. Use ONLY the provided context. If missing, say so. Be concise with bullets."

def _emb():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _load_db():
    return FAISS.load_local(INDEX_PATH, _emb(), allow_dangerous_deserialization=True)

def retrieve(q: str, k: int = 3):
    db = _load_db()
    docs = db.similarity_search(q, k=k)
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source",""), d.page_content[:120])
        if key in seen:
            continue
        seen.add(key); out.append(d)
    return out

def _build_context(docs):
    return "\n\n".join(f"[{i+1}] {d.metadata.get('source','(unknown)')}\n{d.page_content}" for i,d in enumerate(docs))

def answer_once(question: str, k: int = 3, provider: str = "ollama", model: str | None = None):
    t0 = time.time()
    docs = retrieve(question, k=k)
    ctx = _build_context(docs)
    prompt = f"""{SYSTEM}

# Context
{ctx}

# Question
{question}
"""
    if provider == "openai":
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0)
    else:
        model = model or os.getenv("LOCAL_MODEL", "phi3:mini")
        base_url = os.getenv("OLLAMA_BASE_URL","http://127.0.0.1:11434")
        llm = ChatOllama(model=model, base_url=base_url, temperature=0, num_ctx=2048, num_predict=512, keep_alive="20m")
    parts=[]
    for ch in llm.stream(prompt):
        parts.append(getattr(ch, "content", str(ch)))
    ans = "".join(parts)
    latency = time.time() - t0
    srcs = [d.metadata.get("source","(unknown)") for d in docs]
    meta = {"latency": latency, "k": k, "provider": provider, "model": model, "chars": len(ans)}
    return ans, srcs, meta
