from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
INDEX_PATH = PROJECT_ROOT / "indexes" / "pm_faiss_index"

SYSTEM_PROMPT = "You are a precise PM coach. Use ONLY the context. If missing, say so. Be concise with bullets."

def retrieve(q, k=2):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db  = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k":k}).invoke(q)

def main(q):
    docs = retrieve(q, k=2)
    context = "\n\n".join(f"[{i+1}] {d.metadata.get('source','(unknown)')}\n{d.page_content}" for i,d in enumerate(docs))
    prompt = f"""{SYSTEM_PROMPT}

# Context
{context}

# Question
{q}
"""
    llm = ChatOllama(
        model=os.getenv("LOCAL_MODEL", "phi3:mini"),
        base_url="http://127.0.0.1:11434",
        temperature=0,
        num_ctx=2048,
        num_predict=256,
        keep_alive="30m",
    )
    print("\n📓 Answer:\n")
    for chunk in llm.stream(prompt):
        print(getattr(chunk, "content", str(chunk)), end="", flush=True)
    print("\n\n📚 Sources:")
    for i,d in enumerate(docs,1):
        print(f"  [{i}] {d.metadata.get('source','(unknown)')}")
if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        print('Usage: python scripts/query_ollama_stream.py "your question"'); raise SystemExit(1)
    main(sys.argv[1])
