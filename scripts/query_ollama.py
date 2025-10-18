from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS

# Prefer the new package; fall back if not installed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
INDEX_PATH = PROJECT_ROOT / "indexes" / "pm_faiss_index"

SYSTEM_PROMPT = """You are a precise PM coach.
Answer ONLY using the provided context. If the answer is not clearly in the context, say you don't see it in the docs.
Be concise, structured, and actionable. Include a short bullet list of key takeaways.
"""

def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "(unknown)")
        parts.append(f"[{i}] {src}\n{d.page_content}")
    return "\n\n".join(parts)

def retrieve(q: str, k: int = 4):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    # Modern LC API prefers .invoke over .get_relevant_documents
    return retriever.invoke(q)

def main(q: str, k: int = 4):
    docs = retrieve(q, k=k)
    context = build_context(docs)

    prompt = f"""{SYSTEM_PROMPT}

# Context
{context}

# Question
{q}
"""

    llm = ChatOllama(model="llama3.1:8b", temperature=0, base_url="http://127.0.0.1:11434")
    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    print("\n🧠 Answer:\n", answer)
    print("\n📚 Sources:")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] {d.metadata.get('source','(unknown)')}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python scripts/query_ollama.py "your question"'); raise SystemExit(1)
    main(sys.argv[1])
