from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# You can keep this import; it works fine:
from langchain_community.embeddings import HuggingFaceEmbeddings
# If you want the “new” import, swap the line above for:
# from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
INDEX_PATH = PROJECT_ROOT / "indexes" / "pm_faiss_index"

def main(q: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = vectordb.similarity_search(q, k=4)

    print(f"\nQuery: {q}\n")
    print("📚 Top chunks:")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] Source: {d.metadata.get('source','(unknown)')}\n{d.page_content[:800]}\n" + "-"*70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python scripts/query_no_llm.py "your question"'); raise SystemExit(1)
    main(sys.argv[1])
