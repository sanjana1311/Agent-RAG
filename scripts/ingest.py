from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Robust .env loading from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_docs():
    docs = []
    for p in DATA_DIR.glob("*.md"):
        docs.extend(TextLoader(str(p), autodetect_encoding=True).load())
    return docs

def main():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(INDEX_DIR / "pm_faiss_index")
    print(f"Ingestion complete. Chunks: {len(chunks)} | Index: {INDEX_DIR/'pm_faiss_index'}")

if __name__ == "__main__":
    main()
