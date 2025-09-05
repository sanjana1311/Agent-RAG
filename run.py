#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from datetime import datetime
import yaml

def load_config(path="agent-rag/config/app.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def char_params_from_tokens(tokens: int):
    return max(200, tokens * 4)

def simple_chunk(text: str, chunk_chars: int, overlap_chars: int):
    chunks = []
    i, n = 0, len(text)
    while i < n:
        end = min(n, i + chunk_chars)
        chunks.append(text[i:end])
        if end == n: break
        i = max(end - overlap_chars, i + 1)
    return chunks

def naive_hybrid_retrieve(query: str, docs, top_k: int = 5):
    q = query.lower()
    scored = []
    for d in docs:
        score = sum(q.count(w) for w in d["text"].lower().split()[:80])
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def build_demo_if_missing(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    demo = (
        "Return Policy (v1)\n"
        "You can return an item within 30 days of purchase with a receipt. "
        "Used accessories and custom items may be excluded.\n\n"
        "Refunds are issued to the original payment method. "
        "See our help center for exceptions and processing timelines."
    )
    (raw_dir / "policy.txt").write_text(demo)

def prepare_chunks(raw_dir: Path, chunk_dir: Path, cfg):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    csize = char_params_from_tokens(cfg["data"]["chunk_size_tokens"])
    cover = char_params_from_tokens(cfg["data"]["chunk_overlap_tokens"])
    meta_fields = cfg["data"]["store_fields"]

    for path in raw_dir.glob("*.txt"):
        txt = path.read_text()
        parts = simple_chunk(txt, csize, cover)
        out = chunk_dir / f"{path.name}.jsonl"
        with out.open("w", encoding="utf-8") as w:
            for i, ch in enumerate(parts):
                rec = {
                    "id": f"{path.stem}-{i}",
                    "text": ch,
                    "title": path.stem,
                    "section": "main",
                    "url": "",
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                }
                keep = {"id": rec["id"], "text": rec["text"]}
                for f in meta_fields:
                    keep[f] = rec.get(f, "")
                w.write(json.dumps(keep) + "\n")

def load_chunks(chunk_dir: Path):
    records = []
    for p in chunk_dir.glob("*.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records

def answer_as_json(query: str, hits):
    citations = []
    for h in hits:
        citations.append({
            "id": h["id"],
            "url": h.get("url", ""),
            "spans": [[0, min(60, len(h["text"]))]]
        })
    return {
        "answer": "This is a scaffolded answer. Next step: swap in real LLM + guardrails.",
        "citations": citations,
        "confidence": 0.5,
        "policy": {"abstained": False, "reason": None}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="What is the return policy?")
    parser.add_argument("--config", default="agent-rag/config/app.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path("data/raw")
    chunk_dir = Path(cfg["data"]["chunks_dir"])

    build_demo_if_missing(raw_dir)
    prepare_chunks(raw_dir, chunk_dir, cfg)
    docs = load_chunks(chunk_dir)
    hits = naive_hybrid_retrieve(args.query, docs, top_k=int(cfg["retriever"]["top_k"]))
    print(json.dumps(answer_as_json(args.query, hits), indent=2))

if __name__ == "__main__":
    print("🚀 Agent-RAG scaffold starting…")
    main()
    print("✅ Done.")
