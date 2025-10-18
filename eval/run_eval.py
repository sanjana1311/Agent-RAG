import csv, time, statistics as stats
from pathlib import Path
from datetime import datetime
import sys
# add project root to path so 'scripts' resolves
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.rag_core import answer_once

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "report.md"

def score(ans: str, expected_contains: str, sources: list[str], allowed_sources: str) -> int:
    toks = [t.strip().lower() for t in expected_contains.split(",") if t.strip()]
    ok_text = all(t in ans.lower() for t in toks)
    allow = [s for s in sources if any(a.strip() in s for a in allowed_sources.split(";"))]
    ok_src = len(allow) > 0
    return int(ok_text) + int(ok_src)  # 0..2

def main(provider="ollama", model=None, k=3):
    rows=[]
    path = ROOT / "eval" / "golden_set.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            t0=time.time()
            ans, srcs, meta = answer_once(r["question"], k=k, provider=provider, model=model)
            lat=time.time()-t0
            s = score(ans, r["expected_contains"], srcs, r["allowed_sources"])
            rows.append({"id":r["id"], "score":s, "latency":lat, "sources":"|".join(srcs)})
    avg_score = stats.mean([r["score"] for r in rows])
    avg_lat   = stats.mean([r["latency"] for r in rows])
    ts = datetime.utcnow().isoformat(timespec="seconds")+"Z"
    OUT.write_text(
        f"# Eval Report ({ts})\n\n- Provider: **{provider}**\n- Avg score (0..2): **{avg_score:.2f}**\n- Avg latency: **{avg_lat:.2f}s**\n\n## Details\n" +
        "\n".join(f"- {r['id']}: score={r['score']} latency={r['latency']:.2f}s sources={r['sources']}" for r in rows) + "\n"
    )
    print(f"Avg score: {avg_score:.2f} | Avg latency: {avg_lat:.2f}s")
    print(f"Wrote report -> {OUT}")

if __name__ == "__main__":
    main()
