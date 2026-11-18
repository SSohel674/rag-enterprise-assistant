"""Measure retrieval quality (hit@k, MRR) and optionally answer accuracy.

    python eval/evaluate.py              # retrieval only (no LLM needed)
    python eval/evaluate.py --answers    # also check answers contain expected keywords (needs LLM)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import RAGPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    rag = RAGPipeline()
    rag.ingest_directory(Path(__file__).resolve().parent.parent / "data" / "sample_docs")
    data = json.loads((Path(__file__).parent / "eval_set.json").read_text())

    hits_at_k, rr, ans_ok = 0, 0.0, 0
    for item in data:
        hits = rag.retrieve(item["question"], args.k)
        rank = next((i + 1 for i, h in enumerate(hits) if h["source"] == item["expected_source"]), None)
        hits_at_k += rank is not None
        rr += 1 / rank if rank else 0
        line = f"rank={rank}  {item['question']}"
        if args.answers:
            res = rag.ask(item["question"], args.k)
            ok = all(k.lower() in res["answer"].lower() for k in item["expected_keywords"])
            ans_ok += ok
            line += f"  | answer_ok={ok} grounded={res['grounded']}"
        print(line)

    n = len(data)
    print(f"\nHit@{args.k}: {hits_at_k / n:.2%}   MRR: {rr / n:.3f}")
    if args.answers:
        print(f"Answer accuracy (keyword match): {ans_ok / n:.2%}")


if __name__ == "__main__":
    main()
