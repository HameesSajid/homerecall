"""Measure search on your own recordings.

Write a small JSON file of questions and the recordings that should answer them:

    [
      {"question": "did anyone take eggs out of the fridge", "clips": ["20261006_074210.mp4"]},
      {"question": "did a dog come into the kitchen", "clips": []}
    ]

`clips` lists file names (any moment of a listed recording counts); an empty list means
the right answer is "no". Then:

    python tools/eval_search.py labels.json

Reported per question and overall:
- retrieval rank: where the first correct moment sat before verification (recall@k)
- verdict: whether the final yes/no was right, and whether the top match was a correct clip
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homerecall import paths  # noqa: E402  (sets HF_HOME before the model libraries)
from homerecall.config import PROFILES, pick  # noqa: E402


def evaluate(searcher, labels, threshold):
    rows = []
    for item in labels:
        want = {os.path.basename(c).lower() for c in item.get("clips", [])}
        r = searcher.search(item["question"], verify="all", threshold=threshold)
        name = lambda m: os.path.basename(m["clip_path"]).lower()  # noqa: E731
        by_retrieval = sorted(r["candidates"], key=lambda m: -m["retrieval"])
        rank = next((i + 1 for i, m in enumerate(by_retrieval) if name(m) in want), None)
        said_yes = bool(r["matches"])
        top_right = said_yes and name(r["matches"][0]) in want
        rows.append({
            "question": item["question"], "expected": "yes" if want else "no", "answered": "yes" if said_yes else "no",
            "verdict_correct": said_yes == bool(want) and (top_right or not want),
            "retrieval_rank": rank, "moments": r["searched"],
            "best_confidence": r["candidates"][0]["confidence"] if r["candidates"] else None,
            "false_matches": sum(1 for m in r["matches"] if name(m) not in want),
        })
    return rows


def summarise(rows):
    pos = [r for r in rows if r["expected"] == "yes"]
    neg = [r for r in rows if r["expected"] == "no"]
    at = lambda k: sum(1 for r in pos if r["retrieval_rank"] and r["retrieval_rank"] <= k) / max(1, len(pos))  # noqa: E731
    return {
        "questions": len(rows), "expect_yes": len(pos), "expect_no": len(neg),
        "verdict_accuracy": sum(r["verdict_correct"] for r in rows) / max(1, len(rows)),
        "found_when_present": sum(r["verdict_correct"] for r in pos) / max(1, len(pos)),
        "correct_no_when_absent": sum(r["verdict_correct"] for r in neg) / max(1, len(neg)),
        "retrieval_recall@1": at(1), "retrieval_recall@5": at(5), "retrieval_recall@10": at(10),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--index-dir", default=paths.INDEX)
    ap.add_argument("--profile", default="auto", choices=["auto", *PROFILES])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default=None, help="write the full report here as JSON")
    args = ap.parse_args()

    from homerecall.models import Models
    from homerecall.search import Searcher
    from homerecall.store import Store

    cfg = pick(args.profile)
    if not cfg.allow_download and not args.allow_download:
        paths.forbid_downloads()
    store, models = Store(args.index_dir), Models(cfg)
    has_emb = store.stats()["with_embeddings"] > 0
    searcher = Searcher(store, cfg, detector=models.detector, vlm=models.vlm,
                        embedder=models.embedder if (cfg.embedder and has_emb) else None, log=lambda *_: None)
    with open(args.labels, encoding="utf-8") as f:
        rows = evaluate(searcher, json.load(f), args.threshold)
    for r in rows:
        mark = "ok  " if r["verdict_correct"] else "MISS"
        print(f"{mark} expected {r['expected']:3} answered {r['answered']:3}  rank {str(r['retrieval_rank'] or '-'):>3}"
              f"  best {r['best_confidence']}  {r['question']}")
    summary = summarise(rows)
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"profile": cfg.name, "threshold": args.threshold, "summary": summary, "questions": rows}, f, indent=2)


if __name__ == "__main__":
    main()
