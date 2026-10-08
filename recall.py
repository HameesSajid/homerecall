"""HomeRecall command line.

    python recall.py index <folder of recordings>     build or update the index
    python recall.py ask "did anyone take eggs out of the fridge yesterday"
    python recall.py ask                               keep asking, models stay loaded
    python recall.py log --date 2026-10-06             what the camera saw that day
    python recall.py stats                             what is in the index
"""

import argparse
import datetime as dt
import json
import os
import sys

from homerecall import paths  # sets HF_HOME; keep above the model libraries
from homerecall.config import PROFILES, pick


def build_parser():
    ap = argparse.ArgumentParser(description="Search home security recordings in plain language.")
    ap.add_argument("--index-dir", default=paths.INDEX, help="where the index lives (default: ./index)")
    ap.add_argument("--profile", default="auto", choices=["auto", *PROFILES], help="model set (default: by GPU memory)")
    ap.add_argument("--vlm", default=None, help="override the profile's language model")
    ap.add_argument("--detector", default=None, help="override the profile's YOLOE weights")
    ap.add_argument("--allow-download", action="store_true",
                    help="let the laptop profile download a model that is not on disk")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="index new or changed recordings in a folder")
    p.add_argument("folder")
    p.add_argument("--limit", type=int, default=None, help="index at most this many recordings")
    p.add_argument("--reindex", action="store_true", help="redo recordings that are already indexed")
    p.add_argument("--no-captions", action="store_true", help="skip the language model (detector tags only)")

    p = sub.add_parser("ask", help="ask a question; with no question, keeps asking")
    p.add_argument("question", nargs="?")
    p.add_argument("--date", help="only this day, YYYY-MM-DD")
    p.add_argument("--since", help="only from this time, YYYY-MM-DD or 'YYYY-MM-DD HH:MM'")
    p.add_argument("--until", help="only before this time")
    p.add_argument("--top", type=int, default=5, help="results to show")
    p.add_argument("--verify", default=None, help="how many candidates the language model checks, or 'all'")
    p.add_argument("--threshold", type=float, default=0.5, help="confidence needed to count as a match")
    p.add_argument("--no-verify", action="store_true", help="retrieval only, no language model")
    p.add_argument("--no-probe", action="store_true", help="do not run the detector for new question words")
    p.add_argument("--no-clips", action="store_true", help="do not cut the matching clips out")
    p.add_argument("--open", action="store_true", help="open the best match when done")

    p = sub.add_parser("log", help="list indexed moments with their captions")
    p.add_argument("--date", help="only this day, YYYY-MM-DD")

    sub.add_parser("stats", help="summary of the index")
    return ap


class Models:
    """Loads each model the first time it is asked for."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._cache = {}

    def _get(self, key, make):
        if key not in self._cache:
            print(f"Loading {key}...", flush=True)
            self._cache[key] = make()
        return self._cache[key]

    def detector(self):
        from homerecall.detector import OpenVocabDetector

        return self._get(f"detector {self.cfg.detector}",
                         lambda: OpenVocabDetector(self.cfg.detector, imgsz=self.cfg.imgsz))

    def vlm(self):
        from homerecall.vlm import load_vlm

        return self._get(f"language model {self.cfg.vlm}", lambda: load_vlm(self.cfg.vlm, max_side=self.cfg.vlm_side))

    def embedder(self):
        from homerecall.embed import Embedder

        return self._get(f"embedder {self.cfg.embedder}", lambda: Embedder(self.cfg.embedder))


def cmd_index(args, cfg, store):
    from homerecall.indexer import index_folder

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a folder: {args.folder}")
    models = Models(cfg)
    done, moments = index_folder(
        args.folder, store, cfg, models.detector(),
        vlm=None if args.no_captions else models.vlm(),
        embedder=models.embedder() if cfg.embedder else None,
        limit=args.limit, reindex=args.reindex)
    print(f"Indexed {done} recordings, {moments} moments. Index: {store.root}")


def cmd_ask(args, cfg, store):
    from homerecall import report
    from homerecall.search import Searcher, parse_day

    start = parse_day(args.since) if args.since else None
    end = parse_day(args.until) if args.until else None
    if args.date:
        start = parse_day(args.date)
        end = start + dt.timedelta(days=1)
    models = Models(cfg)
    has_embeddings = store.stats()["with_embeddings"] > 0
    searcher = Searcher(store, cfg, detector=None if args.no_probe else models.detector,
                        vlm=None if args.no_verify else models.vlm,
                        embedder=models.embedder if (cfg.embedder and has_embeddings) else None)
    if has_embeddings and not cfg.embedder:
        print("Note: this index has embeddings, but the current profile has no embedder, so they are not used.")

    def answer(question):
        result = searcher.search(question, start, end, verify=args.verify, threshold=args.threshold,
                                 probe=not args.no_probe)
        if result["matches"]:
            report.save(result, store, report.result_folder(paths.RESULTS, question), clips=not args.no_clips)
        print()
        print(report.format_result(result, args.top))
        if args.open and result["matches"]:
            best = result["matches"][0]
            os.startfile(best.get("cut") or best["clip_path"])  # noqa: S606 (Windows only)

    if args.question:
        return answer(args.question)
    print("Ask about your recordings. Empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        answer(question)


def cmd_log(args, store):
    from homerecall.search import parse_day

    start = parse_day(args.date) if args.date else None
    end = start + dt.timedelta(days=1) if start else None
    day = None
    for w in store.windows(start, end):
        if w["start"][:10] != day:
            day = w["start"][:10]
            print(f"\n{dt.date.fromisoformat(day):%A %d %B %Y}")
        tags = [k for k, v in json.loads(w["objects"]).get("tags", {}).items() if v >= 0.25 and k != "person"]
        print(f"  {w['start'][11:]} to {w['end'][11:]}  {w['caption'] or '(no caption)'}"
              + (f"  [{', '.join(tags[:6])}]" if tags else ""))


def main():
    args = build_parser().parse_args()
    from homerecall.store import Store

    if args.command in ("log", "stats"):
        store = Store(args.index_dir)
        if args.command == "stats":
            print(json.dumps(store.stats(), indent=2))
        else:
            cmd_log(args, store)
        return
    cfg = pick(args.profile, vlm=args.vlm, detector=args.detector)
    if not cfg.allow_download and not args.allow_download:
        paths.forbid_downloads()
    print(f"Profile: {cfg.name}  (models from {paths.CACHE})")
    store = Store(args.index_dir)
    (cmd_index if args.command == "index" else cmd_ask)(args, cfg, store)


if __name__ == "__main__":
    main()
