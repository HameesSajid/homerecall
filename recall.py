"""HomeRecall command line.

    python recall.py camera add kitchen                set up a camera (asks for its address)
    python recall.py camera check kitchen              test the connection, save a snapshot
    python recall.py watch kitchen                     record when a person appears, and index
    python recall.py index <folder of recordings>      index recordings exported from an app
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
from homerecall.models import Models


def say(*parts):
    print(*parts, flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description="Search home security recordings in plain language.")
    ap.add_argument("--index-dir", default=paths.INDEX, help="where the index lives (default: ./index)")
    ap.add_argument("--profile", default="auto", choices=["auto", *PROFILES], help="model set (default: by GPU memory)")
    ap.add_argument("--vlm", default=None, help="override the profile's language model")
    ap.add_argument("--detector", default=None, help="override the profile's YOLOE weights")
    ap.add_argument("--allow-download", action="store_true",
                    help="let the laptop profile download a model that is not on disk")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("camera", help="set up, list or test cameras")
    p.add_argument("action", choices=["add", "list", "check", "remove"])
    p.add_argument("name", nargs="?")

    p = sub.add_parser("watch", help="record from a camera whenever a person appears, and index the clips")
    p.add_argument("camera", nargs="?", help="camera name from 'camera add' (optional when there is only one)")
    p.add_argument("--source", help="instead of a named camera: a stream address, a webcam number, or a video file")
    p.add_argument("--recordings", help="where clips are saved (default: ./recordings/<camera>)")
    p.add_argument("--no-index", action="store_true", help="only record; index later with 'index'")
    p.add_argument("--no-captions", action="store_true", help="index without the language model")
    p.add_argument("--pre", type=float, default=3.0, help="seconds kept from before the person appeared")
    p.add_argument("--post", type=float, default=5.0, help="seconds recorded after the person was last seen")
    p.add_argument("--max-clip", type=float, default=120.0, help="longest clip, in seconds, before a new one starts")
    p.add_argument("--min-free-gb", type=float, default=10.0, help="stop recording when the disk has less free space")

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


# ---------------------------------------------------------------- cameras

def cmd_camera(args):
    from homerecall import cameras

    known = cameras.load()
    if args.action == "list":
        if not known:
            say("No cameras yet. Add one with: python recall.py camera add kitchen")
        for name, url in known.items():
            say(f"{name:12} {cameras.mask(url)}")
        return
    if args.action == "add":
        import getpass

        name = args.name or input("A short name for this camera (for example kitchen): ").strip()
        if not name:
            sys.exit("A name is needed.")
        say("Enter the camera's address on your network and the Camera Account you created in its app.")
        host = input("Camera IP address (for example 192.168.1.50): ").strip()
        user = input("Camera account username: ").strip()
        password = getpass.getpass("Camera account password (not shown while typing): ")
        if not (host and user and password):
            sys.exit("All three are needed; nothing was saved.")
        known[name] = cameras.rtsp_url(host, user, password)
        cameras.save(known)
        say(f"Saved {name} as {cameras.mask(known[name])} in {cameras.CONFIG}")
        say(f"Test it with: python recall.py camera check {name}")
        return
    if args.action == "remove":
        if args.name not in known:
            sys.exit(f"No camera called {args.name!r}.")
        del known[args.name]
        cameras.save(known)
        say(f"Removed {args.name}.")
        return
    # check
    import cv2

    from homerecall.recorder import FrameSource

    try:
        name, url = cameras.resolve(args.name)
    except LookupError as e:
        sys.exit(str(e))
    say(f"Connecting to {name} at {cameras.mask(url)} ...")
    src = FrameSource(url, log=say)
    src.start()
    try:
        if not src.connected.wait(25):
            sys.exit("No connection after 25 seconds. Check the IP address, the username and password, that the "
                     "camera and this PC are on the same network, and that the Camera Account is switched on.")
        _, frame = src.frames.get(timeout=15)
    except KeyboardInterrupt:
        sys.exit(1)
    finally:
        src.stop()
    os.makedirs(paths.RESULTS, exist_ok=True)
    shot = os.path.join(paths.RESULTS, f"camera_{name}.jpg")
    cv2.imwrite(shot, frame)
    say(f"Connected: {frame.shape[1]}x{frame.shape[0]} at {src.fps:.0f} frames per second.")
    say(f"Snapshot saved to {shot}")


def cmd_watch(args, cfg):
    from homerecall import cameras
    from homerecall.recorder import IndexWorker, watch

    if args.source:
        source = int(args.source) if args.source.isdigit() else args.source
        name = args.camera or "camera"
    else:
        try:
            name, source = cameras.resolve(args.camera)
        except LookupError as e:
            sys.exit(str(e))
    if isinstance(source, str) and os.path.isfile(source):
        source = os.path.abspath(source)
    folder = os.path.abspath(args.recordings or os.path.join(paths.REPO, "recordings", name))
    index_dir = os.path.abspath(args.index_dir)
    detector = Models(cfg, log=say).detector()  # loaded before any thread starts
    worker = None
    if not args.no_index:
        worker = IndexWorker(index_dir, cfg, captions=not args.no_captions, log=say)
        worker.start()
    say(f"Clips go to {folder}")
    is_file = isinstance(source, str) and os.path.isfile(source)
    saved = watch(source, folder, detector, on_clip=worker.add if worker else None, pre_s=args.pre, post_s=args.post,
                  max_s=args.max_clip, min_free_gb=args.min_free_gb, log=say)
    left = 0
    if worker:
        try:
            if not is_file:
                say("Finishing the clip being indexed (Ctrl+C again to skip)...")
            left = worker.finish(drain=is_file)
        except KeyboardInterrupt:
            left = worker.jobs.qsize()
    say(f"Saved {saved} clip{'s' if saved != 1 else ''} this session.")
    if left or (args.no_index and saved):
        say(f"Some clips are not indexed yet. Index them with: python recall.py index \"{folder}\"")


# ---------------------------------------------------------------- index, ask, log

def cmd_index(args, cfg, store):
    from homerecall.indexer import index_folder

    if not os.path.isdir(args.folder):
        sys.exit(f"Not a folder: {args.folder}")
    models = Models(cfg, log=say)
    done, moments = index_folder(
        args.folder, store, cfg, models.detector(),
        vlm=None if args.no_captions else models.vlm(),
        embedder=models.embedder() if cfg.embedder else None,
        limit=args.limit, reindex=args.reindex, log=say)
    say(f"Indexed {done} recordings, {moments} moments. Index: {store.root}")


def cmd_ask(args, cfg, store):
    from homerecall import report
    from homerecall.search import Searcher, parse_day

    start = parse_day(args.since) if args.since else None
    end = parse_day(args.until) if args.until else None
    if args.date:
        start = parse_day(args.date)
        end = start + dt.timedelta(days=1)
    models = Models(cfg, log=say)
    has_embeddings = store.stats()["with_embeddings"] > 0
    searcher = Searcher(store, cfg, detector=None if args.no_probe else models.detector,
                        vlm=None if args.no_verify else models.vlm,
                        embedder=models.embedder if (cfg.embedder and has_embeddings) else None, log=say)
    if has_embeddings and not cfg.embedder:
        say("Note: this index has embeddings, but the current profile has no embedder, so they are not used.")

    def answer(question):
        result = searcher.search(question, start, end, verify=args.verify, threshold=args.threshold,
                                 probe=not args.no_probe)
        if result["matches"]:
            report.save(result, store, report.result_folder(paths.RESULTS, question), clips=not args.no_clips)
        say()
        say(report.format_result(result, args.top))
        if args.open and result["matches"]:
            best = result["matches"][0]
            os.startfile(best.get("cut") or best["clip_path"])  # noqa: S606 (Windows only)

    if args.question:
        return answer(args.question)
    say("Ask about your recordings. Empty line to quit.")
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
            say(f"\n{dt.date.fromisoformat(day):%A %d %B %Y}")
        tags = [k for k, v in json.loads(w["objects"]).get("tags", {}).items() if v >= 0.25 and k != "person"]
        say(f"  {w['start'][11:]} to {w['end'][11:]}  {w['caption'] or '(no caption)'}"
            + (f"  [{', '.join(tags[:6])}]" if tags else ""))


def main():
    args = build_parser().parse_args()
    from homerecall.store import Store

    if args.command == "camera":
        return cmd_camera(args)
    if args.command in ("log", "stats"):
        store = Store(args.index_dir)
        if args.command == "stats":
            say(json.dumps(store.stats(), indent=2))
        else:
            cmd_log(args, store)
        return
    cfg = pick(args.profile, vlm=args.vlm, detector=args.detector)
    if not cfg.allow_download and not args.allow_download:
        paths.forbid_downloads()
    say(f"Profile: {cfg.name}  (models from {paths.CACHE})")
    if args.command == "watch":
        return cmd_watch(args, cfg)
    store = Store(args.index_dir)
    (cmd_index if args.command == "index" else cmd_ask)(args, cfg, store)


if __name__ == "__main__":
    main()
