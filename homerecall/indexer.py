"""Indexing: turn a folder of recordings into searchable moments.

For each new clip:

1. SCAN     Read the clip once, looking for people a few times per second.
2. CUT      Group those frames into short moments (`segments.make_windows`).
3. KEEP     Read the clip again and save a few key frames per moment, cropped from the
            full-resolution frame to the region around the person.
4. DESCRIBE The language model captions each moment from its key frames, in time order.
5. TAG      The detector looks for household objects and for every object the caption
            named, and records its confidence for each.
6. EMBED    (profiles with an embedder) Each key frame gets an image-text embedding.

Nothing here answers a question. It only makes sure that when one is asked, the right
moment can be found without watching the footage again.
"""

import os
import time

import cv2

from . import text
from .vocab import HOME_VOCAB, SUBJECTS
from .segments import Sample, make_windows, pick_keyframes
from .timestamps import clip_start

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".ts"}


def find_clips(folder):
    """Every video under `folder`, sorted by path."""
    out = []
    for root, _, names in os.walk(folder):
        out += [os.path.join(root, n) for n in names if os.path.splitext(n)[1].lower() in VIDEO_EXT]
    return sorted(out)


def scan(path, detector, sample_fps, person_conf=0.3):
    """Pass 1. Returns (info, samples): clip properties and one Sample per examined frame."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise OSError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not 1.0 <= fps <= 240.0:
        fps = 25.0
    step = max(1, round(fps / sample_fps))
    samples, i, size = [], 0, (0, 0)
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                size = (frame.shape[1], frame.shape[0])
                people = [b for label, _, b in detector.boxes(frame, SUBJECTS, conf=person_conf) if label in SUBJECTS]
                samples.append(Sample(i, i / fps, people))
        i += 1
    cap.release()
    return {"fps": fps, "frames": i, "duration": i / fps, "width": size[0], "height": size[1]}, samples


def _fit(image, side):
    h, w = image.shape[:2]
    s = side / max(h, w)
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else image


def save_keyframes(path, windows, k, side, folder, context_width=960):
    """Pass 2. Writes each window's key frames as JPEGs and returns, per window,
    [(t, crop path, context path)]. Frames are written as they are decoded, so memory
    does not grow with the length of the clip."""
    wanted = {}  # frame index -> [(window number, position in window, crop)]
    plan = []
    for wi, w in enumerate(windows):
        keys = pick_keyframes(w, k)
        plan.append([None] * len(keys))
        for ki, s in enumerate(keys):
            wanted.setdefault(s.index, []).append((wi, ki, s.t, w.crop, ki == len(keys) // 2))
    cap = cv2.VideoCapture(path)
    last = max(wanted, default=-1)
    i = 0
    while i <= last and cap.grab():
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                h, w_px = frame.shape[:2]
                for wi, ki, t, crop, is_middle in wanted[i]:
                    x1, y1, x2, y2 = (round(crop[0] * w_px), round(crop[1] * h), round(crop[2] * w_px), round(crop[3] * h))
                    crop_path = os.path.join(folder, f"w{wi:04d}_{ki}.jpg")
                    cv2.imwrite(crop_path, _fit(frame[y1:y2, x1:x2], side), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    context_path = None
                    if is_middle:
                        context_path = os.path.join(folder, f"w{wi:04d}_context.jpg")
                        view = _fit(frame, context_width).copy()
                        sx, sy = view.shape[1], view.shape[0]
                        cv2.rectangle(view, (round(crop[0] * sx), round(crop[1] * sy)),
                                      (round(crop[2] * sx) - 1, round(crop[3] * sy) - 1), (0, 220, 255), 2)
                        cv2.imwrite(context_path, view, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    plan[wi][ki] = (t, crop_path, context_path)
        i += 1
    cap.release()
    out = []
    for frames in plan:
        frames = [f for f in frames if f]
        context = next((c for _, _, c in frames if c), frames[len(frames) // 2][1] if frames else None)
        out.append([(t, p, context) for t, p, _ in frames])
    return out


def describe(frames, has_subject, vlm, cfg):
    """(caption, objects the model named) for one moment."""
    if cfg.two_step:
        prompt = text.CAPTION_PROMPT if has_subject else text.CAPTION_PROMPT_NOBODY
        caption = vlm.generate(frames, prompt, cfg.caption_tokens)
        proposed = text.parse_list(vlm.generate(frames[len(frames) // 2: len(frames) // 2 + 1], text.LIST_PROMPT, 40))
    else:
        proposed, caption = text.parse_proposal(vlm.generate(frames, text.PROPOSE_PROMPT, cfg.caption_tokens + 50))
    return text.complete_sentences(caption), proposed


def tag(frames, proposed, detector, threshold):
    """Detector confidence for the home vocabulary and for the model's own objects.

    Unlike a live description, nothing is removed from the caption here: at search time a
    missed moment cannot be recovered, a wrong candidate can (the question is checked
    against the frames again). The confidences are kept so search can weigh them."""
    found = detector.check(frames, HOME_VOCAB + [p for p in proposed if p not in HOME_VOCAB], conf=0.1)
    return {
        "tags": {k: round(v, 3) for k, v in sorted(found.items(), key=lambda kv: -kv[1])},
        "proposed": proposed,
        "confirmed": [p for p in proposed if found.get(p, 0.0) >= threshold],
    }


def index_clip(path, store, cfg, detector, vlm=None, embedder=None, log=print):
    """Index one clip. Returns the number of moments stored."""
    t_start = time.time()
    size, mtime = os.path.getsize(path), os.path.getmtime(path)
    store.remove_clip(path)  # a changed or half-indexed clip is redone from scratch
    info, samples = scan(path, detector, cfg.sample_fps)
    start, source = clip_start(path, info["duration"])
    windows = make_windows(samples, info["duration"], max_len=cfg.window_s, overlap=cfg.overlap_s)
    clip_id = store.add_clip(path, size, mtime, start, source, info["duration"], info["fps"], info["width"],
                             info["height"], cfg.name)
    saved = save_keyframes(path, windows, cfg.frames, cfg.keyframe_side, store.frame_dir(clip_id))
    for w, frames_meta in zip(windows, saved):
        if not frames_meta:
            continue
        frames = [cv2.imread(p) for _, p, _ in frames_meta]
        caption, objects = "", {"tags": {}, "proposed": [], "confirmed": []}
        if vlm is not None:
            caption, proposed = describe(frames, w.has_subject, vlm, cfg)
            objects = tag(frames, proposed, detector, cfg.tag_threshold)
        else:
            objects = tag(frames, [], detector, cfg.tag_threshold)
        embs = embedder.images(frames) if embedder is not None else [None] * len(frames)
        wid = store.add_window(clip_id, start, w.t0, w.t1, w.has_subject, w.n_people, w.crop, caption, objects)
        for (t, p, context), e in zip(frames_meta, embs):
            store.add_frame(wid, t, p, context, e)
    store.commit()
    people = "" if any(w.has_subject for w in windows) else ", nobody detected"
    log(f"  {start:%Y-%m-%d %H:%M:%S} ({source})  {info['duration']:.0f} s  {len(windows)} moments{people}  "
        f"[{time.time() - t_start:.0f} s]")
    return len(windows)


def index_folder(folder, store, cfg, detector, vlm=None, embedder=None, limit=None, reindex=False, log=print):
    """Index every clip in `folder` that is new or has changed. Returns (clips done, moments)."""
    clips = find_clips(folder)
    todo = [c for c in clips if reindex or not store.is_indexed(os.path.abspath(c), os.path.getsize(c), os.path.getmtime(c))]
    log(f"{len(clips)} recordings in {folder}, {len(todo)} to index")
    if limit:
        todo = todo[:limit]
    moments = 0
    for n, clip in enumerate(todo, 1):
        log(f"[{n}/{len(todo)}] {os.path.basename(clip)}")
        try:
            moments += index_clip(os.path.abspath(clip), store, cfg, detector, vlm, embedder, log)
        except OSError as e:
            log(f"  skipped: {e}")
    return len(todo), moments
