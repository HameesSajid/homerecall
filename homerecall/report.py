"""Presenting an answer: the text summary, a contact sheet of key frames, and the piece
of the recording that matched, cut out as its own file."""

import datetime as dt
import json
import os
import re

import cv2
import numpy as np


def _when(iso_text):
    return dt.datetime.fromisoformat(iso_text)


def _offset(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def describe_time(m):
    a, b = _when(m["start"]), _when(m["end"])
    approx = "" if m["start_source"] != "file-modified" else "  (time is approximate: taken from the file date)"
    return f"{a:%a %d %b %Y, %H:%M:%S} to {b:%H:%M:%S}{approx}"


def contact_sheet(store, window, path, height=300):
    """Key frames of a moment side by side, each stamped with its date and time."""
    clip_start = _when(window["clip_start"])
    tiles = []
    for row in store.frames(window["id"]):
        img = cv2.imread(store.abs(row["path"]))
        if img is None:
            continue
        s = height / img.shape[0]
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
        stamp = f"{clip_start + dt.timedelta(seconds=row['t']):%Y-%m-%d %H:%M:%S}"
        cv2.rectangle(img, (0, height - 22), (img.shape[1], height), (0, 0, 0), -1)
        cv2.putText(img, stamp, (5, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return None
    gap = np.full((height, 4, 3), 255, np.uint8)
    sheet = tiles[0]
    for t in tiles[1:]:
        sheet = np.hstack([sheet, gap, t])
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def cut_clip(src, t0, t1, path, pad=2.0, max_width=1280):
    """Write [t0 - pad, t1 + pad] of `src` as its own MP4 (video only; OpenCV has no audio)."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first, last = int(max(0.0, t0 - pad) * fps), int((t1 + pad) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer, i = None, first
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, round(frame.shape[0] * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        i += 1
    cap.release()
    if writer is None:
        return None
    writer.release()
    return path


def save(result, store, folder, clips=True):
    """Write a contact sheet (and the cut clip) for every match, plus result.json.
    Adds `sheet` and `cut` paths to each match."""
    os.makedirs(folder, exist_ok=True)
    for n, m in enumerate(result["matches"], 1):
        m["sheet"] = contact_sheet(store, m, os.path.join(folder, f"{n}.jpg"))
        m["cut"] = cut_clip(m["clip_path"], m["t0"], m["t1"], os.path.join(folder, f"{n}.mp4")) if clips else None
    keep = ("start", "end", "clip_path", "t0", "t1", "confidence", "retrieval", "evidence", "caption",
            "start_source", "sheet", "cut")
    slim = lambda m: {k: m.get(k) for k in keep}  # noqa: E731
    with open(os.path.join(folder, "result.json"), "w", encoding="utf-8") as f:
        json.dump({**{k: (str(v) if isinstance(v, dt.datetime) else v) for k, v in result.items()
                      if k not in ("matches", "candidates")},
                   "matches": [slim(m) for m in result["matches"]],
                   "candidates": [slim(m) for m in result["candidates"][:50]]}, f, indent=2)
    return folder


def result_folder(root, question, now=None):
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40] or "question"
    return os.path.join(root, f"{(now or dt.datetime.now()):%Y%m%d-%H%M%S}_{slug}")


def format_result(result, top=5):
    """The answer as plain text."""
    lines = [f"Q: {result['question']}"]
    if not result["searched"]:
        span = ""
        if result["start"] or result["end"]:
            span = f" between {result['start'] or 'the beginning'} and {result['end'] or 'now'}"
        return "\n".join(lines + [f"No recorded moments{span} in the index."])
    span = ""
    if result["start"] or result["end"]:
        span = f" from {result['start']:%a %d %b %H:%M}" if result["start"] else ""
        span += f" until {result['end']:%a %d %b %H:%M}" if result["end"] else ""
    lines.append(f"Searched {result['searched']} moments in {result['clips']} recordings{span}; "
                 f"ranked by {', '.join(result['signals']) or 'recency only'}; "
                 f"checked {result['verified']} with the language model.")
    lines.append("")

    def entry(n, m):
        conf = "not checked" if m["confidence"] is None else f"confidence {m['confidence']:.2f}"
        rows = [f" {n}. {describe_time(m)}   {conf}",
                f"    recording: {m['clip_path']}  at {_offset(m['t0'])} to {_offset(m['t1'])}"]
        if m.get("caption"):
            rows.append(f"    caption:   {m['caption']}")
        if m.get("cut"):
            rows.append(f"    clip:      {m['cut']}")
        if m.get("sheet"):
            rows.append(f"    frames:    {m['sheet']}")
        return rows

    if not result["verified"]:
        lines.append("Not verified (no language model was used). Best candidates by retrieval alone:")
        for n, m in enumerate(result["candidates"][:top], 1):
            lines += entry(n, m)
        return "\n".join(lines)
    if result["matches"]:
        count = len(result["matches"])
        lines.append(f"YES: {count} matching moment{'s' if count != 1 else ''}.")
        for n, m in enumerate(result["matches"][:top], 1):
            lines += entry(n, m)
        if count > top:
            lines.append(f" ... and {count - top} more (use --top).")
    else:
        lines.append(f"NO moment passed the check (needed confidence {result['threshold']:.2f}). Closest:")
        for n, m in enumerate(result["candidates"][:min(top, 3)], 1):
            lines += entry(n, m)
        if result["verified"] < result["searched"]:
            lines.append(f" {result['searched'] - result['verified']} moments were not checked; "
                         "use --verify all to check every one.")
    return "\n".join(lines)
