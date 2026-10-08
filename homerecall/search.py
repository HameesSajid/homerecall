"""Answering a question from the index: retrieve, then verify.

RETRIEVE  Rank every moment in the time range by cheap evidence that it might match:
          - keywords: the question's words in the caption and object tags (BM25)
          - detector: the question's nouns found in the key frames. Words outside the
            indexing vocabulary are looked for now, and the result is cached
          - embedding: similarity between the question and the key frames, when the
            index has embeddings
          The three rankings are combined by reciprocal rank fusion.

VERIFY    The best candidates are shown to the language model again with the question
          itself, and it must say yes. Its probability of "yes" is the confidence.

Retrieval is allowed to be wrong, because verification looks at the frames again.
Verification is never skipped silently: a result that was not verified says so.
"""

import datetime as dt
import json
import re

import cv2
import numpy as np

from . import text
from .vocab import HOME_VOCAB

PROBE_MIN = 0.15   # detector confidence below this is treated as not found
MERGE_GAP_S = 2.0  # matching moments of one clip this close together are one result


def similarity(query_vec, frame_embs):
    """Best cosine similarity between the question and any key frame of a moment."""
    return float(np.max(frame_embs @ query_vec))


def as_question(sentence):
    s = sentence.strip().rstrip(".!?").strip()
    return (s[0].upper() + s[1:] + "?") if s else s


def _doc(w):
    """The text keyword search sees for a moment: its caption, then each object once."""
    objects = w["objects"]
    names = dict.fromkeys(objects.get("confirmed", []) + objects.get("proposed", [])
                          + [k for k, v in objects.get("tags", {}).items() if v >= 0.25])
    return " ".join([w["caption"] or "", *names])


def merge(matches, gap=MERGE_GAP_S):
    """Join matching moments of the same clip that touch or overlap. The joined result
    keeps the frames, caption and score of its most confident part."""
    out = []
    for m in sorted(matches, key=lambda m: (m["clip_path"], m["t0"])):
        last = out[-1] if out else None
        if last and last["clip_path"] == m["clip_path"] and m["t0"] <= last["t1"] + gap:
            best = m if m["confidence"] > last["confidence"] else last
            joined = dict(best, t0=last["t0"], t1=max(last["t1"], m["t1"]), start=last["start"],
                          end=max(last["end"], m["end"]))
            out[-1] = joined
        else:
            out.append(dict(m))
    return sorted(out, key=lambda m: -m["confidence"])


class Searcher:
    """`detector`, `vlm` and `embedder` are zero-argument callables that return the model,
    so a model is only loaded if the search actually needs it."""

    def __init__(self, store, cfg, detector=None, vlm=None, embedder=None, log=print):
        self.store, self.cfg, self.log = store, cfg, log
        self._detector, self._vlm, self._embedder = detector, vlm, embedder

    def _frames(self, window_id):
        return [f for f in (cv2.imread(self.store.abs(r["path"])) for r in self.store.frames(window_id)) if f is not None]

    def _detector_scores(self, windows, terms, probe):
        """{window id: summed confidence for the question's nouns}."""
        if not terms:
            return {}
        known = [t for t in terms if t in HOME_VOCAB]
        new = [t for t in terms if t not in HOME_VOCAB]
        cached = self.store.probes(new)
        missing = [w for w in windows if any(t not in cached.get(w["id"], {}) for t in new)]
        if new and missing and probe and self._detector:
            detector = self._detector()
            self.log(f"Looking for {', '.join(new)} in {len(missing)} moments with the detector...")
            rows = []
            for w in missing:
                found = detector.check(self._frames(w["id"]), new, conf=0.05, competitors=HOME_VOCAB)
                rows += [(w["id"], t, round(found.get(t, 0.0), 3)) for t in new]
            self.store.set_probes(rows)
            cached = self.store.probes(new)
        scores = {}
        for w in windows:
            tags = w["objects"].get("tags", {})
            confs = [tags.get(t, 0.0) for t in known] + [cached.get(w["id"], {}).get(t, 0.0) for t in new]
            scores[w["id"]] = sum(c for c in confs if c >= PROBE_MIN)
        return scores

    def _embedding_scores(self, windows, question):
        embs = self.store.embeddings([w["id"] for w in windows])
        if not embs or not self._embedder:
            return {}
        q = self._embedder().text(question)
        sims = {wid: similarity(q, e) for wid, e in embs.items()}
        # Cosine similarities sit in a narrow band; only the ordering is used.
        low = min(sims.values())
        return {wid: s - low + 1e-6 for wid, s in sims.items()}

    def search(self, question, start=None, end=None, verify=None, threshold=0.5, probe=True, now=None):
        """Returns a dict: the parsed range, counts, `matches` (verified, merged, best
        first) and `candidates` (everything that was checked or ranked, best first)."""
        q_start, q_end, rest = text.parse_when(question, now)
        start, end = start or q_start, end or q_end
        windows = self.store.windows(start, end)
        for w in windows:
            w["objects"] = json.loads(w["objects"]) if isinstance(w["objects"], str) else w["objects"]
        out = {"question": question, "asked": as_question(rest), "start": start, "end": end,
               "searched": len(windows), "clips": len({w["clip_path"] for w in windows}), "verified": 0,
               "threshold": threshold, "matches": [], "candidates": [], "signals": []}
        if not windows:
            return out

        ids = [w["id"] for w in windows]
        keyword = dict(zip(ids, text.BM25([_doc(w) for w in windows]).scores(rest)))
        detector = self._detector_scores(windows, text.query_objects(rest), probe)
        embedding = self._embedding_scores(windows, rest)
        rankings = {"keywords": keyword, "detector": detector, "embedding": embedding}
        out["signals"] = [name for name, r in rankings.items() if any(v > 0 for v in r.values())]
        fused = text.fuse(list(rankings.values()))
        # Moments with no evidence at all still queue up behind the rest, newest first, so
        # that a generous --verify can reach them.
        newest_first = sorted(windows, key=lambda w: w["start"], reverse=True)
        order = sorted(newest_first, key=lambda w: -fused.get(w["id"], 0.0))  # stable

        n = len(order) if verify == "all" else min(len(order), self.cfg.verify_top if verify is None else int(verify))
        for w in order:
            w["retrieval"] = round(fused.get(w["id"], 0.0), 5)
            w["evidence"] = {name: round(r.get(w["id"], 0.0), 3) for name, r in rankings.items() if r.get(w["id"], 0) > 0}
            w["confidence"] = None
        if n and self._vlm:
            vlm = self._vlm()
            prompt = text.VERIFY_PROMPT.format(question=out["asked"])
            self.log(f"Checking {n} of {len(order)} moments with the language model...")
            for w in order[:n]:
                frames = self._frames(w["id"])
                w["confidence"] = round(vlm.yes_probability(frames, prompt), 3) if frames else 0.0
            out["verified"] = n
            checked = sorted(order[:n], key=lambda w: -w["confidence"])
            out["matches"] = merge([w for w in checked if w["confidence"] >= threshold])
            out["candidates"] = checked
        else:
            out["candidates"] = order
        return out


def parse_day(value):
    """A --date / --since / --until value: YYYY-MM-DD, optionally with HH:MM."""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", value.strip())
    if not m:
        raise ValueError(f"expected YYYY-MM-DD or 'YYYY-MM-DD HH:MM', got {value!r}")
    y, mo, d, h, mi = (int(g) if g else 0 for g in m.groups())
    return dt.datetime(y, mo, d, h, mi)
