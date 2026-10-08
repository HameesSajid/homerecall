"""Turn per-frame person detections into moments.

A recording is cut into short windows in which someone is present. A window is the unit
that gets a caption, key frames and a place in the search index, so it has to be short
enough that one caption can cover what happened in it.

Pure Python: no model, unit-tested on synthetic input.
"""

from dataclasses import dataclass, field


@dataclass
class Sample:
    """What the detector found in one examined frame."""

    index: int               # frame number in the video
    t: float                 # seconds from the start of the clip
    boxes: list = field(default_factory=list)  # (x1, y1, x2, y2) of each person, in 0..1


@dataclass
class Window:
    t0: float
    t1: float
    samples: list            # the Samples inside it
    has_subject: bool        # False for a fallback window where nobody was detected
    n_people: int            # most people seen in one frame
    crop: tuple              # (x1, y1, x2, y2) in 0..1: where the activity is


def _groups(samples, gap_s, min_hits):
    """Runs of frames with a person, allowing gaps of up to `gap_s` (a person bending
    behind a counter is still the same visit). Runs with fewer than `min_hits` frames
    are detector noise."""
    runs, run, last = [], [], None
    for s in samples:
        if not s.boxes:
            continue
        if last is not None and s.t - last > gap_s:
            runs.append(run)
            run = []
        run.append(s)
        last = s.t
    if run:
        runs.append(run)
    return [r for r in runs if len(r) >= min_hits]


def _split(t0, t1, max_len, overlap):
    """Cut [t0, t1] into equal pieces no longer than `max_len` that overlap, so an action
    that straddles a cut is whole in at least one piece."""
    span = t1 - t0
    if span <= max_len:
        return [(t0, t1)]
    stride = max_len - overlap
    n = 1
    while overlap + n * stride < span - 1e-9:
        n += 1
    length = (span + (n - 1) * overlap) / n
    step = length - overlap
    return [(t0 + i * step, t0 + i * step + length) for i in range(n)]


def activity_crop(boxes, margin=0.35, min_side=0.4, full_above=0.7):
    """The region to look at: the union of the person boxes, grown by `margin` of its size
    so that what the person reaches for (a shelf, a door) is inside it. A region that is
    already most of the frame becomes the whole frame."""
    if not boxes:
        return (0.0, 0.0, 1.0, 1.0)
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = max(w * (1 + 2 * margin), min_side), max(h * (1 + 2 * margin), min_side)
    x1, x2 = cx - w / 2, cx + w / 2
    y1, y2 = cy - h / 2, cy + h / 2
    # slide back inside the frame before clamping, so the region keeps its size at an edge
    if x1 < 0:
        x1, x2 = 0.0, x2 - x1
    if x2 > 1:
        x1, x2 = x1 - (x2 - 1), 1.0
    if y1 < 0:
        y1, y2 = 0.0, y2 - y1
    if y2 > 1:
        y1, y2 = y1 - (y2 - 1), 1.0
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if (x2 - x1) * (y2 - y1) >= full_above:
        return (0.0, 0.0, 1.0, 1.0)
    return (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))


def make_windows(samples, duration, max_len=8.0, overlap=2.0, gap_s=2.0, min_hits=2, pad_s=1.0):
    """Windows for one clip. If nobody is detected anywhere the whole clip is still cut
    into windows (flagged `has_subject=False`): the camera recorded because it saw
    someone, so a detector miss must not make the clip unsearchable."""
    windows = []
    for run in _groups(samples, gap_s, min_hits):
        a, b = max(0.0, run[0].t - pad_s), min(duration, run[-1].t + pad_s)
        for t0, t1 in _split(a, b, max_len, overlap):
            inside = [s for s in samples if t0 <= s.t <= t1]
            boxes = [bx for s in inside for bx in s.boxes]
            windows.append(Window(t0, t1, inside, True, max((len(s.boxes) for s in inside), default=0),
                                  activity_crop(boxes)))
    if not windows and samples:
        for t0, t1 in _split(0.0, max(duration, samples[-1].t), max_len, overlap):
            inside = [s for s in samples if t0 <= s.t <= t1]
            if inside:
                windows.append(Window(t0, t1, inside, False, 0, (0.0, 0.0, 1.0, 1.0)))
    return windows


def pick_keyframes(window, k):
    """Up to `k` samples spread evenly through the window, preferring frames with a person."""
    pool = [s for s in window.samples if s.boxes] or window.samples
    if len(pool) <= k:
        return list(pool)
    if k == 1:
        return [pool[len(pool) // 2]]
    return [pool[round(i * (len(pool) - 1) / (k - 1))] for i in range(k)]
