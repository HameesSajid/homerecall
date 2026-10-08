"""Unit tests for everything that does not need a model: timestamps, cutting moments,
text handling, time ranges, the index store, and how search combines and merges results.

    python tests/test_core.py
"""

import datetime as dt
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from homerecall import text, timestamps  # noqa: E402
from homerecall.search import Searcher, as_question, merge, parse_day  # noqa: E402
from homerecall.segments import Sample, activity_crop, make_windows, pick_keyframes  # noqa: E402
from homerecall.store import Store  # noqa: E402

D = dt.datetime
BOX = (0.4, 0.3, 0.6, 0.9)


# ---------------------------------------------------------------- timestamps

def test_filename_formats():
    for name in ("20261006_074210.mp4", "2026-10-06 07-42-10.mp4", "Kitchen_2026.10.06_07.42.10_tp.mp4",
                 "20261006074210.mov", "cam1-20261006T074210-event.mp4"):
        assert timestamps.from_filename(name) == D(2026, 10, 6, 7, 42, 10), name


def test_filename_rejects_non_dates():
    assert timestamps.from_filename("IMG_4821.mp4") is None
    assert timestamps.from_filename("20261340_256199.mp4") is None  # month 13, hour 25
    assert timestamps.from_filename("recording 12345678901234567890.mp4") is None


def _mp4(path, seconds_since_1904, version=0):
    body = bytes([version, 0, 0, 0]) + (struct.pack(">Q", seconds_since_1904) if version else
                                        struct.pack(">I", seconds_since_1904)) + b"\0" * 80
    mvhd = struct.pack(">I4s", 8 + len(body), b"mvhd") + body
    with open(path, "wb") as f:
        f.write(struct.pack(">I4s", 16, b"ftyp") + b"isom\0\0\0\0")
        f.write(struct.pack(">I4s", 8 + len(mvhd), b"moov") + mvhd)


def test_metadata_time_and_unset_clock():
    with tempfile.TemporaryDirectory() as d:
        when = D(2026, 10, 6, 7, 42, 10, tzinfo=dt.timezone.utc)
        secs = int((when - D(1904, 1, 1, tzinfo=dt.timezone.utc)).total_seconds())
        for version in (0, 1):
            p = os.path.join(d, f"clip{version}.mp4")
            _mp4(p, secs, version)
            assert timestamps.from_metadata(p) == when.astimezone().replace(tzinfo=None)
            assert timestamps.clip_start(p)[1] == "metadata"
        p = os.path.join(d, "unset.mp4")
        _mp4(p, 0)
        assert timestamps.from_metadata(p) is None
        start, source = timestamps.clip_start(p, duration=30.0)
        assert source == "file-modified"
        assert abs((D.now() - start).total_seconds() - 30) < 5  # saved now, so it began 30 s ago


def test_filename_wins_over_metadata():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "20250101_120000.mp4")
        _mp4(p, 3_900_000_000)
        assert timestamps.clip_start(p) == (D(2025, 1, 1, 12, 0, 0), "filename")


# ---------------------------------------------------------------- moments

def _samples(present, fps=2.0):
    """`present` is a string, one character per examined frame: x = person, . = nobody."""
    return [Sample(i * 15, i / fps, [BOX] if c == "x" else []) for i, c in enumerate(present)]


def test_one_visit_becomes_one_window():
    w = make_windows(_samples("....xxxxxx...."), duration=7.0)
    assert len(w) == 1 and w[0].has_subject and w[0].n_people == 1
    assert w[0].t0 == 1.0 and w[0].t1 == 5.5  # 2.0 .. 4.5 s, padded by 1 s


def test_short_gap_is_bridged_long_gap_splits():
    assert len(make_windows(_samples("xxxx..xxxx"), 5.0)) == 1          # 1.5 s gap
    assert len(make_windows(_samples("xxxx" + "." * 10 + "xxxx"), 9.0)) == 2


def test_single_frame_is_noise():
    w = make_windows(_samples("......x......"), 6.5)
    assert len(w) > 0 and not any(x.has_subject for x in w)  # falls back to the whole clip


def test_long_activity_is_split_with_overlap():
    w = make_windows(_samples("x" * 60), duration=30.0, max_len=8.0, overlap=2.0, pad_s=0.0)
    assert all(x.t1 - x.t0 <= 8.0 + 1e-6 for x in w)
    assert w[0].t0 == 0.0 and abs(w[-1].t1 - 29.5) < 1e-6
    for a, b in zip(w, w[1:]):
        assert abs((a.t1 - b.t0) - 2.0) < 1e-6
    assert len(w) == 5


def test_nobody_detected_still_indexed():
    w = make_windows(_samples("." * 20), duration=10.0)
    assert w and not w[0].has_subject and w[0].crop == (0.0, 0.0, 1.0, 1.0)
    assert sum(len(x.samples) for x in w) >= 20
    assert make_windows([], 0.0) == []


def test_crop_grows_and_stays_inside():
    x1, y1, x2, y2 = activity_crop([(0.45, 0.4, 0.55, 0.8)])
    assert x1 < 0.45 and x2 > 0.55 and y1 < 0.4 and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
    assert x2 - x1 >= 0.4 - 1e-6                      # never a sliver
    edge = activity_crop([(0.0, 0.5, 0.1, 1.0)])
    assert edge[0] == 0.0 and edge[3] == 1.0 and edge[2] - edge[0] >= 0.4 - 1e-6
    assert activity_crop([(0.05, 0.05, 0.95, 0.95)]) == (0.0, 0.0, 1.0, 1.0)
    assert activity_crop([]) == (0.0, 0.0, 1.0, 1.0)


def test_keyframes_spread_and_prefer_people():
    w = make_windows(_samples("xxxxxxxxxxxx"), 6.0, pad_s=0.0)[0]
    keys = pick_keyframes(w, 4)
    assert [k.t for k in keys] == [0.0, 2.0, 3.5, 5.5]
    w = make_windows(_samples("x.x.x.x."), 4.0)[0]
    assert all(k.boxes for k in pick_keyframes(w, 3))
    assert len(pick_keyframes(w, 10)) == 4


# ---------------------------------------------------------------- text

def test_tokens_stem_and_synonyms():
    assert text.tokens("Did anyone take eggs out of the fridge?") == ["take", "egg", "refrigerator"]
    assert text.tokens("A person is taking an egg carton from the refrigerator.") == ["take", "egg", "carton", "refrigerator"]
    assert text.tokens("She grabbed the keys and left") == ["take", "keys", "leave"]
    assert text.stem("running") == "run" and text.stem("opened") == "open" and text.stem("boxes") == "box"


def test_query_objects():
    assert text.query_objects("did anyone take eggs out of the fridge") == ["egg", "refrigerator"]
    assert text.query_objects("who left a parcel at the door") == ["package", "door"]


def test_parse_replies():
    objs, desc = text.parse_proposal("OBJECTS: a fridge, Eggs, two bowls, the man\nDESCRIPTION: A man opens the fridge. He")
    assert objs == ["fridge", "egg", "bowl", "person"] and desc == "A man opens the fridge. He"
    assert text.complete_sentences(desc) == "A man opens the fridge."
    assert text.parse_list("Objects: cup, plates and a wooden table") == ["cup", "plate", "wooden table"]
    assert text.parse_proposal("Someone walks in.") == ([], "Someone walks in.")


def test_bm25_ranks_the_matching_caption_first():
    docs = ["A person walks through the kitchen.",
            "A person opens the refrigerator and takes out a carton of eggs.",
            "A person sits on the sofa and watches tv.",
            "A person opens the refrigerator and takes a bottle."]
    s = text.BM25(docs).scores("did anyone take eggs out of the fridge")
    assert s.index(max(s)) == 1 and s[3] > s[0] and s[2] == 0.0


def test_fusion():
    f = text.fuse([{1: 5.0, 2: 1.0, 3: 0.0}, {2: 0.9, 3: 0.8}])
    assert f[2] > f[1] > f[3] and set(f) == {1, 2, 3}
    assert text.fuse([{1: 0.0}, {}]) == {}


NOW = D(2026, 10, 8, 15, 0)  # a Thursday


def test_when_days():
    assert text.parse_when("did anyone take eggs yesterday", NOW) == (D(2026, 10, 7), D(2026, 10, 8), "did anyone take eggs")
    assert text.parse_when("who came in today?", NOW)[:2] == (D(2026, 10, 8), D(2026, 10, 9))
    assert text.parse_when("who came on monday", NOW)[:2] == (D(2026, 10, 5), D(2026, 10, 6))
    assert text.parse_when("was the door open on 2026-10-03", NOW)[:2] == (D(2026, 10, 3), D(2026, 10, 4))
    assert text.parse_when("anyone here on october 5th?", NOW) == (D(2026, 10, 5), D(2026, 10, 6), "anyone here?")
    assert text.parse_when("anyone here on 5 Oct", NOW)[:2] == (D(2026, 10, 5), D(2026, 10, 6))
    assert text.parse_when("anyone here on december 30", NOW)[0] == D(2025, 12, 30)


def test_when_parts_of_day():
    assert text.parse_when("did the cat eat this morning", NOW) == (D(2026, 10, 8, 5), D(2026, 10, 8, 12), "did the cat eat")
    assert text.parse_when("who was in the kitchen last night", NOW) == (D(2026, 10, 7, 21), D(2026, 10, 8, 5), "who was in the kitchen")
    assert text.parse_when("anyone at the door yesterday evening", NOW)[:2] == (D(2026, 10, 7, 17), D(2026, 10, 7, 22))
    assert text.parse_when("anything in the last 3 days", NOW)[:2] == (D(2026, 10, 5), NOW)


def test_when_absent_and_not_fooled():
    assert text.parse_when("did anyone take eggs out of the fridge", NOW) == (None, None, "did anyone take eggs out of the fridge")
    # "may" and "night" without a day are ordinary words
    assert text.parse_when("who may have opened the night stand", NOW)[:2] == (None, None)


# ---------------------------------------------------------------- store and search

def _store(d):
    s = Store(d)
    start = D(2026, 10, 6, 7, 42, 10)
    cid = s.add_clip("C:/cam/20261006_074210.mp4", 10, 1.0, start, "filename", 30.0, 15.0, 1920, 1080, "laptop")
    rows = [(0, 8, "A person walks into the kitchen.", {}),
            (6, 14, "A person opens the refrigerator and takes out a carton of eggs.", {"refrigerator": 0.8}),
            (12, 20, "A person closes the refrigerator holding eggs.", {"refrigerator": 0.6}),
            (24, 30, "A person sits at the table with a laptop.", {"laptop": 0.7, "table": 0.5})]
    ids = []
    for t0, t1, caption, tags in rows:
        wid = s.add_window(cid, start, t0, t1, True, 1, (0, 0, 1, 1), caption, {"tags": tags, "proposed": [], "confirmed": []})
        s.add_frame(wid, t0 + 1.0, os.path.join(s.frame_dir(cid), f"{wid}.jpg"), os.path.join(s.frame_dir(cid), f"{wid}.jpg"),
                    np.eye(4, dtype=np.float32)[len(ids)])
        ids.append(wid)
    s.commit()
    return s, ids


class _FakeVLM:
    """Says yes to a question about eggs when the 'frame' is one that shows eggs."""

    def __init__(self, store):
        self.store, self.calls = store, 0

    def yes_probability(self, frames, prompt):
        self.calls += 1
        return 0.9 if frames[0] == "eggs" and "egg" in prompt else 0.1


def _searcher(store, vlm):
    s = Searcher(store, None, vlm=lambda: vlm, log=lambda *_: None)
    captions = {w["id"]: w["caption"] for w in store.windows()}
    s._frames = lambda wid: ["eggs" if "egg" in captions[wid] else "other"]
    s.cfg = type("Cfg", (), {"verify_top": 3})()
    return s


def test_store_round_trip_and_time_filter():
    with tempfile.TemporaryDirectory() as d:
        s, ids = _store(d)
        assert s.is_indexed("C:/cam/20261006_074210.mp4", 10, 1.0) and not s.is_indexed("C:/cam/20261006_074210.mp4", 11, 1.0)
        w = s.windows()
        assert [x["start"] for x in w][:2] == ["2026-10-06 07:42:10", "2026-10-06 07:42:16"]
        assert len(s.windows(D(2026, 10, 6, 7, 42, 31))) == 1 and s.windows(D(2026, 10, 7)) == []
        assert s.frames(ids[0])[0]["path"].startswith("frames/")
        assert s.embeddings(ids)[ids[2]].shape == (1, 4)
        s.set_probes([(ids[0], "egg", 0.0), (ids[1], "egg", 0.4)])
        assert s.probes(["egg"])[ids[1]] == {"egg": 0.4}
        st = s.stats()
        assert st["clips"] == 1 and st["moments"] == 4 and st["time_sources"] == {"filename": 1}
        s.remove_clip("C:/cam/20261006_074210.mp4")
        assert s.stats()["moments"] == 0 and s.stats()["key_frames"] == 0
        s.db.close()


def test_search_finds_verifies_and_merges():
    with tempfile.TemporaryDirectory() as d:
        s, ids = _store(d)
        vlm = _FakeVLM(s)
        r = _searcher(s, vlm).search("did anyone take eggs out of the fridge on october 6th", probe=False, now=NOW)
        assert r["asked"] == "Did anyone take eggs out of the fridge?"
        assert r["searched"] == 4 and r["verified"] == 3 and vlm.calls == 3
        assert set(r["signals"]) == {"keywords", "detector"}
        assert len(r["matches"]) == 1                     # the two egg moments overlap: one result
        m = r["matches"][0]
        assert (m["start"], m["end"], m["t0"], m["t1"]) == ("2026-10-06 07:42:16", "2026-10-06 07:42:30", 6, 20)
        assert m["confidence"] == 0.9 and m["clip_path"].endswith("20261006_074210.mp4")
        s.db.close()


def test_search_says_no_and_respects_time_range():
    with tempfile.TemporaryDirectory() as d:
        s, _ = _store(d)
        r = _searcher(s, _FakeVLM(s)).search("did a dog come in", probe=False, now=NOW)
        assert r["matches"] == [] and r["verified"] == 3 and r["candidates"][0]["confidence"] == 0.1
        r = _searcher(s, _FakeVLM(s)).search("did anyone take eggs today", probe=False, now=NOW)
        assert r["searched"] == 0 and r["matches"] == []
        s.db.close()


def test_search_without_verification_is_labelled():
    with tempfile.TemporaryDirectory() as d:
        s, ids = _store(d)
        searcher = _searcher(s, None)
        searcher._vlm = None
        r = searcher.search("eggs from the fridge", probe=False, now=NOW)
        assert r["verified"] == 0 and r["matches"] == [] and {c["id"] for c in r["candidates"][:2]} == {ids[1], ids[2]}
        assert r["candidates"][0]["confidence"] is None
        from homerecall.report import format_result

        assert "Not verified" in format_result(r)
        s.db.close()


def test_merge_keeps_separate_visits_apart():
    base = {"clip_path": "a.mp4", "start": "2026-10-06 07:00:00", "end": "2026-10-06 07:00:08"}
    a = dict(base, t0=0, t1=8, confidence=0.7)
    b = dict(base, t0=20, t1=28, confidence=0.8, start="2026-10-06 07:00:20", end="2026-10-06 07:00:28")
    c = dict(base, t0=0, t1=8, confidence=0.6, clip_path="b.mp4")
    assert [m["confidence"] for m in merge([a, b, c])] == [0.8, 0.7, 0.6]


def test_small_helpers():
    assert as_question("did anyone take eggs") == "Did anyone take eggs?"
    assert parse_day("2026-10-06") == D(2026, 10, 6) and parse_day("2026-10-06 07:30") == D(2026, 10, 6, 7, 30)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
