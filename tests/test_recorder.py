"""Unit tests for recording from a stream: when clips start and stop, that their length
follows the clock, and that camera addresses never show a password. No camera, no GPU.

    python tests/test_recorder.py
"""

import datetime as dt
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from homerecall import cameras, timestamps  # noqa: E402
from homerecall.indexer import find_clips  # noqa: E402
from homerecall.recorder import INCOMING, ClipRecorder, FrameSource  # noqa: E402

FRAME = np.zeros((48, 64, 3), np.uint8)
T0 = dt.datetime(2026, 10, 9, 7, 42, 10).timestamp()
FPS = 10.0


class _Writer:
    """Counts frames and leaves an empty file where the clip would be."""

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size, self.frames = path, fps, size, 0
        open(path, "wb").close()

    def write(self, frame):
        assert (frame.shape[1], frame.shape[0]) == self.size
        self.frames += 1

    def release(self):
        pass


def _recorder(folder, **kw):
    writers, closed = [], []

    def make(path, fps, size):
        writers.append(_Writer(path, fps, size))
        return writers[-1]

    rec = ClipRecorder(folder, FPS, on_close=lambda *a: closed.append(a), open_writer=make, log=lambda *_: None,
                       min_free_gb=0, **kw)
    return rec, writers, closed


def _play(rec, person, start=0.0, check_every=5, skip=()):
    """Feed one frame per 1/FPS s. `person` has one character per second: x or ."""
    n0 = round(start * FPS)
    for i in range(n0, n0 + len(person) * int(FPS)):
        if i in skip:
            continue
        seen = person[(i - n0) // int(FPS)] == "x" if i % check_every == 0 else None
        rec.feed(T0 + i / FPS, FRAME, seen)


def test_clip_has_preroll_and_stops_after_the_person_leaves():
    with tempfile.TemporaryDirectory() as d:
        rec, writers, closed = _recorder(d, pre_s=3.0, post_s=5.0)
        _play(rec, "." * 10 + "x" * 6 + "." * 10)
        assert len(closed) == 1 and not rec.recording
        path, start, seconds = closed[0]
        # seen at 10.0 and 10.5 s -> starts 3 s before the second sighting; last seen 15.5 s -> ends 5 s later
        assert start == dt.datetime(2026, 10, 9, 7, 42, 17, 500000) and os.path.basename(path) == "20261009_074217.mp4"
        assert abs(seconds - 13.1) < 0.11 and writers[0].frames == round(seconds * FPS)
        assert os.path.isfile(path) and os.listdir(os.path.join(d, INCOMING)) == []
        assert timestamps.from_filename(path) == start.replace(microsecond=0)


def test_one_sighting_is_not_enough():
    with tempfile.TemporaryDirectory() as d:
        rec, writers, _ = _recorder(d)
        for i in range(100):
            rec.feed(T0 + i / FPS, FRAME, (i == 50) if i % 5 == 0 else None)
        assert writers == [] and not rec.recording


def test_length_follows_the_clock_when_frames_are_lost():
    with tempfile.TemporaryDirectory() as d:
        rec, writers, closed = _recorder(d, pre_s=0.0, post_s=2.0)
        lost = set(range(40, 60)) - {40, 45, 50, 55}  # 2 s with only the checked frames arriving
        _play(rec, "x" * 10 + "." * 5, skip=lost)
        seconds = closed[0][2]
        assert abs(seconds - 11.1) < 0.11            # 0.5 .. 11.6 s whatever was lost
        assert writers[0].frames == round(seconds * FPS)


def test_long_activity_rolls_into_a_new_clip():
    with tempfile.TemporaryDirectory() as d:
        rec, writers, closed = _recorder(d, pre_s=0.0, post_s=2.0, max_s=10.0)
        _play(rec, "x" * 25 + "." * 5)
        assert len(closed) == 3 and [round(c[2]) for c in closed] == [10, 10, 6]
        assert [os.path.basename(c[0]) for c in closed] == ["20261009_074210.mp4", "20261009_074220.mp4", "20261009_074230.mp4"]


def test_break_in_the_stream_closes_the_clip():
    with tempfile.TemporaryDirectory() as d:
        rec, _, closed = _recorder(d, pre_s=0.0, post_s=5.0)
        _play(rec, "x" * 4)
        assert rec.recording
        rec.idle(T0 + 6.0)
        assert rec.recording          # 2 s of silence: keep waiting
        rec.idle(T0 + 9.5)
        assert not rec.recording and abs(closed[0][2] - 3.5) < 0.11
        _play(rec, "x" * 4, start=20.0)   # and a gap noticed only when frames return
        assert rec.recording
        rec.feed(T0 + 60.0, FRAME, None)
        assert len(closed) == 2 and not rec.recording


def test_stop_closes_an_open_clip_and_names_do_not_collide():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "20261009_074210.mp4"), "wb").close()
        rec, _, closed = _recorder(d, pre_s=0.0)
        _play(rec, "xx")
        path = rec.close()
        assert os.path.basename(path) == "20261009_074210_1.mp4" and closed[0][0] == path and rec.close() is None
        assert timestamps.from_filename(path) == dt.datetime(2026, 10, 9, 7, 42, 10)


def test_full_disk_stops_recording():
    with tempfile.TemporaryDirectory() as d:
        notes = []
        rec = ClipRecorder(d, FPS, min_free_gb=10**9, open_writer=_Writer, log=notes.append)
        _play(rec, "x" * 5)
        assert not rec.recording and len(notes) == 1 and "free" in notes[0]


def test_real_writer_and_file_source_round_trip():
    with tempfile.TemporaryDirectory() as d:
        rec = ClipRecorder(os.path.join(d, "out"), FPS, pre_s=1.0, post_s=1.0, min_free_gb=0, log=lambda *_: None)
        for i in range(60):
            frame = np.full((48, 64, 3), i * 4, np.uint8)
            rec.feed(T0 + i / FPS, frame, (10 <= i < 40) if i % 5 == 0 else None)
        clips = find_clips(os.path.join(d, "out"))
        assert [os.path.basename(c) for c in clips] == ["20261009_074210.mp4"]   # .incoming is not listed
        src = FrameSource(clips[0], log=lambda *_: None, start_time=T0)
        src.start()
        assert src.connected.wait(10) and not src.live and src.size == (64, 48)
        times = []
        while not (src.ended.is_set() and src.frames.empty()):
            try:
                times.append(src.frames.get(timeout=0.2)[0])
            except Exception:  # noqa: BLE001
                continue
        assert len(times) == 41 and abs((times[-1] - times[0]) - 4.0) < 0.01 and src.dropped == 0


def test_addresses_are_masked_and_escaped():
    url = cameras.rtsp_url("192.168.1.50", "hamees", "p@ss:w/rd#1")
    assert url == "rtsp://hamees:p%40ss%3Aw%2Frd%231@192.168.1.50:554/stream1"
    assert cameras.mask(url) == "rtsp://***@192.168.1.50:554/stream1"
    assert "p%40ss" not in cameras.mask(url) and "hamees" not in cameras.mask(url)
    assert cameras.mask(0) == "0" and cameras.mask("C:/clips/a.mp4") == "C:/clips/a.mp4"


def test_camera_file():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cameras.json")
        try:
            cameras.resolve(None, path)
            raise AssertionError("expected LookupError")
        except LookupError as e:
            assert "camera add" in str(e)
        cameras.save({"kitchen": "rtsp://u:p@10.0.0.2:554/stream1"}, path)
        assert cameras.resolve(None, path) == ("kitchen", "rtsp://u:p@10.0.0.2:554/stream1")
        cameras.save({"kitchen": "rtsp://a", "door": "rtsp://b"}, path)
        assert cameras.resolve("door", path) == ("door", "rtsp://b")
        for name in (None, "garage"):
            try:
                cameras.resolve(name, path)
                raise AssertionError("expected LookupError")
            except LookupError as e:
                assert "p@" not in str(e)


if __name__ == "__main__":
    assert cv2.__version__
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
