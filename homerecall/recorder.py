"""Recording from a live camera stream.

Instead of exporting clips from the camera's app, HomeRecall can watch the camera's
stream itself and save a clip whenever a person is in view:

    FrameSource   reads the stream on its own thread and stamps each frame with the
                  clock time it arrived, reconnecting if the stream drops
    watch()       looks for a person a few times a second
    ClipRecorder  decides when a clip starts and ends, and writes it
    IndexWorker   indexes each finished clip in the background

A clip is named after the moment it starts (`20261009_074210.mp4`), which is the file
name format `timestamps.from_filename` reads, so times in the index are exact.
"""

import collections
import datetime as dt
import os
import queue
import shutil
import threading
import time

import cv2

from .cameras import mask
from .vocab import SUBJECTS

INCOMING = ".incoming"  # clips are written here and moved out when complete


class FrameSource(threading.Thread):
    """Reads frames continuously so the stream never backs up.

    `source` is a stream address, a webcam number, or a video file (for testing: its
    frames are given the times they would have had if the file were playing live from
    `start_time`). Frames wait in `frames` as (time, image). When the reader of that
    queue falls behind a live stream, the oldest waiting frame is dropped."""

    def __init__(self, source, log=print, buffer_frames=30, start_time=None):
        super().__init__(daemon=True)
        self.source = source
        self.live = not (isinstance(source, str) and os.path.isfile(source))
        self.label = mask(source)
        self.log = log
        self.frames = queue.Queue(maxsize=buffer_frames)
        self.fps = None
        self.size = None
        self.dropped = 0
        self.start_time = start_time
        self.connected = threading.Event()
        self.ended = threading.Event()
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def _open(self):
        if self.live and isinstance(self.source, str):
            # TCP: UDP loses packets on Wi-Fi, which shows up as grey smeared frames.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
            params = []
            for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                if hasattr(cv2, name):
                    params += [getattr(cv2, name), 10000]
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
        return cv2.VideoCapture(self.source)

    def _put(self, item):
        if self.live:
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self.frames.put(item)
            return
        while not self._halt.is_set():  # a file is never dropped; wait for room
            try:
                self.frames.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def run(self):
        wait = 2.0
        while not self._halt.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                if not self.live:
                    self.log(f"Cannot open {self.label}")
                    break
                self.log(f"Cannot connect to {self.label}; trying again in {wait:.0f} s")
                self._halt.wait(wait)
                wait = min(60.0, wait * 2)
                continue
            wait = 2.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.fps = fps if 1.0 <= fps <= 60.0 else 15.0
            self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.connected.set()
            t0, i = self.start_time or time.time(), 0
            while not self._halt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                self._put((time.time() if self.live else t0 + i / self.fps, frame))
                i += 1
            cap.release()
            if not self.live:
                break
            if not self._halt.is_set():
                self.log(f"Lost the stream from {self.label}; reconnecting")
        self.ended.set()


def open_writer(path, fps, size):
    """MPEG-4 writer. It is the encoder OpenCV ships on every platform; on this project's
    test machine Windows' own H.264 encoder produced files eight times larger."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError(f"cannot write {path}")
    return writer


class ClipRecorder:
    """Turns a stream of frames and person sightings into clips.

    A clip starts once a person has been seen in `min_hits` checks in a row, and begins
    `pre_s` seconds earlier (those frames are kept in memory as JPEGs). It ends when
    nobody has been seen for `post_s` seconds. Activity longer than `max_s` continues in
    a new clip. `on_close(path, start, seconds)` is called for each finished clip.

    Frames are written by the clock, not by count: a frame that arrives late is repeated
    and one that arrives early is skipped, so a time inside the clip is the true time
    even when the network drops or delays frames."""

    def __init__(self, folder, fps, pre_s=3.0, post_s=5.0, max_s=120.0, min_hits=2, min_free_gb=10.0,
                 on_close=None, open_writer=open_writer, log=print):
        self.folder = os.path.abspath(folder)
        self.fps = float(fps)
        self.pre_s, self.post_s, self.max_s, self.min_hits = pre_s, post_s, max_s, min_hits
        self.min_free = min_free_gb * 2**30
        self.on_close, self.open_writer, self.log = on_close, open_writer, log
        os.makedirs(os.path.join(self.folder, INCOMING), exist_ok=True)
        self._pre = collections.deque()
        self._hits = 0
        self._last_person = float("-inf")
        self._last_frame = float("-inf")
        self._writer = None
        self._warned_disk = float("-inf")

    @property
    def recording(self):
        return self._writer is not None

    def feed(self, t, frame, person=None):
        """One frame. `person` is True or False when the detector looked at this frame,
        None when it did not."""
        if self._writer is not None and t - self._last_frame > self.post_s:
            self.close()  # the stream stopped for a while; what follows is a new clip
        self._last_frame = t
        if person is not None:
            self._hits = self._hits + 1 if person else 0
            if person:
                self._last_person = t
        if self._writer is None:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                self._pre.append((t, jpg))
            while self._pre and t - self._pre[0][0] > self.pre_s:
                self._pre.popleft()
            if self._hits >= self.min_hits and self._disk_ok(t):
                frames = [(ft, cv2.imdecode(j, cv2.IMREAD_COLOR)) for ft, j in self._pre]
                self._pre.clear()
                self._begin(frames)
            return
        self._write(t, frame)
        if t - self._last_person > self.post_s:
            self.close()
        elif t - self._t0 >= self.max_s:
            self.close()
            self._begin([(t, frame)])

    def idle(self, now):
        """Call when no frame has arrived for a while: a clip is not left open across a
        break in the stream."""
        if self._writer is not None and now - self._last_frame > self.post_s:
            self.close()

    def _disk_ok(self, t):
        if shutil.disk_usage(self.folder).free >= self.min_free:
            return True
        if t - self._warned_disk > 600:
            self._warned_disk = t
            self.log(f"Not recording: less than {self.min_free / 2**30:.0f} GB free on the disk")
        return False

    def _begin(self, frames):
        self._t0 = frames[0][0]
        h, w = frames[0][1].shape[:2]
        self._size = (w, h)
        name = dt.datetime.fromtimestamp(self._t0).strftime("%Y%m%d_%H%M%S")
        final, n = os.path.join(self.folder, name + ".mp4"), 1
        while os.path.exists(final):
            final, n = os.path.join(self.folder, f"{name}_{n}.mp4"), n + 1
        self._final = final
        self._part = os.path.join(self.folder, INCOMING, os.path.basename(final))
        self._writer = self.open_writer(self._part, self.fps, self._size)
        self._written = 0
        self.log(f"{dt.datetime.fromtimestamp(self._t0):%H:%M:%S}  person in view, recording")
        for t, frame in frames:
            self._write(t, frame)

    def _write(self, t, frame):
        due = int((t - self._t0) * self.fps) + 1 - self._written  # frames the clock says are owed
        if due <= 0:
            return
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size)
        for _ in range(due):
            self._writer.write(frame)
            self._written += 1  # counted one by one, so an interrupt cannot leave the count wrong

    def close(self):
        """Finish the clip being written, if any. Returns its path."""
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        self._hits = 0  # the next clip needs fresh sightings
        os.replace(self._part, self._final)
        seconds = self._written / self.fps
        self.log(f"{dt.datetime.fromtimestamp(self._t0 + seconds):%H:%M:%S}  saved {os.path.basename(self._final)} "
                 f"({seconds:.0f} s)")
        if self.on_close:
            self.on_close(self._final, dt.datetime.fromtimestamp(self._t0), seconds)
        return self._final


class IndexWorker(threading.Thread):
    """Indexes finished clips one at a time on its own thread, with its own models and
    its own database connection, so recording never waits for it."""

    def __init__(self, index_dir, cfg, captions=True, log=print):
        super().__init__(daemon=True)
        self.index_dir, self.cfg, self.captions, self.log = index_dir, cfg, captions, log
        self.jobs = queue.Queue()
        self._quit = threading.Event()

    def add(self, path):
        self.jobs.put(path)

    def finish(self, drain=False):
        """Stop after the clip being indexed (or, with `drain`, after every waiting clip).
        Returns how many clips were left unindexed."""
        if not drain:
            self._quit.set()
        self.jobs.put(None)
        self.join()
        return sum(1 for j in list(self.jobs.queue) if j is not None)

    def run(self):
        from .indexer import index_clip
        from .models import Models
        from .store import Store

        store, models = Store(self.index_dir), Models(self.cfg, log=self.log)
        while not self._quit.is_set():
            path = self.jobs.get()
            if path is None or self._quit.is_set():
                if path is not None:
                    self.jobs.put(path)  # leave it counted as unindexed
                break
            try:
                self.log(f"Indexing {os.path.basename(path)}")
                index_clip(path, store, self.cfg, models.detector(),
                           vlm=models.vlm() if self.captions else None,
                           embedder=models.embedder() if self.cfg.embedder else None, log=self.log)
            except Exception as e:  # noqa: BLE001  (one bad clip must not stop the recorder)
                self.log(f"  could not index {os.path.basename(path)}: {e}")
        store.db.close()


def watch(source, folder, detector, on_clip=None, detect_fps=2.0, person_conf=0.35, pre_s=3.0, post_s=5.0,
          max_s=120.0, min_free_gb=10.0, start_time=None, log=print):
    """Record a clip into `folder` whenever a person is in view, until interrupted (or,
    for a file, until it ends). Returns the number of clips saved."""
    src = FrameSource(source, log=log, start_time=start_time)
    src.start()
    saved, recorder, last_check, last_drop_report = [], None, float("-inf"), 0
    try:
        while not src.connected.wait(1.0):
            if src.ended.is_set():
                return 0
        log(f"Connected to {src.label}: {src.size[0]}x{src.size[1]} at {src.fps:.0f} frames per second. "
            "Watching for people; press Ctrl+C to stop.")

        def closed(path, start, seconds):
            saved.append(path)
            if on_clip:
                on_clip(path)

        recorder = ClipRecorder(folder, src.fps, pre_s, post_s, max_s, min_free_gb=min_free_gb, on_close=closed, log=log)
        while True:
            try:
                t, frame = src.frames.get(timeout=1.0)
            except queue.Empty:
                if src.ended.is_set():
                    break
                recorder.idle(time.time())
                continue
            person = None
            if t - last_check >= 1.0 / detect_fps:
                last_check = t
                person = any(label in SUBJECTS for label, _, _ in detector.boxes(frame, SUBJECTS, conf=person_conf))
            recorder.feed(t, frame, person)
            if src.dropped - last_drop_report >= 500:
                last_drop_report = src.dropped
                log(f"Note: {src.dropped} frames skipped so far because this PC could not keep up with the stream")
    except KeyboardInterrupt:
        log("Stopping.")
    finally:
        src.stop()
        if recorder:
            recorder.close()
    return len(saved)
