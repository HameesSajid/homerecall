"""The index: one SQLite file plus a folder of key frames.

    clips    one row per recording: path, start time and where that time came from
    windows  one row per moment: when, the caption, the object tags, the crop
    frames   one row per key frame: its JPEG, and its embedding when a profile makes one
    probes   detector results for question words, cached so a word is only looked for once

Times are stored as ISO strings in the camera's local time.
"""

import datetime as dt
import json
import os
import sqlite3

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE, size INTEGER, mtime REAL,
    start TEXT, start_source TEXT, duration REAL, fps REAL, width INTEGER, height INTEGER,
    profile TEXT, indexed_at TEXT);
CREATE TABLE IF NOT EXISTS windows (
    id INTEGER PRIMARY KEY, clip_id INTEGER REFERENCES clips(id) ON DELETE CASCADE,
    t0 REAL, t1 REAL, start TEXT, end TEXT, has_subject INTEGER, n_people INTEGER,
    crop TEXT, caption TEXT, objects TEXT);
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY, window_id INTEGER REFERENCES windows(id) ON DELETE CASCADE,
    t REAL, path TEXT, context_path TEXT, emb BLOB);
CREATE TABLE IF NOT EXISTS probes (
    window_id INTEGER REFERENCES windows(id) ON DELETE CASCADE, term TEXT, conf REAL,
    PRIMARY KEY (window_id, term));
CREATE INDEX IF NOT EXISTS windows_start ON windows(start);
CREATE INDEX IF NOT EXISTS frames_window ON frames(window_id);
"""


def iso(when):
    return when.replace(microsecond=0).isoformat(sep=" ")


class Store:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(os.path.join(self.root, "frames"), exist_ok=True)
        # The recorder's indexing thread and an `ask` in another window may use the index
        # at the same time; wait for the other to finish instead of failing.
        self.db = sqlite3.connect(os.path.join(self.root, "index.db"), timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(_SCHEMA)

    # ---- writing

    def is_indexed(self, path, size, mtime):
        row = self.db.execute("SELECT size, mtime FROM clips WHERE path = ?", (path,)).fetchone()
        return bool(row) and row["size"] == size and abs(row["mtime"] - mtime) < 1.0

    def remove_clip(self, path):
        """Forget a clip: its rows and its key frame files."""
        row = self.db.execute("SELECT id FROM clips WHERE path = ?", (path,)).fetchone()
        if not row:
            return
        folder = os.path.join(self.root, "frames", str(row["id"]))
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                os.remove(os.path.join(folder, name))
            os.rmdir(folder)
        self.db.execute("DELETE FROM clips WHERE id = ?", (row["id"],))
        self.db.commit()

    def add_clip(self, path, size, mtime, start, start_source, duration, fps, width, height, profile):
        cur = self.db.execute(
            "INSERT INTO clips (path, size, mtime, start, start_source, duration, fps, width, height, profile, "
            "indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (path, size, mtime, iso(start), start_source, duration, fps, width, height, profile,
             iso(dt.datetime.now())))
        os.makedirs(os.path.join(self.root, "frames", str(cur.lastrowid)), exist_ok=True)
        return cur.lastrowid

    def add_window(self, clip_id, clip_start, t0, t1, has_subject, n_people, crop, caption, objects):
        cur = self.db.execute(
            "INSERT INTO windows (clip_id, t0, t1, start, end, has_subject, n_people, crop, caption, objects) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (clip_id, t0, t1, iso(clip_start + dt.timedelta(seconds=t0)), iso(clip_start + dt.timedelta(seconds=t1)),
             int(has_subject), n_people, json.dumps(crop), caption, json.dumps(objects)))
        return cur.lastrowid

    def add_frame(self, window_id, t, path, context_path, emb=None):
        blob = None if emb is None else np.asarray(emb, np.float16).tobytes()
        self.db.execute("INSERT INTO frames (window_id, t, path, context_path, emb) VALUES (?,?,?,?,?)",
                        (window_id, t, self.rel(path), self.rel(context_path), blob))

    def set_probes(self, rows):
        self.db.executemany("INSERT OR REPLACE INTO probes (window_id, term, conf) VALUES (?,?,?)", rows)
        self.db.commit()

    def commit(self):
        self.db.commit()

    # ---- paths: stored relative to the index folder, so the index can be moved or
    # built on another machine

    def rel(self, path):
        return os.path.relpath(path, self.root).replace("\\", "/")

    def abs(self, rel):
        return os.path.join(self.root, *rel.split("/"))

    def frame_dir(self, clip_id):
        return os.path.join(self.root, "frames", str(clip_id))

    # ---- reading

    def windows(self, start=None, end=None):
        """Moments that overlap [start, end), oldest first, each with its clip's fields."""
        sql = ("SELECT w.*, c.path AS clip_path, c.start AS clip_start, c.start_source, c.duration "
               "FROM windows w JOIN clips c ON c.id = w.clip_id WHERE 1=1")
        args = []
        if start is not None:
            sql += " AND w.end >= ?"
            args.append(iso(start))
        if end is not None:
            sql += " AND w.start < ?"
            args.append(iso(end))
        return [dict(r) for r in self.db.execute(sql + " ORDER BY w.start, w.id", args)]

    def frames(self, window_id):
        return [dict(r) for r in self.db.execute("SELECT * FROM frames WHERE window_id = ? ORDER BY t", (window_id,))]

    def embeddings(self, window_ids):
        """{window id: array [n frames, dim]} for the windows that have embeddings."""
        out = {}
        for wid in window_ids:
            rows = [np.frombuffer(r["emb"], np.float16).astype(np.float32)
                    for r in self.db.execute("SELECT emb FROM frames WHERE window_id = ? AND emb IS NOT NULL", (wid,))]
            if rows:
                out[wid] = np.stack(rows)
        return out

    def probes(self, terms):
        """{window id: {term: conf}} for cached detector lookups of `terms`."""
        out = {}
        if not terms:
            return out
        marks = ",".join("?" * len(terms))
        for r in self.db.execute(f"SELECT * FROM probes WHERE term IN ({marks})", list(terms)):
            out.setdefault(r["window_id"], {})[r["term"]] = r["conf"]
        return out

    def stats(self):
        one = lambda q: self.db.execute(q).fetchone()[0]  # noqa: E731
        return {
            "clips": one("SELECT COUNT(*) FROM clips"),
            "moments": one("SELECT COUNT(*) FROM windows"),
            "key_frames": one("SELECT COUNT(*) FROM frames"),
            "with_embeddings": one("SELECT COUNT(*) FROM frames WHERE emb IS NOT NULL"),
            "first": one("SELECT MIN(start) FROM windows"),
            "last": one("SELECT MAX(end) FROM windows"),
            "time_sources": {r[0]: r[1] for r in self.db.execute(
                "SELECT start_source, COUNT(*) FROM clips GROUP BY start_source")},
            "profiles": [r[0] for r in self.db.execute("SELECT DISTINCT profile FROM clips")],
        }
