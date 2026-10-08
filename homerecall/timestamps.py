"""When did a recording start?

A clip is only useful as evidence if its date and time are right, so three sources are
tried in order of reliability and the one that was used is stored with the clip:

1. `filename`       a date and time in the file name (how camera apps name exports)
2. `metadata`       the creation time inside the MP4 container
3. `file-modified`  the file's modification time (when it was saved, so only approximate)
"""

import datetime as dt
import os
import re
import struct

# 20261006_074210, 2026-10-06 07-42-10, 2026.10.06_07.42.10, 20261006074210 ...
_STAMP = re.compile(r"(?<!\d)(20\d{2})[-_.]?([01]\d)[-_.]?([0-3]\d)[-_ T.]?([0-2]\d)[-_.:h]?([0-5]\d)[-_.:m]?([0-5]\d)(?!\d)")
_MP4_EPOCH = dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc)


def from_filename(path):
    """The first valid date-time found in the file name, or None."""
    for m in _STAMP.finditer(os.path.basename(path)):
        try:
            return dt.datetime(*map(int, m.groups()))
        except ValueError:
            continue
    return None


def _atoms(f, start, end):
    """Yield (type, body offset, body end) for the MP4 atoms between two file offsets."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        header = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        yield kind, pos + header, pos + size
        pos += size


def from_metadata(path):
    """Creation time from the MP4 movie header, converted to local time, or None.
    Cameras that never set their clock write zero (the year 1904); that is ignored."""
    try:
        with open(path, "rb") as f:
            total = os.fstat(f.fileno()).st_size
            for kind, a, b in _atoms(f, 0, total):
                if kind != b"moov":
                    continue
                for sub, sa, _ in _atoms(f, a, b):
                    if sub != b"mvhd":
                        continue
                    f.seek(sa)
                    version = f.read(4)[0]
                    seconds = struct.unpack(">Q" if version == 1 else ">I", f.read(8 if version == 1 else 4))[0]
                    when = _MP4_EPOCH + dt.timedelta(seconds=seconds)
                    if when.year < 2005:
                        return None
                    return when.astimezone().replace(tzinfo=None)
    except (OSError, struct.error, OverflowError):
        pass
    return None


def clip_start(path, duration=0.0):
    """(start as a local naive datetime, source name)."""
    when = from_filename(path)
    if when:
        return when, "filename"
    when = from_metadata(path)
    if when:
        return when, "metadata"
    # A file is finished being written at about the end of the recording.
    saved = dt.datetime.fromtimestamp(os.path.getmtime(path)).replace(microsecond=0)
    return saved - dt.timedelta(seconds=duration), "file-modified"
