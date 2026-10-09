"""Camera addresses.

Cameras are kept in `cameras.json` in the project folder, which is excluded from git
because the address of a network camera contains its username and password:

    {"kitchen": "rtsp://USERNAME:PASSWORD@192.168.1.50:554/stream1"}

Nothing in this project prints an address without hiding that part first (`mask`).
"""

import json
import os
import re
from urllib.parse import quote

from .paths import REPO

CONFIG = os.path.join(REPO, "cameras.json")


def mask(source):
    """The address with the username and password replaced by ***."""
    return re.sub(r"//[^/@]+@", "//***@", str(source))


def rtsp_url(host, username, password, stream="stream1", port=554):
    """Address of a camera's RTSP stream. Special characters in the username and
    password are escaped, so any password works."""
    return f"rtsp://{quote(username, safe='')}:{quote(password, safe='')}@{host}:{port}/{stream}"


def load(path=CONFIG):
    """{name: address} from the camera file; empty if there is no file."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def save(cameras, path=CONFIG):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cameras, f, indent=2)
        f.write("\n")


def resolve(name=None, path=CONFIG):
    """(name, address) of the camera to use. With no name, the only camera in the file,
    or the HOMERECALL_CAMERA_URL environment variable."""
    cameras = load(path)
    if name is None:
        if len(cameras) == 1:
            return next(iter(cameras.items()))
        if not cameras and os.environ.get("HOMERECALL_CAMERA_URL"):
            return "camera", os.environ["HOMERECALL_CAMERA_URL"]
        if not cameras:
            raise LookupError("No camera is set up yet. Run: python recall.py camera add kitchen")
        raise LookupError(f"Several cameras are set up; name one of: {', '.join(cameras)}")
    if name not in cameras:
        known = f" Known cameras: {', '.join(cameras)}." if cameras else ""
        raise LookupError(f"No camera called {name!r}.{known} Add it with: python recall.py camera add {name}")
    return name, cameras[name]
