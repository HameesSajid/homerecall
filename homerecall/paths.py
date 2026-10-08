"""Where models, the index and results are kept.

Import this before transformers or huggingface_hub so HF_HOME takes effect.
"""

import contextlib
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model downloads. HOMERECALL_CACHE wins. Otherwise this project's own .cache is used, unless
# it does not exist and a LiveScene checkout sits next to this one: the two projects use the
# same detector and language model, so its downloads are reused instead of fetched twice.
_own = os.path.join(REPO, ".cache")
_shared = os.path.join(os.path.dirname(REPO), "livescene", ".cache")
CACHE = os.environ.get("HOMERECALL_CACHE") or (_shared if os.path.isdir(_shared) and not os.path.isdir(_own) else _own)
os.environ.setdefault("HF_HOME", os.path.join(CACHE, "hf"))

WEIGHTS = os.path.join(CACHE, "weights")
INDEX = os.environ.get("HOMERECALL_INDEX", os.path.join(REPO, "index"))
RESULTS = os.path.join(REPO, "results")


def forbid_downloads():
    """Make a missing model an error instead of a silent multi-gigabyte download."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def downloads_forbidden():
    return os.environ.get("HF_HUB_OFFLINE") == "1"


@contextlib.contextmanager
def in_weights_dir():
    """Ultralytics looks for (and downloads) weights in the working directory."""
    os.makedirs(WEIGHTS, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(WEIGHTS)
    try:
        yield
    finally:
        os.chdir(cwd)
