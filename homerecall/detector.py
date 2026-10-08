"""Open-vocabulary detection with YOLOE (adapted from LiveScene's `perception.py`).

The detector is prompted with words, so the same model finds people while indexing, tags
household objects, checks the objects a caption names, and looks for the nouns of a
question that nobody anticipated at indexing time.
"""

import os

import torch

from .paths import WEIGHTS, downloads_forbidden, in_weights_dir
from .vocab import SUBJECTS


class OpenVocabDetector:
    """YOLOE promptable detector driven by text prompts."""

    def __init__(self, weights="yoloe-26m-seg.pt", vocab=None, device="cuda", imgsz=960, conf=0.1):
        from ultralytics import YOLOE

        if downloads_forbidden() and not os.path.isfile(os.path.join(WEIGHTS, weights)):
            raise FileNotFoundError(
                f"{weights} is not in {WEIGHTS} and this profile does not download models. "
                "Use --allow-download, or run a Colab profile on Colab.")
        self.weights = weights
        with in_weights_dir():
            self.model = YOLOE(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self._pe = {}  # word -> text embedding, so each word is encoded once
        self.set_vocab(vocab or SUBJECTS)

    def embed(self, names):
        """Text embeddings for `names` as {name: tensor}."""
        missing = [n for n in dict.fromkeys(names) if n not in self._pe]
        if missing:
            with in_weights_dir():  # the text encoder is loaded from here on first use
                pe = self.model.get_text_pe(missing)
            for i, n in enumerate(missing):
                self._pe[n] = pe[:, i:i + 1].detach()
        return {n: self._pe[n] for n in names}

    def set_vocab(self, vocab):
        """Re-prompt the detector. With cached embeddings this takes milliseconds."""
        vocab = list(dict.fromkeys(vocab))
        if vocab == getattr(self, "vocab", None):
            return
        self.vocab = vocab
        pe = self.embed(vocab)
        self.model.set_classes(vocab, torch.cat([pe[n] for n in vocab], dim=1))

    @torch.inference_mode()
    def detect(self, frame_bgr, conf=None):
        """Raw Ultralytics result for one frame."""
        return self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
                                  device=self.device, verbose=False)[0]

    def boxes(self, frame_bgr, vocab, conf=0.3):
        """[(label, confidence, (x1, y1, x2, y2) in 0..1)] for `vocab`."""
        self.set_vocab(vocab)
        r = self.detect(frame_bgr, conf=conf)
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxyn = r.boxes.xyxyn.cpu().tolist()
        return [(r.names[int(c)], float(p), tuple(b))
                for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), xyxyn)]

    def check(self, frames_bgr, names, conf=0.05, competitors=()):
        """{name: best confidence over the frames} for each of `names` that is found.

        The detector assigns a region to the closest prompt it was given, so `competitors`
        give regions that are something else a better label to go to."""
        names = list(dict.fromkeys(names))
        if not names:
            return {}
        self.set_vocab(names + [c for c in competitors if c not in names])
        best = {}
        for frame in frames_bgr:
            r = self.detect(frame, conf=conf)
            if r.boxes is None:
                continue
            for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = r.names[int(c)]
                best[name] = max(best.get(name, 0.0), float(p))
        return {n: best[n] for n in names if n in best}
