"""Loads each model the first time it is asked for, and keeps it."""


class Models:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self._cache = {}

    def _get(self, key, make):
        if key not in self._cache:
            self.log(f"Loading {key}...")
            self._cache[key] = make()
        return self._cache[key]

    def detector(self):
        from .detector import OpenVocabDetector

        return self._get(f"detector {self.cfg.detector}",
                         lambda: OpenVocabDetector(self.cfg.detector, imgsz=self.cfg.imgsz))

    def vlm(self):
        from .vlm import load_vlm

        return self._get(f"language model {self.cfg.vlm}", lambda: load_vlm(self.cfg.vlm, max_side=self.cfg.vlm_side))

    def embedder(self):
        from .embed import Embedder

        return self._get(f"embedder {self.cfg.embedder}", lambda: Embedder(self.cfg.embedder))
