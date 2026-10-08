"""Model profiles.

Indexing is offline, so every profile favours accuracy over speed; they differ in how much
GPU memory the models need. `auto` picks by the memory that is present.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    name: str
    detector: str            # YOLOE weights: finds people, tags objects, checks proposed objects
    imgsz: int               # detector input size
    vlm: str                 # spec for homerecall.vlm.load_vlm
    vlm_side: int            # longest image side given to the language model
    frames: int              # key frames kept per moment (and shown to the language model)
    caption_tokens: int      # new tokens per caption
    two_step: bool           # ask for caption and object list separately (small models)
    embedder: str | None     # image-text embedding model for semantic retrieval, or None
    sample_fps: float = 2.0  # frames per second examined for people
    window_s: float = 8.0    # longest moment; longer activity is split into several
    overlap_s: float = 2.0   # overlap between the pieces of a split
    keyframe_side: int = 896  # longest side of a stored key frame
    verify_top: int = 24     # candidates checked by the language model per question
    tag_threshold: float = 0.25  # detector confidence for an object tag to count as confirmed
    allow_download: bool = True  # False: a model that is not on disk is an error


PROFILES = {
    # 6 GB GPU. Uses only models LiveScene already downloaded, and refuses to fetch others.
    "laptop": Config(
        name="laptop", detector="yoloe-26m-seg.pt", imgsz=960,
        vlm="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", vlm_side=512, frames=4, caption_tokens=70,
        two_step=True, embedder=None, keyframe_side=768, verify_top=24, allow_download=False),
    # 12 to 16 GB (Colab T4): 3B language model in half precision, large detector, SigLIP 2.
    "colab-t4": Config(
        name="colab-t4", detector="yoloe-26l-seg.pt", imgsz=1280,
        vlm="Qwen/Qwen2.5-VL-3B-Instruct@fp16", vlm_side=728, frames=6, caption_tokens=140,
        two_step=False, embedder="google/siglip2-base-patch16-384", verify_top=40),
    # 22 GB or more (Colab L4 / A100): 7B language model, largest detector and embedder.
    "colab": Config(
        name="colab", detector="yoloe-26x-seg.pt", imgsz=1280,
        vlm="Qwen/Qwen2.5-VL-7B-Instruct@fp16", vlm_side=728, frames=8, caption_tokens=160,
        two_step=False, embedder="google/siglip2-so400m-patch14-384", verify_top=60),
}


def pick(name="auto", **overrides):
    """Resolve a profile name (or `auto`) and apply any explicit overrides that are not None."""
    if name == "auto":
        import torch

        gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
        name = "colab" if gb >= 22 else "colab-t4" if gb >= 12 else "laptop"
    return replace(PROFILES[name], **{k: v for k, v in overrides.items() if v is not None})
