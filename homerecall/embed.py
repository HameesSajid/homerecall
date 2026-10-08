"""Image-text embeddings (SigLIP 2) for semantic retrieval.

Key frames are embedded once at indexing time; a question is embedded at search time and
compared with them. This finds moments whose caption used different words from the
question, or missed the thing asked about altogether.

Only the Colab profiles use it. The laptop profile has no embedder, and search falls back
to captions, tags and detector lookups.
"""

import torch
from PIL import Image


class Embedder:
    def __init__(self, model_id, device="cuda"):
        from transformers import AutoModel, AutoProcessor

        self.name = model_id
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def images(self, frames_bgr):
        """Unit-length embeddings, one row per frame."""
        pil = [Image.fromarray(f[:, :, ::-1]) for f in frames_bgr]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(pixel_values=inputs["pixel_values"].to(self.model.dtype))
        out = getattr(out, "pooler_output", out)
        return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def text(self, sentence):
        """Unit-length embedding of one sentence. SigLIP was trained on lower-case text
        padded to 64 tokens, so the question is given the same way."""
        inputs = self.processor(text=[sentence.lower().strip()], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(input_ids=inputs["input_ids"])
        out = getattr(out, "pooler_output", out)
        return torch.nn.functional.normalize(out.float(), dim=-1).cpu().numpy()[0]
