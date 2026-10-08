"""Vision-language model backends (adapted from LiveScene's `vlm.py`).

Every backend exposes two blocking calls:

    generate(frames_bgr, prompt, max_new_tokens) -> str
    yes_probability(frames_bgr, question) -> float in 0..1

`frames_bgr` is a list of BGR numpy frames in time order.
"""

import base64
import json
import urllib.request

import cv2
import torch
from PIL import Image

YES_NO_SUFFIX = " Answer yes or no."


def _to_pil(frame_bgr, max_side):
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    img.thumbnail((max_side, max_side))
    return img


def parse_yes_no(reply):
    """1.0 / 0.0 from a text reply; 0.5 when it is neither."""
    word = reply.strip().lower().lstrip("*\"' ")
    return 1.0 if word.startswith("yes") else 0.0 if word.startswith("no") else 0.5


class HFVLM:
    """Any Hugging Face image-text-to-text chat model, run in-process with PyTorch."""

    def __init__(self, model_id, device="cuda", dtype="float32", max_side=512, load_4bit=False):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.name = model_id
        self.device = device
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):
            # SmolVLM-style processors upscale and tile the image into many crops (over a
            # thousand visual tokens). One view per frame is far faster.
            ip.do_image_splitting = False
            ip.size = {"longest_edge": max_side}
        kwargs = {"dtype": getattr(torch, dtype)}
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
            kwargs["device_map"] = {"": device}
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        if not load_4bit:
            self.model = self.model.to(device)
        self.model.eval()
        self._yes, self._no = self._answer_tokens()

    def _answer_tokens(self):
        """Ids of every single token that starts a yes or a no answer."""
        tok = self.processor.tokenizer
        ids = {}
        for key, words in (("yes", ("Yes", "yes", " Yes", " yes", "YES")), ("no", ("No", "no", " No", " no", "NO"))):
            found = set()
            for w in words:
                enc = tok.encode(w, add_special_tokens=False)
                if len(enc) == 1:
                    found.add(enc[0])
            ids[key] = sorted(found)
        return ids["yes"], ids["no"]

    def _inputs(self, frames_bgr, prompt):
        content = [{"type": "image", "image": _to_pil(f, self.max_side)} for f in frames_bgr]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, processor_kwargs={"return_tensors": "pt"}).to(self.device)
        dtype = next(self.model.parameters()).dtype
        return {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in inputs.items()}

    @torch.inference_mode()
    def generate(self, frames_bgr, prompt, max_new_tokens=48):
        inputs = self._inputs(frames_bgr, prompt)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def yes_probability(self, frames_bgr, question):
        """P(yes) / (P(yes) + P(no)) for the first word of the answer.

        Reading the probability instead of the generated word gives a graded score, so
        candidates can be ranked and a threshold chosen, where the text alone would only
        say yes or no."""
        if not self._yes or not self._no:
            return parse_yes_no(self.generate(frames_bgr, question + YES_NO_SUFFIX, 4))
        inputs = self._inputs(frames_bgr, question + YES_NO_SUFFIX)
        probs = torch.softmax(self.model(**inputs).logits[0, -1].float(), dim=-1)
        yes, no = probs[self._yes].sum().item(), probs[self._no].sum().item()
        return yes / (yes + no) if yes + no > 1e-6 else 0.5


class ServerVLM:
    """A model behind an OpenAI-compatible chat endpoint (for example llama.cpp's server)."""

    def __init__(self, url, model="default", max_side=512, timeout=120):
        self.name = f"{url} ({model})"
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_side = max_side
        self.timeout = timeout

    def generate(self, frames_bgr, prompt, max_new_tokens=48):
        content = []
        for f in frames_bgr:
            h, w = f.shape[:2]
            s = min(1.0, self.max_side / max(h, w))
            small = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1.0 else f
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_new_tokens, "temperature": 0}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"].strip()

    def yes_probability(self, frames_bgr, question):
        return parse_yes_no(self.generate(frames_bgr, question + YES_NO_SUFFIX, 4))


def load_vlm(spec, **kwargs):
    """`spec` is a Hugging Face model id, or `server:<base url>` for an HTTP endpoint.
    Append `@4bit` or `@fp16` to a model id to change how it is loaded."""
    if spec.startswith("server:"):
        return ServerVLM(spec[len("server:"):], **kwargs)
    model_id, _, mode = spec.partition("@")
    if mode == "4bit":
        return HFVLM(model_id, load_4bit=True, **kwargs)
    if mode == "fp16":
        return HFVLM(model_id, dtype="float16", **kwargs)
    return HFVLM(model_id, **kwargs)
