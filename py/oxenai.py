"""Oxen AI Chat Completions API node.

Calls the OpenAI-compatible chat completions endpoint hosted by Oxen.ai:

    POST https://hub.oxen.ai/api/ai/chat/completions
    Authorization: Bearer $OXEN_API_KEY

Docs: https://docs.oxen.ai/examples/inference/chat_completions
"""

import os
import io
import json
import base64
import requests

import numpy as np
from PIL import Image

from .status_utils import EventLog, push_node_status, finalize_status


OXEN_CHAT_COMPLETIONS_URL = "https://hub.oxen.ai/api/ai/chat/completions"
HTTP_TIMEOUT = (10, 300)

# Display label -> model id used in the API request body.
# Keep insertion order — it controls dropdown order in the ComfyUI widget.
OXEN_MODEL_MAP = {
    "Claude Opus 4.7": "claude-opus-4-7",
    "GPT 5.5": "gpt-5-5-2026-04-23",
    "DeepSeek V4 Pro": "deepseek-v4-pro",
    "GPT 5.5 Pro": "gpt-5-5-pro-2026-04-23",
    "QWEN 3.6 Plus": "qwen3-6-plus",
    "Opus 4.6": "claude-opus-4-6",
}
OXEN_MODEL_LABELS = list(OXEN_MODEL_MAP.keys())

# Max images per IMAGE input. Each ComfyUI IMAGE tensor may itself be a batch
# (B, H, W, C); we iterate frames and encode each one as its own image_url part.
MAX_IMAGE_INPUTS = 4


def _tensor_to_data_urls(image_tensor, fmt="PNG", jpeg_quality=92):
    """Convert a ComfyUI IMAGE tensor (B,H,W,C float in 0-1) to data URLs.

    Returns a list of data: URLs, one per frame in the batch.
    """
    if image_tensor is None:
        return []
    arr = image_tensor
    # Torch tensor -> numpy
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(arr)

    # ComfyUI shape is (B, H, W, C). If we got (H, W, C) treat as batch=1.
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        return []

    urls = []
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    for frame in arr:
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        if frame_u8.shape[-1] == 4:
            pil = Image.fromarray(frame_u8, mode="RGBA")
            if fmt.upper() != "PNG":
                pil = pil.convert("RGB")
        else:
            pil = Image.fromarray(frame_u8[..., :3], mode="RGB")

        buf = io.BytesIO()
        if fmt.upper() == "PNG":
            pil.save(buf, format="PNG", optimize=False)
        else:
            pil.save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:{mime};base64,{b64}")
    return urls


class Soze_OxenAIChatCompletion:
    """Call Oxen.ai's chat completions API and return the assistant text.

    Optional IMAGE inputs are encoded as base64 data URLs and attached to the
    user message in OpenAI-compatible vision format. Not every Oxen-hosted model
    supports vision — if the chosen model rejects images, the API will return
    an error in `status`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "system_prompt": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Optional system message prepended to the conversation.",
            }),
            "temperature": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Sampling temperature (0–2). Pass -1 to omit and let the model default.",
            }),
            "max_tokens": ("INT", {
                "default": 0, "min": 0, "max": 1048576, "step": 1,
                "tooltip": "Max output tokens. 0 = omit and let the model default.",
            }),
            "image_format": (["PNG", "JPEG"], {
                "default": "JPEG",
                "tooltip": "Encoding for attached images. JPEG is smaller; PNG is lossless.",
            }),
        }
        for i in range(MAX_IMAGE_INPUTS):
            optional[f"image{i+1}"] = ("IMAGE", {
                "tooltip": f"Optional image attached to the user message (slot {i+1}). "
                           "Batch tensors are sent as multiple images.",
            })
        return {
            "required": {
                "model": (OXEN_MODEL_LABELS, {"default": OXEN_MODEL_LABELS[0]}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "User message sent to the model.",
                }),
            },
            "optional": optional,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("response_text", "prompt_tokens", "completion_tokens", "total_tokens", "status")
    FUNCTION = "chat"
    CATEGORY = "soze/oxenai"

    def chat(self, model, prompt, system_prompt="", temperature=1.0, max_tokens=0,
             image_format="JPEG", unique_id=None, **image_kwargs):
        log = EventLog()

        key = os.environ.get("OXEN_API_KEY", "").strip()
        if not key:
            headline = "ERROR: missing Oxen API key — set the OXEN_API_KEY environment variable."
            push_node_status(unique_id, headline, log)
            raise ValueError(headline)

        model_id = OXEN_MODEL_MAP.get(model, model)

        # Collect images in input-slot order (image1, image2, ...)
        image_urls = []
        for i in range(MAX_IMAGE_INPUTS):
            img = image_kwargs.get(f"image{i+1}")
            if img is None:
                continue
            try:
                urls = _tensor_to_data_urls(img, fmt=image_format)
                if urls:
                    push_node_status(unique_id, f"image{i+1}: encoded {len(urls)} frame(s)", log)
                image_urls.extend(urls)
            except Exception as e:
                push_node_status(unique_id, f"WARN image{i+1}: encode failed ({e!r})", log)

        # Build the user message — string content when no images, content-array
        # (OpenAI vision format) when images are attached.
        if image_urls:
            user_content = []
            if prompt and prompt.strip():
                user_content.append({"type": "text", "text": prompt})
            for url in image_urls:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
            user_message = {"role": "user", "content": user_content}
        else:
            user_message = {"role": "user", "content": prompt or ""}

        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append(user_message)

        payload = {
            "model": model_id,
            "messages": messages,
        }
        if temperature is not None and temperature >= 0:
            payload["temperature"] = float(temperature)
        if max_tokens and int(max_tokens) > 0:
            payload["max_tokens"] = int(max_tokens)

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

        push_node_status(
            unique_id,
            f"POST {model_id} ({len(messages)} msg, {len(image_urls)} image(s))",
            log,
        )
        try:
            resp = requests.post(
                OXEN_CHAT_COMPLETIONS_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as e:
            headline = f"ERROR: request failed: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        if resp.status_code >= 400:
            body = resp.text[:1000]
            headline = f"ERROR: HTTP {resp.status_code} — {body}"
            push_node_status(unique_id, headline, log)
            resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError as e:
            headline = f"ERROR: invalid JSON response: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        try:
            choice = data["choices"][0]
            message = choice.get("message", {}) or {}
            text = message.get("content", "") or ""
            finish_reason = choice.get("finish_reason", "")
        except (KeyError, IndexError, TypeError) as e:
            headline = f"ERROR: unexpected response shape: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        headline = (
            f"OK: {model_id} — {completion_tokens} out / {prompt_tokens} in "
            f"({total_tokens} total){f', finish={finish_reason}' if finish_reason else ''}"
            f"{f', images={len(image_urls)}' if image_urls else ''}"
        )
        push_node_status(unique_id, headline, log)

        return (text, prompt_tokens, completion_tokens, total_tokens, finalize_status(headline, log))
