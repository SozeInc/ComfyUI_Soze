"""MiniMax (Hailuo) H3 video generation node.

Async task API:
    POST {base}/v2/video_generation                    -> {"task_id": "..."}
    GET  {base}/v2/query/video_generation/{task_id}     -> {"task": {"status", "content": {"url"}, "error"}}

Auth:  Authorization: Bearer $MINIMAX_API_KEY
Base:  https://api.minimax.io (override with MINIMAX_BASE_URL, e.g. the CN host)

Docs: https://platform.minimax.io/docs/guides/video-generation
"""

import io
import os
import base64
import logging
import asyncio

import numpy as np
import requests
import torch
from PIL import Image

from comfy_api_nodes.util import download_url_to_video_output

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)

MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io"
MINIMAX_CREATE_PATH = "/v2/video_generation"
MINIMAX_QUERY_PATH = "/v2/query/video_generation"
MINIMAX_MODEL_ID = "MiniMax-H3"

HTTP_TIMEOUT = (10, 120)
DEFAULT_POLL_INTERVAL = 10   # seconds between status polls
DEFAULT_POLL_TIMEOUT = 900   # max seconds to wait for a task

MINIMAX_RESOLUTION_CHOICES = ["2K", "1080P", "768P"]
MINIMAX_RATIO_CHOICES = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]


def _minimax_api_key():
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MINIMAX_API_KEY is not set. Add it to your environment.")
    return key


def _minimax_base_url():
    return os.environ.get("MINIMAX_BASE_URL", MINIMAX_DEFAULT_BASE_URL).rstrip("/")


def _tensor_to_data_uri(image_tensor, fmt="PNG"):
    """Convert the first frame of a ComfyUI IMAGE tensor to a base64 data URI."""
    if image_tensor is None:
        return None
    arr = image_tensor
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    a = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if a.shape[-1] == 4:
        pil = Image.fromarray(a, mode="RGBA").convert("RGB")
    else:
        pil = Image.fromarray(a[..., :3], mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _check_base_resp(data):
    """MiniMax returns HTTP 200 with a base_resp.status_code != 0 on logical errors."""
    base = data.get("base_resp") if isinstance(data, dict) else None
    if isinstance(base, dict) and int(base.get("status_code", 0) or 0) != 0:
        raise RuntimeError(
            f"MiniMax error: status_code={base.get('status_code')} | {base.get('status_msg')!r}"
        )


def _extract_task_id(data):
    if not isinstance(data, dict):
        return None
    if data.get("task_id"):
        return data["task_id"]
    task = data.get("task")
    if isinstance(task, dict):
        return task.get("task_id")
    return None


def _extract_status_and_url(data):
    """Return (status_lower, download_url_or_None) from a query response."""
    task = data.get("task") if isinstance(data, dict) and isinstance(data.get("task"), dict) else data
    status = (task.get("status") or "").lower() if isinstance(task, dict) else ""
    url = None
    content = task.get("content") if isinstance(task, dict) else None
    if isinstance(content, dict):
        url = content.get("url")
    # Fallbacks seen across MiniMax revisions.
    if not url and isinstance(task, dict):
        url = task.get("video_url") or task.get("download_url")
    return status, url


MINIMAX_MAX_REFERENCE_IMAGES = 9  # MiniMax H3 accepts up to 9 reference images.


async def _submit_and_download(content, duration, resolution, ratio,
                               poll_interval, poll_timeout, unique_id, log, label=""):
    """Shared MiniMax flow: create task -> poll -> download MP4.

    Returns (video, video_url, task_id, status_string). All error paths return
    a (None, ...) tuple with a finalized status so nodes can return it directly.
    """
    try:
        key = _minimax_api_key()
    except RuntimeError as e:
        headline = f"ERROR: {e}"
        push_node_status(unique_id, headline, log)
        return (None, "", "", finalize_status(headline, log))

    payload = {
        "model": MINIMAX_MODEL_ID,
        "content": content,
        "duration": int(duration),
        "resolution": resolution,
        "ratio": ratio,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    base = _minimax_base_url()

    # 1) Create the task.
    create_url = base + MINIMAX_CREATE_PATH
    push_node_status(unique_id, f"POST {create_url}{(' | ' + label) if label else ''}", log)
    try:
        resp = requests.post(create_url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        headline = f"ERROR: create request failed: {e!r}"
        push_node_status(unique_id, headline, log)
        return (None, "", "", finalize_status(headline, log))

    if resp.status_code >= 400:
        headline = f"ERROR: create HTTP {resp.status_code} — {resp.text[:500]}"
        push_node_status(unique_id, headline, log)
        return (None, "", "", finalize_status(headline, log))

    try:
        create_data = resp.json()
        _check_base_resp(create_data)
    except Exception as e:
        headline = f"ERROR: bad create response: {e!r} | {resp.text[:300]}"
        push_node_status(unique_id, headline, log)
        return (None, "", "", finalize_status(headline, log))

    task_id = _extract_task_id(create_data)
    if not task_id:
        headline = f"ERROR: no task_id in create response: {create_data!r}"
        push_node_status(unique_id, headline, log)
        return (None, "", "", finalize_status(headline, log))
    push_node_status(unique_id, f"Task created: {task_id}", log)

    # 2) Poll to completion.
    query_url = f"{base}{MINIMAX_QUERY_PATH}/{task_id}"
    elapsed = 0
    last_status = None
    video_url = None
    while True:
        try:
            q = requests.get(query_url, headers=headers, timeout=HTTP_TIMEOUT)
            q.raise_for_status()
            qdata = q.json()
        except Exception as e:
            headline = f"ERROR polling task {task_id}: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", task_id, finalize_status(headline, log))

        status, url = _extract_status_and_url(qdata)
        if status != last_status:
            push_node_status(unique_id, f"Task {task_id} status: {status or '(unknown)'}", log)
            last_status = status

        if status in ("succeeded", "success", "completed"):
            video_url = url
            break
        if status in ("failed", "fail", "cancelled", "canceled", "expired"):
            task = qdata.get("task") if isinstance(qdata.get("task"), dict) else qdata
            err = task.get("error") if isinstance(task, dict) else None
            headline = f"ERROR: MiniMax task ended '{status}'" + (f" | {err!r}" if err else "")
            push_node_status(unique_id, headline, log)
            return (None, "", task_id, finalize_status(headline, log))
        if elapsed >= poll_timeout:
            headline = f"ERROR: task {task_id} did not finish within {poll_timeout}s (last status: {status or 'unknown'})."
            push_node_status(unique_id, headline, log)
            return (None, "", task_id, finalize_status(headline, log))

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    if not video_url:
        headline = f"ERROR: task {task_id} succeeded but no video URL was returned: {qdata!r}"
        push_node_status(unique_id, headline, log)
        return (None, "", task_id, finalize_status(headline, log))

    # 3) Download the MP4.
    push_node_status(unique_id, "Downloading video...", log)
    try:
        video = await download_url_to_video_output(video_url)
    except Exception as e:
        logger.exception("MiniMax H3 video download failed (url=%s)", video_url)
        headline = f"ERROR downloading video: {e!r}"
        push_node_status(unique_id, headline, log)
        return (None, video_url, task_id, finalize_status(headline, log))

    headline = f"OK (MiniMax-H3): {resolution} {ratio} {duration}s, task={task_id}"
    push_node_status(unique_id, headline, log)
    return (video, video_url, task_id, finalize_status(headline, log))


def _collect_reference_image_uris(slots, image_batch, max_count):
    """Collect image_1..image_N (slots win) or fan out image_batch → data URIs."""
    uris = []
    # Individual slots take precedence.
    slot_tensors = []
    for i in range(1, max_count + 1):
        t = slots.get(f"image_{i}")
        if t is not None:
            slot_tensors.append(t[0:1] if hasattr(t, "shape") else t)
    if slot_tensors:
        sources = slot_tensors
    elif image_batch is not None and hasattr(image_batch, "shape") and image_batch.shape[0] > 0:
        sources = [image_batch[i:i+1] for i in range(min(image_batch.shape[0], max_count))]
    else:
        sources = []
    for t in sources:
        uri = _tensor_to_data_uri(t)
        if uri:
            uris.append(uri)
    return uris


class Soze_MiniMaxH3Video:
    """MiniMax (Hailuo) H3 video generation.

    Text-to-video, or image-to-video by connecting a first_frame_image (and
    optionally a last_frame_image). Submits the task, polls to completion, and
    downloads the resulting MP4.

    Set MINIMAX_API_KEY in your environment (and optionally MINIMAX_BASE_URL).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt (<= 7000 chars)."}),
                "duration": ("INT", {"default": 6, "min": 4, "max": 15, "step": 1, "tooltip": "Clip length in seconds (4-15)."}),
                "resolution": (MINIMAX_RESOLUTION_CHOICES, {"default": "2K"}),
                "ratio": (MINIMAX_RATIO_CHOICES, {"default": "16:9", "tooltip": "Use 'adaptive' for image-to-video to match the input frame."}),
            },
            "optional": {
                "first_frame_image": ("IMAGE", {"tooltip": "Optional first frame (image-to-video). Sent as role=first_frame."}),
                "last_frame_image": ("IMAGE", {"tooltip": "Optional last frame. Sent as role=last_frame."}),
                "poll_interval": ("INT", {"default": DEFAULT_POLL_INTERVAL, "min": 1, "max": 60, "step": 1, "tooltip": "Seconds between status polls."}),
                "poll_timeout": ("INT", {"default": DEFAULT_POLL_TIMEOUT, "min": 30, "max": 3600, "step": 30, "tooltip": "Max seconds to wait before giving up."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "status")
    FUNCTION = "generate"
    CATEGORY = "MiniMax/VideoGeneration"

    async def generate(self, prompt, duration, resolution, ratio,
                       first_frame_image=None, last_frame_image=None,
                       poll_interval=DEFAULT_POLL_INTERVAL, poll_timeout=DEFAULT_POLL_TIMEOUT,
                       unique_id=None):
        log = EventLog()
        log.add(f"Config: MiniMax-H3 | {resolution} | {ratio} | {duration}s")

        if not prompt or not prompt.strip():
            headline = "MiniMax H3 skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", "", finalize_status(headline, log))

        # Build the multimodal content array (text + optional first/last frame).
        content = [{"type": "text", "text": prompt.strip()[:7000]}]
        if first_frame_image is not None:
            uri = _tensor_to_data_uri(first_frame_image)
            if uri:
                content.append({"type": "image_url", "image_url": {"url": uri}, "role": "first_frame"})
        if last_frame_image is not None:
            uri = _tensor_to_data_uri(last_frame_image)
            if uri:
                content.append({"type": "image_url", "image_url": {"url": uri}, "role": "last_frame"})

        label = f"FF={first_frame_image is not None},LF={last_frame_image is not None}"
        return await _submit_and_download(
            content, duration, resolution, ratio, poll_interval, poll_timeout, unique_id, log, label
        )


class Soze_MiniMaxH3VideoReference:
    """MiniMax (Hailuo) H3 video generation from reference images.

    Connect up to 9 reference images (image_1..image_9, individual slots take
    precedence) or an image_batch; each is sent with role=reference_image so the
    model uses them for subject/style guidance. Submits the task, polls to
    completion, and downloads the resulting MP4.

    Set MINIMAX_API_KEY in your environment (and optionally MINIMAX_BASE_URL).
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt (<= 7000 chars)."}),
            "duration": ("INT", {"default": 6, "min": 4, "max": 15, "step": 1, "tooltip": "Clip length in seconds (4-15)."}),
            "resolution": (MINIMAX_RESOLUTION_CHOICES, {"default": "2K"}),
            "ratio": (MINIMAX_RATIO_CHOICES, {"default": "adaptive", "tooltip": "'adaptive' matches the reference frames; or pick a fixed ratio."}),
        }
        optional = {
            "poll_interval": ("INT", {"default": DEFAULT_POLL_INTERVAL, "min": 1, "max": 60, "step": 1, "tooltip": "Seconds between status polls."}),
            "poll_timeout": ("INT", {"default": DEFAULT_POLL_TIMEOUT, "min": 30, "max": 3600, "step": 30, "tooltip": "Max seconds to wait before giving up."}),
            "image_batch": ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {MINIMAX_MAX_REFERENCE_IMAGES} frames). Ignored if any image_N slot is connected."}),
        }
        for i in range(1, MINIMAX_MAX_REFERENCE_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i} (role=reference_image). Overrides image_batch when connected."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "task_id", "status")
    FUNCTION = "generate"
    CATEGORY = "MiniMax/VideoGeneration"

    async def generate(self, prompt, duration, resolution, ratio,
                       poll_interval=DEFAULT_POLL_INTERVAL, poll_timeout=DEFAULT_POLL_TIMEOUT,
                       image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: MiniMax-H3 reference | {resolution} | {ratio} | {duration}s")

        if not prompt or not prompt.strip():
            headline = "MiniMax H3 reference skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", "", finalize_status(headline, log))

        ref_uris = _collect_reference_image_uris(slots, image_batch, MINIMAX_MAX_REFERENCE_IMAGES)
        if not ref_uris:
            headline = "MiniMax H3 reference skipped: no reference images connected."
            push_node_status(unique_id, headline, log)
            return (None, "", "", finalize_status(headline, log))

        content = [{"type": "text", "text": prompt.strip()[:7000]}]
        for uri in ref_uris:
            content.append({"type": "image_url", "image_url": {"url": uri}, "role": "reference_image"})

        label = f"reference_images={len(ref_uris)}"
        return await _submit_and_download(
            content, duration, resolution, ratio, poll_interval, poll_timeout, unique_id, log, label
        )
