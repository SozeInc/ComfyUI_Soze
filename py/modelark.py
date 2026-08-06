"""BytePlus ModelArk — Seedance 2.0 video generation nodes.

ModelArk's Seedance 2.0 is the same family of models as the FAL-routed
Seedance 2.0 nodes, but called directly via BytePlus's task API:

    POST   {base}/api/v3/contents/generations/tasks      -> {"id": "task_..."}
    GET    {base}/api/v3/contents/generations/tasks/{id} -> {"status": ..., "content": {"video_url": ...}}

`Authorization: Bearer $ARK_API_KEY` is required. The endpoint base defaults to
the global Asia-Pacific Southeast host; override via `ARK_BASE_URL` if you have
a different regional account.

Image / video / audio inputs are uploaded to FAL (via ImageUtils.upload_*) so
ModelArk can fetch them at a public URL — FAL is reused only as an image host,
the generation itself is entirely on ModelArk's side.
"""

import asyncio
import base64
import io
import logging
import os

import numpy as np
import requests
import torch
from comfy.utils import common_upscale
from comfy_api_nodes.util import download_url_to_video_output
from PIL import Image

from .fal_utils import ImageUtils
from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


MODELARK_DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com"
MODELARK_TASKS_PATH = "/api/v3/contents/generations/tasks"
MODELARK_MODEL_IDS = {
    "standard": "dreamina-seedance-2-0-260128",
    "fast": "dreamina-seedance-2-0-fast-260128",
}
MODELARK_TIER_CHOICES = ["standard", "fast"]

MODELARK_RATIO_CHOICES = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
MODELARK_RESOLUTION_CHOICES = ["480p", "720p", "1080p"]
MODELARK_DURATION_CHOICES = ["-1", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
MODELARK_DURATION_AUTO = "-1"  # ModelArk's documented auto sentinel.

# Match the FAL Seedance reference node's slot counts (9/3/3).
MODELARK_MAX_REF_IMAGES = 9
MODELARK_MAX_REF_VIDEOS = 3
MODELARK_MAX_REF_AUDIOS = 3

HTTP_TIMEOUT = (10, 120)
DEFAULT_POLL_INTERVAL = 5  # seconds between GET polls
DEFAULT_POLL_TIMEOUT = 600  # max seconds to wait for a task

# ModelArk caps `seed` at 2^32 (4,294,967,296) per the API error responses.
# Widget `max` is only a UI clamp — values coming through wires bypass it —
# so we re-clamp defensively at the payload assembly site.
MODELARK_MAX_SEED = 4294967296


def _clamp_seed(seed):
    """Map any incoming seed deterministically into the [1, MODELARK_MAX_SEED] range.

    Returns (clamped_int, was_wrapped). Returns (0, False) for None / non-numeric /
    0 / negative inputs, which the caller interprets as 'omit the field, let
    ModelArk pick'. Same input always maps to the same output (no RNG).
    """
    if seed is None:
        return 0, False
    try:
        s = int(seed)
    except (TypeError, ValueError):
        return 0, False
    if s <= 0:
        return 0, False
    clamped = s % (MODELARK_MAX_SEED + 1)
    if clamped == 0:
        # Don't collide with our "auto" sentinel — bump to MAX instead. This
        # still preserves "same input always maps to the same output".
        clamped = MODELARK_MAX_SEED
    return clamped, clamped != s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _modelark_base_url() -> str:
    return os.environ.get("ARK_BASE_URL", MODELARK_DEFAULT_BASE_URL).rstrip("/")


def _modelark_api_key() -> str:
    key = os.environ.get("ARK_API_KEY")
    if not key:
        raise EnvironmentError(
            "ARK_API_KEY environment variable not set. Get one from "
            "https://console.byteplus.com/ark and set it before running this node."
        )
    return key


def _build_config_tag(tier: str, resolution: str, duration: str, ratio: str, generate_audio: bool) -> str:
    speed_tag = "Fast" if tier == "fast" else "Std"
    res_tag = resolution.rstrip("p") if resolution else "Auto"
    dur_tag = "Auto" if not duration or str(duration) in (MODELARK_DURATION_AUTO, "auto") else f"{duration}s"
    ar_tag = ratio.replace(":", "") if ratio else "Auto"
    audio_tag = "Audio" if generate_audio else "NoAudio"
    return f"{speed_tag}-{res_tag}-{dur_tag}-{ar_tag}-{audio_tag}"


def _add_image_entry(content_list, url, role):
    """Append an image_url content entry with a role tag."""
    if url:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": url},
            "role": role,
        })


def _add_media_entry(content_list, url, role, kind):
    """Append a video_url or audio_url entry. kind is 'video_url' or 'audio_url'."""
    if url:
        content_list.append({
            "type": kind,
            kind: {"url": url},
            "role": role,
        })


def _upload_optional_image(tensor, label, log, unique_id, failures):
    """Upload one IMAGE frame via FAL. Logs the step. Returns URL or None."""
    if tensor is None:
        return None
    push_node_status(unique_id, f"Uploading {label}...", log)
    try:
        url = ImageUtils.upload_image(tensor[0:1])
    except Exception as e:
        logger.exception("ModelArk: %s upload raised", label)
        failures.append(f"{label}: {e!r}")
        return None
    if not url:
        failures.append(f"{label}: uploader returned None (check FAL config / earlier log)")
        return None
    return url


def _post_create_task(payload, log, unique_id):
    """Submit the create-task POST. Returns the parsed JSON or raises."""
    url = _modelark_base_url() + MODELARK_TASKS_PATH
    headers = {
        "Authorization": f"Bearer {_modelark_api_key()}",
        "Content-Type": "application/json",
    }
    push_node_status(unique_id, f"POST {url}", log)
    resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        # Surface the response body so users see ModelArk's actual error reason.
        body = ""
        try:
            body = resp.text
        except Exception:
            pass
        raise RuntimeError(
            f"ModelArk create-task failed: HTTP {resp.status_code} | body={body.strip()[:500]}"
        )
    return resp.json()


def _get_task(task_id):
    url = f"{_modelark_base_url()}{MODELARK_TASKS_PATH}/{task_id}"
    headers = {"Authorization": f"Bearer {_modelark_api_key()}"}
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    if resp.status_code >= 400:
        body = ""
        try:
            body = resp.text
        except Exception:
            pass
        raise RuntimeError(
            f"ModelArk get-task failed: HTTP {resp.status_code} | body={body.strip()[:500]}"
        )
    return resp.json()


def _extract_task_id(create_response):
    """The create response shape varies slightly across ModelArk versions —
    try a few likely fields."""
    if not isinstance(create_response, dict):
        return None
    for key in ("id", "task_id", "job_id"):
        if create_response.get(key):
            return create_response[key]
    # Sometimes nested under {"data": {"id": ...}}
    data = create_response.get("data")
    if isinstance(data, dict):
        for key in ("id", "task_id", "job_id"):
            if data.get(key):
                return data[key]
    return None


def _extract_last_frame_url(task_response):
    """Pull the last-frame image URL out of a succeeded task response.

    ModelArk's exact field name for the last frame isn't documented; this
    walks the likely shapes (last_frame_url string, last_frame.url object,
    image_url, images[0].url, frames[-1].url, etc.) so the node tolerates
    minor response-shape variation.
    """
    if not isinstance(task_response, dict):
        return None
    candidates = []
    content = task_response.get("content")
    if isinstance(content, dict):
        candidates.append(content)
    for outer in ("result", "data", "output"):
        node = task_response.get(outer)
        if isinstance(node, dict):
            candidates.append(node)
    for c in candidates:
        # Flat string fields
        for key in ("last_frame_url", "last_frame_image_url", "image_url"):
            val = c.get(key)
            if isinstance(val, str) and val:
                return val
        # Object with url
        for key in ("last_frame", "last_frame_image"):
            val = c.get(key)
            if isinstance(val, dict) and val.get("url"):
                return val["url"]
        # Arrays
        images = c.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict) and first.get("url"):
                return first["url"]
            if isinstance(first, str):
                return first
        frames = c.get("frames")
        if isinstance(frames, list) and frames:
            last = frames[-1]
            if isinstance(last, dict) and last.get("url"):
                return last["url"]
            if isinstance(last, str):
                return last
    return None


def _extract_video_url(task_response):
    """Pull the result video URL out of a succeeded task response."""
    if not isinstance(task_response, dict):
        return None
    content = task_response.get("content")
    if isinstance(content, dict):
        url = content.get("video_url") or content.get("url")
        if url:
            return url
        # video might be a nested object
        video = content.get("video")
        if isinstance(video, dict) and video.get("url"):
            return video["url"]
    # Older shapes: result/data/output
    for outer in ("result", "data", "output"):
        node = task_response.get(outer)
        if isinstance(node, dict):
            url = node.get("video_url") or node.get("url")
            if url:
                return url
    return None


async def _poll_task(task_id, log, unique_id, poll_interval, poll_timeout):
    """Poll a task until terminal or timeout. Returns the final task dict."""
    elapsed = 0
    last_status = None
    while True:
        task = _get_task(task_id)
        status = (task.get("status") or "").lower() if isinstance(task, dict) else ""
        if status != last_status:
            push_node_status(unique_id, f"Task {task_id} status: {status or '(unknown)'}", log)
            last_status = status
        if status in ("succeeded", "success", "completed"):
            return task
        if status in ("failed", "expired", "cancelled", "canceled"):
            error = task.get("error") if isinstance(task, dict) else None
            err_msg = ""
            if isinstance(error, dict):
                err_msg = f" | error.code={error.get('code')!r}, error.message={error.get('message')!r}"
            raise RuntimeError(f"ModelArk task ended with status '{status}'{err_msg}")
        if elapsed >= poll_timeout:
            raise TimeoutError(f"ModelArk task {task_id} did not finish within {poll_timeout}s (last status: {status or 'unknown'})")
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval


# ---------------------------------------------------------------------------
# Node — Seedance 2.0 / 2.0 Fast (single node, tier dropdown)
# ---------------------------------------------------------------------------


class Soze_ModelArkSeedance2:
    """BytePlus ModelArk — Seedance 2.0 video generation.

    Submits a content array (text + optional first/last frame + up to 5 reference
    images + optional reference video/audio), polls the task to completion, then
    downloads the resulting MP4. `tier` switches between the standard and fast
    model IDs.

    Set `ARK_API_KEY` (and optionally `ARK_BASE_URL`) in your environment.
    Image/video/audio inputs are uploaded to FAL's CDN so ModelArk can fetch
    them at a public URL — FAL is reused as image-hosting only.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "tier": (MODELARK_TIER_CHOICES, {"default": "standard", "tooltip": "'fast' uses the *-fast-* model id."}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "resolution": (MODELARK_RESOLUTION_CHOICES, {"default": "720p"}),
            "ratio": (MODELARK_RATIO_CHOICES, {"default": "16:9"}),
            "duration": (MODELARK_DURATION_CHOICES, {"default": MODELARK_DURATION_AUTO, "tooltip": "-1 = auto (ModelArk picks). 4-15 = explicit seconds."}),
            "generate_audio": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967296, "tooltip": "0 = let ModelArk choose. Max 4294967296 (2^32) per ModelArk API."}),
        }
        optional = {
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "return_last_frame": ("BOOLEAN", {"default": False, "tooltip": "If True, the task also returns the final frame in the response."}),
            "callback_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Optional webhook ModelArk will hit when the task completes."}),
            "first_frame_image": ("IMAGE", {"tooltip": "Optional first-frame anchor (role=first_frame). Use for image-to-video or FF/LF; leave empty for pure reference-to-video."}),
            "last_frame_image": ("IMAGE", {"tooltip": "Optional last-frame anchor (role=last_frame). Pair with first_frame_image for FF/LF."}),
            "poll_interval": ("INT", {"default": DEFAULT_POLL_INTERVAL, "min": 1, "max": 60, "step": 1, "tooltip": "Seconds between status polls."}),
            "poll_timeout": ("INT", {"default": DEFAULT_POLL_TIMEOUT, "min": 30, "max": 3600, "step": 30, "tooltip": "Max seconds to wait for the task before giving up."}),
        }
        for i in range(1, MODELARK_MAX_REF_IMAGES + 1):
            optional[f"reference_image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i} (role=reference_image). Cite as @Image{i} in the prompt."})
        for i in range(1, MODELARK_MAX_REF_VIDEOS + 1):
            optional[f"reference_video_{i}"] = ("VIDEO", {"tooltip": f"Reference video slot {i} (role=reference_video). Cite as @Video{i} in the prompt."})
        for i in range(1, MODELARK_MAX_REF_AUDIOS + 1):
            optional[f"reference_audio_{i}"] = ("AUDIO", {"tooltip": f"Reference audio slot {i} (role=reference_audio). Cite as @Audio{i} in the prompt."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "last_frame_image", "last_frame_url", "task_id", "config", "status")
    FUNCTION = "generate"
    CATEGORY = "ModelArk/VideoGeneration"

    async def generate(self, tier, prompt, resolution, ratio, duration, generate_audio, seed,
                       negative_prompt="", return_last_frame=False, callback_url="",
                       first_frame_image=None, last_frame_image=None,
                       poll_interval=DEFAULT_POLL_INTERVAL, poll_timeout=DEFAULT_POLL_TIMEOUT,
                       unique_id=None, **slots):
        config = _build_config_tag(tier, resolution, duration, ratio, generate_audio)
        log = EventLog()
        log.add(f"Config: modelark-seedance-2.0 {tier} | {config}")

        # Placeholders for the last-frame outputs. Stay blank/empty unless we
        # actually fetch a last frame after a successful generation. Named
        # _out to avoid shadowing the `last_frame_image` *input* parameter.
        last_frame_image_out = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        last_frame_url_out = ""

        # Quick scan of connected reference slots so we can short-circuit empty submissions.
        any_ref_connected = any(
            slots.get(f"reference_image_{i}") is not None for i in range(1, MODELARK_MAX_REF_IMAGES + 1)
        ) or any(
            slots.get(f"reference_video_{i}") is not None for i in range(1, MODELARK_MAX_REF_VIDEOS + 1)
        ) or any(
            slots.get(f"reference_audio_{i}") is not None for i in range(1, MODELARK_MAX_REF_AUDIOS + 1)
        )
        if _is_blank(prompt) and first_frame_image is None and last_frame_image is None and not any_ref_connected:
            # An entirely empty submission. Don't waste an API call.
            headline = "ModelArk Seedance 2.0 skipped: prompt is blank and no image/video/audio reference connected."
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, "", config, finalize_status(headline, log))

        # ------------------------------------------------------------------
        # 1) Upload any connected media to FAL's CDN to get public URLs.
        # ------------------------------------------------------------------
        upload_failures = []
        first_url = _upload_optional_image(first_frame_image, "first_frame_image", log, unique_id, upload_failures)
        last_url = _upload_optional_image(last_frame_image, "last_frame_image", log, unique_id, upload_failures)

        ref_image_urls = []
        for i in range(1, MODELARK_MAX_REF_IMAGES + 1):
            tensor = slots.get(f"reference_image_{i}")
            if tensor is None:
                continue
            url = _upload_optional_image(tensor, f"reference_image_{i}", log, unique_id, upload_failures)
            if url:
                ref_image_urls.append(url)

        ref_video_urls = []
        for i in range(1, MODELARK_MAX_REF_VIDEOS + 1):
            video_obj = slots.get(f"reference_video_{i}")
            if video_obj is None:
                continue
            label = f"reference_video_{i}"
            push_node_status(unique_id, f"Uploading {label}...", log)
            try:
                url = ImageUtils.upload_video(video_obj)
            except Exception as e:
                logger.exception("ModelArk: %s upload raised", label)
                upload_failures.append(f"{label}: {e!r}")
                continue
            if url:
                ref_video_urls.append(url)
            else:
                upload_failures.append(f"{label}: uploader returned None")

        ref_audio_urls = []
        for i in range(1, MODELARK_MAX_REF_AUDIOS + 1):
            audio_obj = slots.get(f"reference_audio_{i}")
            if audio_obj is None:
                continue
            label = f"reference_audio_{i}"
            push_node_status(unique_id, f"Uploading {label}...", log)
            try:
                url = ImageUtils.upload_audio(audio_obj)
            except Exception as e:
                logger.exception("ModelArk: %s upload raised", label)
                upload_failures.append(f"{label}: {e!r}")
                continue
            if url:
                ref_audio_urls.append(url)
            else:
                upload_failures.append(f"{label}: uploader returned None")

        # ------------------------------------------------------------------
        # 2) Build the content array.
        # ------------------------------------------------------------------
        content = []
        if not _is_blank(prompt):
            content.append({"type": "text", "text": prompt.strip()})
        _add_image_entry(content, first_url, "first_frame")
        _add_image_entry(content, last_url, "last_frame")
        for u in ref_image_urls:
            _add_image_entry(content, u, "reference_image")
        for u in ref_video_urls:
            _add_media_entry(content, u, "reference_video", "video_url")
        for u in ref_audio_urls:
            _add_media_entry(content, u, "reference_audio", "audio_url")

        if not content:
            headline = "ModelArk Seedance 2.0 skipped: content array is empty after uploads."
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, "", config, finalize_status(headline, log))

        # ------------------------------------------------------------------
        # 3) Build payload + submit.
        # ------------------------------------------------------------------
        payload = {
            "model": MODELARK_MODEL_IDS[tier],
            "content": content,
            "ratio": ratio,
            "resolution": resolution,
            "generate_audio": bool(generate_audio),
        }
        # `duration` accepts -1 (auto) or 4..15 (seconds).
        if duration:
            payload["duration"] = int(duration)
        clamped_seed, seed_wrapped = _clamp_seed(seed)
        if clamped_seed > 0:
            payload["seed"] = clamped_seed
            if seed_wrapped:
                push_node_status(unique_id, f"Seed {seed} wrapped to {clamped_seed} (API cap {MODELARK_MAX_SEED}).", log)
        if not _is_blank(negative_prompt):
            payload["negative_prompt"] = negative_prompt.strip()
        if return_last_frame:
            payload["return_last_frame"] = True
        if not _is_blank(callback_url):
            payload["callback_url"] = callback_url.strip()

        try:
            create_resp = _post_create_task(payload, log, unique_id)
        except Exception as e:
            logger.exception("ModelArk create-task failed")
            headline = f"ERROR creating task: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, "", config, finalize_status(headline, log))

        task_id = _extract_task_id(create_resp)
        if not task_id:
            headline = f"ERROR: create-task returned no task id. Response: {create_resp!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, "", config, finalize_status(headline, log))

        push_node_status(unique_id, f"Task created: {task_id}. Polling every {poll_interval}s (timeout {poll_timeout}s)...", log)

        # ------------------------------------------------------------------
        # 4) Poll.
        # ------------------------------------------------------------------
        try:
            final_task = await _poll_task(task_id, log, unique_id, int(poll_interval), int(poll_timeout))
        except Exception as e:
            logger.exception("ModelArk polling failed")
            headline = f"ERROR polling task {task_id}: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, task_id, config, finalize_status(headline, log))

        video_url = _extract_video_url(final_task)
        if not video_url:
            headline = f"ERROR: succeeded task {task_id} returned no video_url. Response: {final_task!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", last_frame_image_out, last_frame_url_out, task_id, config, finalize_status(headline, log))

        # ------------------------------------------------------------------
        # 5) Download the video.
        # ------------------------------------------------------------------
        push_node_status(unique_id, "Downloading video...", log)
        try:
            video = await download_url_to_video_output(video_url)
        except Exception as e:
            logger.exception("ModelArk video download failed (url=%s)", video_url)
            headline = f"ERROR downloading video: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, video_url, last_frame_image_out, last_frame_url_out, task_id, config, finalize_status(headline, log))

        if upload_failures:
            push_node_status(unique_id, f"Note: {len(upload_failures)} media upload(s) failed: " + "; ".join(upload_failures), log)

        # If the caller asked for the last frame, pull its URL out of the
        # final task response and download it into an IMAGE tensor.
        if return_last_frame:
            extracted_url = _extract_last_frame_url(final_task)
            if extracted_url:
                last_frame_url_out = extracted_url
                push_node_status(unique_id, "Downloading last frame...", log)
                try:
                    last_frame_image_out = _download_image_url_to_tensor(extracted_url)
                except Exception as e:
                    logger.exception("ModelArk: last-frame download failed (url=%s)", extracted_url)
                    push_node_status(unique_id, f"Last-frame download failed (keeping blank image): {e!r}", log)
            else:
                push_node_status(unique_id, "Note: return_last_frame=True but no last-frame URL found in task response.", log)

        headline = f"OK (modelark-seedance-2.0/{tier}): task={task_id}, url={video_url}"
        push_node_status(unique_id, headline, log)
        return (video, video_url, last_frame_image_out, last_frame_url_out, task_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# ModelArk Seedream Images (4.0 / 4.5 / 5.0 Lite)
# ---------------------------------------------------------------------------


MODELARK_IMAGES_PATH = "/api/v3/images/generations"

# Default model IDs per version. Only the 4.0 id is publicly confirmed
# (`seedream-4-0-250828`); the 4.5 / 5.0-Lite dated suffixes are not published
# outside the BytePlus console, so we pick reasonable defaults the user can
# override with `model_id_override`.
MODELARK_SEEDREAM_DEFAULT_IDS = {
    "seedream-4.0": "seedream-4-0-250828",
    "seedream-4.5": "seedream-4-5-250901",
    "seedream-5.0-lite": "seedream-5-0-lite-250901",
    "seedream-5.0-pro": "seedream-5-0-pro-250901",
}
MODELARK_SEEDREAM_VERSION_CHOICES = list(MODELARK_SEEDREAM_DEFAULT_IDS.keys())

MODELARK_SEEDREAM_SIZE_CHOICES = ["2K", "4K", "1024x1024", "1536x1536", "2048x2048", "3072x3072"]
MODELARK_SEEDREAM_RESPONSE_FORMAT_CHOICES = ["url", "b64_json"]
MODELARK_SEEDREAM_MAX_IMAGES = 10  # up to 10 reference images for blending/editing


def _download_image_url_to_tensor(url):
    """Download a URL (or decode a data: URI) into a [1, H, W, 3] float32 tensor."""
    if url.startswith("data:"):
        _, _, b64 = url.partition(",")
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw))
    else:
        resp = requests.get(url, timeout=HTTP_TIMEOUT, stream=True)
        resp.raise_for_status()
        pil = Image.open(io.BytesIO(resp.content))
    pil = pil.convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _decode_b64_image_to_tensor(b64_string):
    """Decode a bare base64 string (no data: prefix) into a [1, H, W, 3] tensor."""
    raw = base64.b64decode(b64_string)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _batch_align_image_tensors(tensors):
    """Cat a list of [1,H,W,3] tensors, resizing later ones to match the first."""
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    base = tensors[0]
    base_h, base_w = base.shape[1], base.shape[2]
    aligned = [base]
    for t in tensors[1:]:
        if t.shape[1] != base_h or t.shape[2] != base_w:
            t = common_upscale(t.movedim(-1, 1), base_w, base_h, "bilinear", "center").movedim(1, -1)
        aligned.append(t)
    return torch.cat(aligned, dim=0)


def _collect_seedream_image_slots(slots, image_batch, log, unique_id, max_images):
    """Collect connected image_N slots OR fan out an image_batch; upload via FAL.

    Returns (urls, failures). image_N slots take precedence over image_batch
    (same convention as the FAL Seedream / GPT-Image-2 nodes).
    """
    slot_pairs = []
    for i in range(1, max_images + 1):
        tensor = slots.get(f"image_{i}")
        if tensor is not None:
            slot_pairs.append((f"image_{i}", tensor[0:1]))

    if slot_pairs:
        if image_batch is not None:
            push_node_status(unique_id, "Note: image_batch is ignored because individual image_N slot(s) are connected.", log)
    elif image_batch is not None and image_batch.shape[0] > 0:
        n = min(image_batch.shape[0], max_images)
        if image_batch.shape[0] > max_images:
            push_node_status(unique_id, f"image_batch has {image_batch.shape[0]} frames; using first {max_images} (API max).", log)
        for i in range(n):
            slot_pairs.append((f"batch[{i}]", image_batch[i:i+1]))

    urls = []
    failures = []
    if not slot_pairs:
        return urls, failures

    push_node_status(unique_id, f"Uploading {len(slot_pairs)} reference image(s) to FAL CDN for ModelArk...", log)
    for label, tensor in slot_pairs:
        try:
            url = ImageUtils.upload_image(tensor)
        except Exception as e:
            logger.exception("ModelArk Seedream: %s upload raised", label)
            failures.append(f"{label}: {e!r}")
            continue
        if url:
            urls.append(url)
        else:
            failures.append(f"{label}: uploader returned None (check FAL config)")
    return urls, failures


def _resolve_seedream_model_id(version, override):
    """Pick the right model id. Override wins if non-blank."""
    if not _is_blank(override):
        return override.strip()
    return MODELARK_SEEDREAM_DEFAULT_IDS.get(version, version)


class Soze_ModelArkSeedreamImages:
    """BytePlus ModelArk — Seedream image generation / editing.

    Single node covering Seedream 4.0, 4.5, 5.0 Lite, and 5.0 Pro. Pick a
    `model_version` (preset → default dated model id) or set `model_id_override`
    to paste the exact dated id from your BytePlus console — useful when ModelArk
    publishes a new dated revision your preset doesn't know about.

    Endpoint: POST {ARK_BASE_URL}/api/v3/images/generations
    Auth:     Authorization: Bearer $ARK_API_KEY

    Up to 10 reference images can be sent for editing/blending. Connect them
    via image_1..image_10 (individual slots) or image_batch (auto-fanned); the
    individual slots override the batch. They are uploaded to FAL's CDN so
    ModelArk can fetch them at a public URL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "model_version": (MODELARK_SEEDREAM_VERSION_CHOICES, {"default": "seedream-5.0-lite"}),
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "size": (MODELARK_SEEDREAM_SIZE_CHOICES, {"default": "2K", "tooltip": "Preset or use model_id_override + a custom size_override."}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1, "tooltip": "Maps to `n` (and sequential_image_generation_options.max_images)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 4294967296, "tooltip": "0 = let ModelArk choose. Max 4294967296 (2^32) per ModelArk API."}),
            "watermark": ("BOOLEAN", {"default": False}),
            "response_format": (MODELARK_SEEDREAM_RESPONSE_FORMAT_CHOICES, {"default": "url"}),
        }
        optional = {
            "model_id_override": ("STRING", {"default": "", "multiline": False, "tooltip": "Paste a specific dated model id (e.g. seedream-4-0-250828). Overrides model_version."}),
            "size_override": ("STRING", {"default": "", "multiline": False, "tooltip": "If set, replaces `size`. e.g. '2048x1152' for a custom aspect."}),
            "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "0 = use server default (typically ~7.5)."}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "sequential_image_generation": (["auto", "disabled"], {"default": "disabled", "tooltip": "auto = let model emit multiple images sequentially."}),
            "image_batch": ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {MODELARK_SEEDREAM_MAX_IMAGES} frames). Ignored if any image_N slot is connected."}),
        }
        for i in range(1, MODELARK_SEEDREAM_MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i}. Overrides image_batch when connected."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "model_id", "status")
    FUNCTION = "generate"
    CATEGORY = "ModelArk/ImageGeneration"

    def generate(self, model_version, prompt, size, num_images, seed, watermark, response_format,
                 model_id_override="", size_override="", guidance_scale=0.0, negative_prompt="",
                 sequential_image_generation="disabled", image_batch=None, unique_id=None, **slots):
        log = EventLog()
        model_id = _resolve_seedream_model_id(model_version, model_id_override)
        effective_size = size_override.strip() if not _is_blank(size_override) else size
        log.add(f"Config: modelark-seedream {model_version} (id={model_id}) | size={effective_size} | n={num_images} | watermark={watermark} | seq={sequential_image_generation}")

        if _is_blank(prompt):
            headline = "ModelArk Seedream skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, model_id, finalize_status(headline, log))

        # 1) Upload any connected reference images to FAL for public URLs.
        ref_urls, upload_failures = _collect_seedream_image_slots(
            slots, image_batch, log, unique_id, MODELARK_SEEDREAM_MAX_IMAGES
        )

        # 2) Build payload. The `image` field accepts a single URL or a list;
        # we always send a list when references are present.
        payload = {
            "model": model_id,
            "prompt": prompt.strip(),
            "size": effective_size,
            "n": int(num_images),
            "watermark": bool(watermark),
            "response_format": response_format,
        }
        clamped_seed, seed_wrapped = _clamp_seed(seed)
        if clamped_seed > 0:
            payload["seed"] = clamped_seed
            if seed_wrapped:
                push_node_status(unique_id, f"Seed {seed} wrapped to {clamped_seed} (API cap {MODELARK_MAX_SEED}).", log)
        if isinstance(guidance_scale, (int, float)) and guidance_scale > 0:
            payload["guidance_scale"] = float(guidance_scale)
        if not _is_blank(negative_prompt):
            payload["negative_prompt"] = negative_prompt.strip()
        if sequential_image_generation and sequential_image_generation != "disabled":
            payload["sequential_image_generation"] = sequential_image_generation
            payload["sequential_image_generation_options"] = {"max_images": int(num_images)}
        if ref_urls:
            payload["image"] = ref_urls

        # 3) Submit.
        url = _modelark_base_url() + MODELARK_IMAGES_PATH
        headers = {
            "Authorization": f"Bearer {_modelark_api_key()}",
            "Content-Type": "application/json",
        }
        push_node_status(unique_id, f"POST {url}", log)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(10, 300))
        except Exception as e:
            logger.exception("ModelArk Seedream POST failed")
            headline = f"ERROR submitting: {e!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, model_id, finalize_status(headline, log))

        if resp.status_code >= 400:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            headline = f"ERROR: HTTP {resp.status_code} | body={body.strip()[:500]}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, model_id, finalize_status(headline, log))

        try:
            result = resp.json()
        except Exception as e:
            headline = f"ERROR parsing response JSON: {e!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, model_id, finalize_status(headline, log))

        # 4) Parse the data array. Each entry has either `url` or `b64_json`.
        data_entries = result.get("data") or []
        if not data_entries:
            headline = f"ERROR: response has no `data` images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, model_id, finalize_status(headline, log))

        push_node_status(unique_id, f"Decoding {len(data_entries)} result image(s)...", log)
        tensors = []
        emitted_urls = []
        for i, entry in enumerate(data_entries):
            if not isinstance(entry, dict):
                continue
            try:
                if entry.get("url"):
                    emitted_urls.append(entry["url"])
                    tensors.append(_download_image_url_to_tensor(entry["url"]))
                elif entry.get("b64_json"):
                    # No URL in b64 mode — keep an empty marker for the URL list.
                    emitted_urls.append("")
                    tensors.append(_decode_b64_image_to_tensor(entry["b64_json"]))
                else:
                    push_node_status(unique_id, f"Skip result {i+1}: no url/b64_json in entry {entry!r}", log)
            except Exception as e:
                logger.exception("ModelArk Seedream: result %d decode failed", i)
                push_node_status(unique_id, f"Skip result {i+1}: {e!r}", log)

        if not tensors:
            headline = "ERROR: every result image failed to decode."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(emitted_urls), 0, model_id, finalize_status(headline, log))

        image_batch_out = _batch_align_image_tensors(tensors)
        if upload_failures:
            push_node_status(unique_id, f"Note: {len(upload_failures)} reference upload(s) failed: " + "; ".join(upload_failures), log)

        urls_str = "\n".join(emitted_urls)
        headline = f"OK ({model_id}): {len(tensors)}/{len(data_entries)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), model_id, finalize_status(headline, log))
