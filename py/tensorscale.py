"""TensorScale — synchronous world-model inference nodes.

TensorScale's v2 API is *synchronous*: every endpoint is a single POST that
blocks until the generation finishes and then streams the finished media back
as the response body. There is no task id and no polling — unlike the ModelArk
/ MiniMax / FAL nodes in this pack, which all submit-then-poll.

    POST {base}/v2/MiniMax-H3/fl2va                 -> video/mp4
    POST {base}/v2/MiniMax-H3/ref2va                -> video/mp4
    POST {base}/v2/ltx2-5/fast                      -> video/mp4
    POST {base}/v2/ltx-2.3/fast                     -> video/mp4
    POST {base}/v2/cosmos3-nano/t2v                 -> video/mp4
    POST {base}/v2/cosmos3-nano/v2v                 -> video/mp4
    POST {base}/v2/hunyuan-image-3-instruct/fast    -> image/jpeg
    POST {base}/v2/sensenova-u1.5/t2i               -> image/png

Auth:  Authorization: Bearer $TENSORSCALE_API_KEY
Base:  https://api.tensorscale.io (override with TENSORSCALE_BASE_URL)

TensorScale keys are *model-scoped* — a key authorized for `ltx2-5-fast` will
be rejected by the MiniMax endpoints. Every node therefore exposes an
`api_key_env` widget naming a per-model environment variable, and falls back to
`TENSORSCALE_API_KEY` when that variable is unset. Set one shared key or a key
per model, whichever matches your account.

Reference media can be sent two ways:
  * inline as a base64 `data:` URI, built from the connected IMAGE/VIDEO/AUDIO
    socket. Simple, but the whole JSON body is capped at 10 MiB.
  * as a public HTTPS URL, either typed into the `*_url` widgets or produced by
    flipping `use_fal_upload` on (which pushes the socket's media to FAL's CDN
    and sends the resulting URL). Use this for anything video-sized.
LTX-2.5 Fast is the exception: it accepts inline data URIs only.

Docs: https://www.tensorscale.io/docs.html
"""

import asyncio
import base64
import io
import json
import logging
import os
import uuid

import folder_paths  # type: ignore
import numpy as np
import requests
import torch
from comfy_api.input_impl import VideoFromFile
from PIL import Image

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


TENSORSCALE_DEFAULT_BASE_URL = "https://api.tensorscale.io"

# Connect fast, then wait a long time: these calls hold the socket open for the
# entire generation. TensorScale's own examples use read timeouts of 600-7200s.
HTTP_CONNECT_TIMEOUT = 15
DEFAULT_READ_TIMEOUT = 1800
MAX_READ_TIMEOUT = 7200

# TensorScale caps the whole JSON body at 10 MiB (documented for MiniMax H3 and
# for LTX-2.5's decoded image). We check before sending so an oversized inline
# data URI fails locally with an actionable message instead of a 413.
MAX_BODY_BYTES = 10 * 1024 * 1024

CONTENT_TYPE_SUFFIXES = {
    "video/mp4": ".mp4",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

BLANK_IMAGE_SHAPE = (1, 64, 64, 3)

# Paths
PATH_MINIMAX_FL2VA = "/v2/MiniMax-H3/fl2va"
PATH_MINIMAX_REF2VA = "/v2/MiniMax-H3/ref2va"
PATH_LTX25_FAST = "/v2/ltx2-5/fast"
PATH_LTX23_FAST = "/v2/ltx-2.3/fast"
PATH_COSMOS_T2V = "/v2/cosmos3-nano/t2v"
PATH_COSMOS_V2V = "/v2/cosmos3-nano/v2v"
PATH_HUNYUAN_IMAGE3 = "/v2/hunyuan-image-3-instruct/fast"
PATH_SENSENOVA_T2I = "/v2/sensenova-u1.5/t2i"

# Choice lists straight from the schema tables.
MINIMAX_RATIO_CHOICES = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "4:1", "1:4"]
LTX25_RESOLUTION_CHOICES = ["1920x1088", "1088x1920", "1280x768", "768x1280", "576x1024"]
LTX25_NUM_FRAMES_CHOICES = ["121", "241"]  # 5s / 10s at the fixed 24 fps
LTX25_FRAME_RATE = 24
LTX23_RESOLUTION_CHOICES = ["1920x1088", "1088x1920", "1280x768", "768x1280"]
LTX23_DURATION_CHOICES = ["5", "8"]
COSMOS_RESOLUTION_CHOICES = ["256", "480", "704", "720", "768"]
COSMOS_RATIO_CHOICES = ["16,9", "9,16", "1,1", "4,3", "3,4"]
HUNYUAN_BOT_TASK_CHOICES = ["image", "recaption", "think", "think_recaption"]
IMAGE_ENCODE_CHOICES = ["PNG", "JPEG"]

# Slot counts match the MiniMax/ModelArk reference nodes elsewhere in this pack.
MINIMAX_MAX_REF_IMAGES = 9
MINIMAX_MAX_REF_VIDEOS = 3
MINIMAX_MAX_REF_AUDIOS = 3

# Default per-model env var names (each falls back to TENSORSCALE_API_KEY).
ENV_MINIMAX = "TENSORSCALE_API_KEY_MINIMAX_H3"
ENV_LTX25 = "TENSORSCALE_API_KEY_LTX2_5_FAST"
ENV_LTX23 = "TENSORSCALE_API_KEY_LTX_2_3_FAST"
ENV_COSMOS = "TENSORSCALE_API_KEY_COSMOS3_NANO"
ENV_HUNYUAN = "TENSORSCALE_API_KEY_HUNYUAN_IMAGE_3"
ENV_SENSENOVA = "TENSORSCALE_API_KEY_SENSENOVA_U1_5"


# ---------------------------------------------------------------------------
# Config / auth
# ---------------------------------------------------------------------------


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _base_url() -> str:
    return os.environ.get("TENSORSCALE_BASE_URL", TENSORSCALE_DEFAULT_BASE_URL).rstrip("/")


def _api_key(api_key_env: str = "") -> str:
    """Resolve the bearer token.

    Tries each comma-separated name in `api_key_env` first, then the shared
    TENSORSCALE_API_KEY. TensorScale keys are model-scoped, so a per-model
    variable is the normal setup once you have more than one model enabled.
    """
    names = [n.strip() for n in (api_key_env or "").split(",") if n.strip()]
    if "TENSORSCALE_API_KEY" not in names:
        names.append("TENSORSCALE_API_KEY")
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "No TensorScale API key found. Set " + " or ".join(names) +
        " in your environment. Keys are model-scoped — the key must be authorized "
        "for the model this node calls."
    )


def _blank_image():
    return torch.zeros(BLANK_IMAGE_SHAPE, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Inline media encoding
# ---------------------------------------------------------------------------


def _image_to_data_uri(image_tensor, fmt: str = "PNG"):
    """First frame of a ComfyUI IMAGE tensor -> `data:image/...;base64,...`."""
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
    fmt = (fmt or "PNG").upper()
    pil.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _video_to_path(video):
    """Materialize a ComfyUI VIDEO to a readable file path.

    Returns (path, is_temp). Mirrors the probe order in fal_utils.ImageUtils so
    a VideoFromFile is reused in place rather than re-encoded.
    """
    if video is None:
        return None, False
    for attr in ("_VideoFromFile__file", "file", "path", "_path"):
        candidate = getattr(video, attr, None)
        if isinstance(candidate, str) and os.path.isfile(candidate):
            return candidate, False
    if not hasattr(video, "save_to"):
        raise ValueError(f"Unsupported VIDEO object (no save_to): {type(video)!r}")
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"soze_ts_src_{uuid.uuid4().hex[:12]}.mp4")
    video.save_to(temp_path)
    return temp_path, True


def _video_to_data_uri(video):
    """ComfyUI VIDEO -> `data:video/mp4;base64,...`. Bodies are capped at 10 MiB,
    so this only works for short clips; prefer a URL or use_fal_upload."""
    path, is_temp = _video_to_path(video)
    if not path:
        return None
    try:
        raw = open(path, "rb").read()
    finally:
        if is_temp:
            try:
                os.unlink(path)
            except OSError:
                pass
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")


def _audio_to_data_uri(audio):
    """ComfyUI AUDIO dict -> `data:audio/wav;base64,...`."""
    if audio is None:
        return None
    try:
        import torchaudio  # local import: heavy, and only needed for audio refs
    except ImportError as e:
        raise RuntimeError(f"torchaudio is required to inline AUDIO inputs: {e}")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3:  # Comfy AUDIO is [B, C, S]; torchaudio wants [C, S]
        waveform = waveform[0]

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"soze_ts_src_{uuid.uuid4().hex[:12]}.wav")
    try:
        torchaudio.save(temp_path, waveform.cpu(), sample_rate)
        raw = open(temp_path, "rb").read()
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")


def _fal_upload(kind, media):
    """Push a socket's media to FAL's CDN and return the public URL.

    FAL is used purely as a file host here — generation happens on TensorScale.
    Imported lazily so this module still loads when fal_client isn't installed.
    """
    from .fal_utils import ImageUtils
    if kind == "image":
        return ImageUtils.upload_image(media[0:1])
    if kind == "video":
        return ImageUtils.upload_video(media)
    if kind == "audio":
        return ImageUtils.upload_audio(media)
    raise ValueError(f"Unknown media kind {kind!r}")


def _media_reference(kind, media, use_fal_upload, image_format="PNG"):
    """Turn a connected socket into ('uri'|'data', value) for the request body."""
    if media is None:
        return None, None
    if use_fal_upload:
        url = _fal_upload(kind, media)
        if not url:
            raise RuntimeError(
                f"FAL upload returned no URL for the {kind} input. Check FAL_KEY / fal_client setup, "
                "or turn use_fal_upload off to inline the media instead."
            )
        return "uri", url
    if kind == "image":
        return "data", _image_to_data_uri(media, image_format)
    if kind == "video":
        return "data", _video_to_data_uri(media)
    if kind == "audio":
        return "data", _audio_to_data_uri(media)
    raise ValueError(f"Unknown media kind {kind!r}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _format_api_error(status_code, text, request_id):
    """TensorScale returns JSON {code, message} on failure — surface both."""
    detail = (text or "").strip()
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            code = parsed.get("code") or parsed.get("error") or ""
            message = parsed.get("message") or parsed.get("detail") or ""
            if code or message:
                detail = f"{code}: {message}".strip(": ")
    except (ValueError, TypeError):
        pass
    rid = f" | request_id={request_id}" if request_id else ""
    return f"TensorScale HTTP {status_code}{rid} | {detail[:800]}"


def _request_binary(path, payload, accept, api_key, read_timeout, label):
    """POST JSON, stream the binary response into Comfy's temp dir.

    Blocking on purpose — callers run this under asyncio.to_thread so a
    30-minute generation doesn't stall the ComfyUI event loop.

    Returns (file_path, request_id, byte_count).
    """
    url = _base_url() + path
    body = json.dumps(payload).encode("utf-8")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(
            f"Request body is {len(body) / 1048576:.1f} MiB, over TensorScale's 10 MiB limit. "
            "Send media as a public HTTPS URL (or enable use_fal_upload) instead of inlining it."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": accept,
    }
    resp = requests.post(
        url, headers=headers, data=body,
        timeout=(HTTP_CONNECT_TIMEOUT, read_timeout), stream=True,
    )
    try:
        request_id = (
            resp.headers.get("X-Tensorscale-Request-Id")
            or resp.headers.get("X-Request-Id")
            or ""
        )
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()

        # Errors come back as JSON regardless of the Accept header.
        if resp.status_code >= 400 or content_type.startswith("application/json"):
            raise RuntimeError(_format_api_error(resp.status_code, resp.text, request_id))

        suffix = CONTENT_TYPE_SUFFIXES.get(content_type) or CONTENT_TYPE_SUFFIXES.get(accept, ".bin")
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        dest = os.path.join(temp_dir, f"soze_ts_{label}_{uuid.uuid4().hex[:12]}{suffix}")

        total = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    finally:
        resp.close()

    if total == 0:
        try:
            os.unlink(dest)
        except OSError:
            pass
        raise RuntimeError(
            f"TensorScale returned an empty body (HTTP {resp.status_code}, request_id={request_id or 'n/a'})."
        )
    return dest, request_id, total


async def _post_binary(path, payload, accept, api_key_env, read_timeout, label, log, unique_id):
    """Async wrapper around _request_binary with status-line reporting."""
    api_key = _api_key(api_key_env)
    push_node_status(unique_id, f"POST {_base_url()}{path} (sync, up to {read_timeout}s)...", log)
    result = await asyncio.to_thread(
        _request_binary, path, payload, accept, api_key, int(read_timeout), label
    )
    dest, request_id, total = result
    push_node_status(
        unique_id,
        f"Received {total / 1048576:.2f} MiB -> {os.path.basename(dest)}"
        + (f" (request_id={request_id})" if request_id else ""),
        log,
    )
    return dest, request_id, total


def _file_to_image_tensor(path):
    """Decode a downloaded image file into a [1, H, W, 3] float32 tensor."""
    pil = Image.open(path).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


# ---------------------------------------------------------------------------
# Shared widget fragments
# ---------------------------------------------------------------------------


def _api_widgets(default_env, default_timeout=DEFAULT_READ_TIMEOUT):
    return {
        "api_key_env": ("STRING", {
            "default": default_env,
            "multiline": False,
            "tooltip": "Environment variable holding the model-scoped key. "
                       "Falls back to TENSORSCALE_API_KEY when unset.",
        }),
        "timeout": ("INT", {
            "default": default_timeout, "min": 60, "max": MAX_READ_TIMEOUT, "step": 60,
            "tooltip": "Read timeout in seconds. These endpoints are synchronous — "
                       "the socket stays open for the whole generation.",
        }),
    }


def _video_returns():
    return ("VIDEO", "STRING", "STRING", "STRING", "STRING"), \
           ("video", "video_path", "request_id", "config", "status")


def _image_returns():
    return ("IMAGE", "STRING", "STRING", "STRING", "STRING"), \
           ("image", "image_path", "request_id", "config", "status")


def _video_error(headline, config, log, unique_id, request_id=""):
    push_node_status(unique_id, headline, log)
    return (None, "", request_id, config, finalize_status(headline, log))


def _image_error(headline, config, log, unique_id, request_id=""):
    push_node_status(unique_id, headline, log)
    return (_blank_image(), "", request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# MiniMax H3 — FL2VA (text / first-last frame -> video + audio)
# ---------------------------------------------------------------------------


class Soze_TensorScaleMiniMaxH3:
    """TensorScale — MiniMax H3 text-or-frame to video+audio.

    Endpoint: POST /v2/MiniMax-H3/fl2va

    Leave both frame inputs empty for pure text-to-audio-video; connect
    first_image (and optionally last_image) to condition the endpoints of the
    clip. The service snaps `duration_seconds` to H3's native 24 fps / 17n+5
    frame boundary, and derives the canvas from `aspect_ratio` — 16:9 becomes
    1344x768, 1:1 becomes 768x768, 9:16 becomes 768x1344, and so on.

    The first supplied frame is stretched onto the canvas; when both are given
    the last image is cover-cropped onto it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "duration_seconds": ("FLOAT", {
                "default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Target length. Aligned server-side to the 24 fps / 17n+5 frame grid.",
            }),
            "aspect_ratio": (MINIMAX_RATIO_CHOICES, {"default": "16:9"}),
            "seed": ("INT", {"default": 1101, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "first_image": ("IMAGE", {"tooltip": "Optional first-frame anchor."}),
            "last_image": ("IMAGE", {"tooltip": "Optional last-frame anchor. Pair with first_image for FF/LF."}),
            "first_image_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS URL. Overrides the first_image socket."}),
            "last_image_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS URL. Overrides the last_image socket."}),
            "aspect_ratio_override": ("STRING", {"default": "", "multiline": False, "tooltip": "Any W:H from 1:4 to 4:1, e.g. '2:1'. Replaces the dropdown when set."}),
            "num_inference_steps": ("INT", {"default": 50, "min": 2, "max": 200, "tooltip": "Scheduler points. The native profile uses 50."}),
            "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Video scheduler flow shift. The service profile requires 12.0."}),
            "audio_flow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 50.0, "step": 0.5, "tooltip": "Audio scheduler flow shift. The service profile requires 3.0."}),
            "use_fal_upload": ("BOOLEAN", {"default": False, "tooltip": "Upload connected images to FAL's CDN and send URLs instead of inline base64. Keeps the body under the 10 MiB cap."}),
            "image_format": (IMAGE_ENCODE_CHOICES, {"default": "PNG", "tooltip": "Encoding used for inline data URIs."}),
        }
        optional.update(_api_widgets(ENV_MINIMAX))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    async def generate(self, prompt, duration_seconds, aspect_ratio, seed,
                       first_image=None, last_image=None, first_image_url="", last_image_url="",
                       aspect_ratio_override="", num_inference_steps=50, flow_shift=12.0,
                       audio_flow_shift=3.0, use_fal_upload=False, image_format="PNG",
                       api_key_env=ENV_MINIMAX, timeout=DEFAULT_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        ratio = aspect_ratio_override.strip() if not _is_blank(aspect_ratio_override) else aspect_ratio
        config = f"minimax-h3/fl2va | {ratio} | {duration_seconds}s | steps={num_inference_steps} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _video_error("TensorScale MiniMax H3 skipped: prompt is blank.", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "aspect_ratio": ratio,
            "duration_seconds": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "flow_shift": float(flow_shift),
            "audio_flow_shift": float(audio_flow_shift),
            "seed": int(seed),
        }

        # A typed URL wins over the socket for the same frame.
        try:
            for field, url_value, tensor in (
                ("first_image", first_image_url, first_image),
                ("last_image", last_image_url, last_image),
            ):
                if not _is_blank(url_value):
                    payload[field] = url_value.strip()
                    push_node_status(unique_id, f"{field}: using URL {url_value.strip()}", log)
                elif tensor is not None:
                    kind, value = _media_reference("image", tensor, use_fal_upload, image_format)
                    payload[field] = value
                    push_node_status(unique_id, f"{field}: sent as {kind}", log)
        except Exception as e:
            logger.exception("TensorScale MiniMax H3: frame encoding failed")
            return _video_error(f"ERROR preparing frames: {e!r}", config, log, unique_id)

        try:
            path, request_id, _ = await _post_binary(
                PATH_MINIMAX_FL2VA, payload, "video/mp4", api_key_env, timeout, "minimax_fl2va", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale MiniMax H3 fl2va failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (MiniMax-H3/fl2va): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# MiniMax H3 — REF2VA (ordered references -> video + audio)
# ---------------------------------------------------------------------------


class Soze_TensorScaleMiniMaxH3Reference:
    """TensorScale — MiniMax H3 reference-to-video+audio.

    Endpoint: POST /v2/MiniMax-H3/ref2va

    Takes the same generation fields as the FL2VA node plus an ordered
    `references` list. At least one image or video reference is required —
    audio-only requests are rejected by the API.

    Reference order is: connected image slots, then video slots, then audio
    slots, then any lines from `reference_uris`. Each `reference_uris` line is
    `type|url` (e.g. `audio|https://media.example.com/voice.mp3`); a bare URL
    has its type inferred from the file extension.

    Video references blow past the 10 MiB body cap almost immediately — use
    `reference_uris` or turn on `use_fal_upload` for those.
    """

    _URI_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
    _URI_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
    _URI_AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "duration_seconds": ("FLOAT", {
                "default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5,
                "tooltip": "Target length. Aligned server-side to the 24 fps / 17n+5 frame grid.",
            }),
            "aspect_ratio": (MINIMAX_RATIO_CHOICES, {"default": "16:9"}),
            "seed": ("INT", {"default": 1101, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "reference_uris": ("STRING", {
                "default": "", "multiline": True,
                "tooltip": "One reference per line: 'type|https://...' with type image/video/audio, "
                           "or a bare URL whose extension implies the type. Appended after the sockets.",
            }),
            "aspect_ratio_override": ("STRING", {"default": "", "multiline": False, "tooltip": "Any W:H from 1:4 to 4:1. Replaces the dropdown when set."}),
            "num_inference_steps": ("INT", {"default": 50, "min": 2, "max": 200}),
            "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            "audio_flow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            "use_fal_upload": ("BOOLEAN", {"default": False, "tooltip": "Upload connected media to FAL's CDN and send URLs. Strongly recommended for video/audio references."}),
            "image_format": (IMAGE_ENCODE_CHOICES, {"default": "PNG"}),
        }
        optional.update(_api_widgets(ENV_MINIMAX))
        for i in range(1, MINIMAX_MAX_REF_IMAGES + 1):
            optional[f"reference_image_{i}"] = ("IMAGE", {"tooltip": f"Image reference slot {i}."})
        for i in range(1, MINIMAX_MAX_REF_VIDEOS + 1):
            optional[f"reference_video_{i}"] = ("VIDEO", {"tooltip": f"Video reference slot {i}. Use use_fal_upload — inline video rarely fits the 10 MiB body cap."})
        for i in range(1, MINIMAX_MAX_REF_AUDIOS + 1):
            optional[f"reference_audio_{i}"] = ("AUDIO", {"tooltip": f"Audio reference slot {i} (voice cloning / native sound)."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    @classmethod
    def _infer_uri_type(cls, url):
        lowered = url.split("?", 1)[0].lower()
        if lowered.endswith(cls._URI_IMAGE_EXTS):
            return "image"
        if lowered.endswith(cls._URI_VIDEO_EXTS):
            return "video"
        if lowered.endswith(cls._URI_AUDIO_EXTS):
            return "audio"
        return None

    @classmethod
    def _parse_uri_lines(cls, text):
        """Parse the reference_uris textarea into reference dicts."""
        refs = []
        for lineno, raw in enumerate((text or "").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                kind, _, url = line.partition("|")
                kind, url = kind.strip().lower(), url.strip()
            else:
                url = line
                kind = cls._infer_uri_type(url)
            if kind not in ("image", "video", "audio"):
                raise ValueError(
                    f"reference_uris line {lineno}: cannot determine reference type for {line!r}. "
                    "Prefix it, e.g. 'audio|https://...'."
                )
            if not url:
                raise ValueError(f"reference_uris line {lineno}: missing URL.")
            refs.append({"type": kind, "uri": url})
        return refs

    async def generate(self, prompt, duration_seconds, aspect_ratio, seed,
                       reference_uris="", aspect_ratio_override="", num_inference_steps=50,
                       flow_shift=12.0, audio_flow_shift=3.0, use_fal_upload=False,
                       image_format="PNG", api_key_env=ENV_MINIMAX,
                       timeout=DEFAULT_READ_TIMEOUT, unique_id=None, **slots):
        log = EventLog()
        ratio = aspect_ratio_override.strip() if not _is_blank(aspect_ratio_override) else aspect_ratio
        config = f"minimax-h3/ref2va | {ratio} | {duration_seconds}s | steps={num_inference_steps} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _video_error("TensorScale MiniMax H3 Reference skipped: prompt is blank.", config, log, unique_id)

        # Build the ordered reference list: images, videos, audio, then URI lines.
        references = []
        try:
            for kind, count in (("image", MINIMAX_MAX_REF_IMAGES),
                                ("video", MINIMAX_MAX_REF_VIDEOS),
                                ("audio", MINIMAX_MAX_REF_AUDIOS)):
                for i in range(1, count + 1):
                    media = slots.get(f"reference_{kind}_{i}")
                    if media is None:
                        continue
                    push_node_status(unique_id, f"Encoding reference_{kind}_{i}...", log)
                    field, value = _media_reference(kind, media, use_fal_upload, image_format)
                    references.append({"type": kind, field: value})
            references.extend(self._parse_uri_lines(reference_uris))
        except Exception as e:
            logger.exception("TensorScale MiniMax H3 ref2va: reference encoding failed")
            return _video_error(f"ERROR preparing references: {e}", config, log, unique_id)

        if not any(r["type"] in ("image", "video") for r in references):
            return _video_error(
                "TensorScale MiniMax H3 Reference skipped: at least one image or video reference "
                "is required (the API rejects audio-only requests).",
                config, log, unique_id,
            )

        counts = {k: sum(1 for r in references if r["type"] == k) for k in ("image", "video", "audio")}
        push_node_status(unique_id, f"References: {counts['image']} image, {counts['video']} video, {counts['audio']} audio", log)

        payload = {
            "prompt": prompt.strip(),
            "references": references,
            "aspect_ratio": ratio,
            "duration_seconds": float(duration_seconds),
            "num_inference_steps": int(num_inference_steps),
            "flow_shift": float(flow_shift),
            "audio_flow_shift": float(audio_flow_shift),
            "seed": int(seed),
        }

        try:
            path, request_id, _ = await _post_binary(
                PATH_MINIMAX_REF2VA, payload, "video/mp4", api_key_env, timeout, "minimax_ref2va", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale MiniMax H3 ref2va failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (MiniMax-H3/ref2va): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# LTX-2.5 Fast
# ---------------------------------------------------------------------------


class Soze_TensorScaleLTX25Fast:
    """TensorScale — LTX-2.5 Fast video + synchronized audio.

    Endpoint: POST /v2/ltx2-5/fast
    Billing:  $0.04 per generated video second.

    The API key must carry the exact scope `ltx2-5-fast`. Frame rate is fixed
    at 24 fps, so `num_frames` picks the length: 121 = 5s, 241 = 10s.

    Image-to-video takes exactly one image, conditioned at frame 0, and it must
    be inline — this endpoint does not accept URLs, so there is no FAL-upload
    option here. Decoded size is capped at 10 MiB.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Describes both the video and the audio to generate."}),
            "resolution": (LTX25_RESOLUTION_CHOICES, {"default": "1920x1088"}),
            "num_frames": (LTX25_NUM_FRAMES_CHOICES, {"default": "121", "tooltip": "121 = 5 seconds, 241 = 10 seconds (at the fixed 24 fps)."}),
            "seed": ("INT", {"default": 10, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "image": ("IMAGE", {"tooltip": "Optional first-frame conditioning (frame_idx 0). Inline only — this endpoint rejects URLs."}),
            "image_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = omit and let the service default."}),
            "crf": ("INT", {"default": 0, "min": 0, "max": 51, "tooltip": "0 = omit. Otherwise overrides the output CRF."}),
            "image_format": (IMAGE_ENCODE_CHOICES, {"default": "PNG"}),
        }
        optional.update(_api_widgets(ENV_LTX25, default_timeout=MAX_READ_TIMEOUT))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    async def generate(self, prompt, resolution, num_frames, seed,
                       image=None, image_strength=0.0, crf=0, image_format="PNG",
                       api_key_env=ENV_LTX25, timeout=MAX_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        frames = int(num_frames)
        seconds = 5 if frames == 121 else 10
        config = f"ltx2-5-fast | {resolution} | {frames}f (~{seconds}s @24fps) | seed={seed}"
        log.add(f"Config: {config}")
        log.add(f"Estimated cost: ~${seconds * 0.04:.2f} at $0.04/video second.")

        if _is_blank(prompt):
            return _video_error("TensorScale LTX-2.5 Fast skipped: prompt is blank.", config, log, unique_id)

        width, _, height = resolution.partition("x")
        payload = {
            "prompt": prompt.strip(),
            "width": int(width),
            "height": int(height),
            "num_frames": frames,
            "frame_rate": LTX25_FRAME_RATE,
            "seed": int(seed),
        }

        if image is not None:
            try:
                data_uri = _image_to_data_uri(image, image_format)
            except Exception as e:
                logger.exception("TensorScale LTX-2.5: image encoding failed")
                return _video_error(f"ERROR encoding image: {e!r}", config, log, unique_id)
            entry = {"data": data_uri, "frame_idx": 0}
            if image_strength and image_strength > 0:
                entry["strength"] = float(image_strength)
            if crf and crf > 0:
                entry["crf"] = int(crf)
            payload["images"] = [entry]
            push_node_status(unique_id, "Image-to-video: 1 inline frame at frame_idx 0.", log)

        try:
            path, request_id, _ = await _post_binary(
                PATH_LTX25_FAST, payload, "video/mp4", api_key_env, timeout, "ltx25", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale LTX-2.5 Fast failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (ltx2-5-fast): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# LTX-2.3 Fast
# ---------------------------------------------------------------------------


class Soze_TensorScaleLTX23Fast:
    """TensorScale — LTX-2.3 Fast video generation.

    Endpoint: POST /v2/ltx-2.3/fast

    Text-to-video by default. For image-to-video either paste a public HTTPS
    URL into `image_url` (sent as the flat `image_url` + `image_strength`
    pair) or connect the `image` socket (sent as an `image_inputs` entry with
    an inline data URI, so you can also aim it at a frame other than 0).
    A typed URL wins if both are supplied.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "duration": (LTX23_DURATION_CHOICES, {"default": "5", "tooltip": "Clip length in seconds."}),
            "resolution": (LTX23_RESOLUTION_CHOICES, {"default": "1280x768"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "image": ("IMAGE", {"tooltip": "Optional conditioning frame, sent inline via image_inputs."}),
            "image_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS image URL. Overrides the image socket."}),
            "image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "frame_idx": ("INT", {"default": 0, "min": 0, "max": 240, "tooltip": "Which frame the socket image conditions (image_inputs only)."}),
            "use_fal_upload": ("BOOLEAN", {"default": False, "tooltip": "Upload the image socket to FAL's CDN and use the flat image_url form instead of inlining."}),
            "image_format": (IMAGE_ENCODE_CHOICES, {"default": "PNG"}),
        }
        optional.update(_api_widgets(ENV_LTX23))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    async def generate(self, prompt, duration, resolution, seed,
                       image=None, image_url="", image_strength=1.0, frame_idx=0,
                       use_fal_upload=False, image_format="PNG",
                       api_key_env=ENV_LTX23, timeout=DEFAULT_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        config = f"ltx-2.3-fast | {resolution} | {duration}s | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _video_error("TensorScale LTX-2.3 Fast skipped: prompt is blank.", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "duration": int(duration),
            "resolution": resolution,
            "seed": int(seed),
        }

        try:
            if not _is_blank(image_url):
                payload["image_url"] = image_url.strip()
                payload["image_strength"] = float(image_strength)
                push_node_status(unique_id, f"Image-to-video via URL: {image_url.strip()}", log)
            elif image is not None:
                field, value = _media_reference("image", image, use_fal_upload, image_format)
                if field == "uri":
                    payload["image_url"] = value
                    payload["image_strength"] = float(image_strength)
                    push_node_status(unique_id, f"Image-to-video via FAL URL: {value}", log)
                else:
                    payload["image_inputs"] = [{
                        "data": value,
                        "frame_idx": int(frame_idx),
                        "strength": float(image_strength),
                    }]
                    push_node_status(unique_id, f"Image-to-video: inline frame at frame_idx {int(frame_idx)}.", log)
        except Exception as e:
            logger.exception("TensorScale LTX-2.3: image encoding failed")
            return _video_error(f"ERROR preparing image: {e!r}", config, log, unique_id)

        try:
            path, request_id, _ = await _post_binary(
                PATH_LTX23_FAST, payload, "video/mp4", api_key_env, timeout, "ltx23", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale LTX-2.3 Fast failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (ltx-2.3-fast): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Cosmos3 Nano — T2V
# ---------------------------------------------------------------------------


class Soze_TensorScaleCosmos3NanoT2V:
    """TensorScale — Cosmos3 Nano text-to-video.

    Endpoint: POST /v2/cosmos3-nano/t2v

    `resolution` is an output *height* profile, not a WxH pair; the width comes
    from `aspect_ratio`, which Cosmos encodes comma-separated ("16,9") rather
    than with a colon. `seconds` is converted server-side to a valid Cosmos
    frame count.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "seconds": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 60.0, "step": 0.5, "tooltip": "Converted server-side to a valid Cosmos frame count."}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60, "tooltip": "24 matches the native generation profile."}),
            "resolution": (COSMOS_RESOLUTION_CHOICES, {"default": "720", "tooltip": "Output height profile."}),
            "aspect_ratio": (COSMOS_RATIO_CHOICES, {"default": "16,9", "tooltip": "Comma-separated W,H — Cosmos does not use the colon form."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "num_steps": ("INT", {"default": 35, "min": 1, "max": 100, "tooltip": "Diffusion steps. The native default is 35."}),
            "guidance": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Prompt guidance strength. The native default is 6.0."}),
        }
        optional.update(_api_widgets(ENV_COSMOS))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    async def generate(self, prompt, seconds, fps, resolution, aspect_ratio, seed,
                       num_steps=35, guidance=6.0,
                       api_key_env=ENV_COSMOS, timeout=DEFAULT_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        config = f"cosmos3-nano/t2v | {resolution}p {aspect_ratio} | {seconds}s @{fps}fps | steps={num_steps} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _video_error("TensorScale Cosmos3 Nano T2V skipped: prompt is blank.", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "seconds": float(seconds),
            "fps": int(fps),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_steps": int(num_steps),
            "guidance": float(guidance),
            "seed": int(seed),
        }

        try:
            path, request_id, _ = await _post_binary(
                PATH_COSMOS_T2V, payload, "video/mp4", api_key_env, timeout, "cosmos_t2v", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale Cosmos3 Nano t2v failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (cosmos3-nano/t2v): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Cosmos3 Nano — V2V
# ---------------------------------------------------------------------------


class Soze_TensorScaleCosmos3NanoV2V:
    """TensorScale — Cosmos3 Nano video-to-video (sim-to-real restyle).

    Endpoint: POST /v2/cosmos3-nano/v2v

    Preserves the source camera, geometry, timing and motion while retargeting
    appearance to the prompt. Supply the source either as a public HTTPS URL in
    `video_url` or through the `video` socket; inline video is base64-encoded
    into the body, so anything more than a few seconds needs `video_url` or
    `use_fal_upload` to stay under the 10 MiB cap.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Describes the target appearance, not the motion."}),
            "seconds": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 60.0, "step": 0.5}),
            "fps": ("INT", {"default": 24, "min": 1, "max": 60}),
            "resolution": (COSMOS_RESOLUTION_CHOICES, {"default": "720", "tooltip": "Output height profile."}),
            "aspect_ratio": (COSMOS_RATIO_CHOICES, {"default": "16,9"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "video": ("VIDEO", {"tooltip": "Source clip. Inline unless use_fal_upload is on."}),
            "video_url": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS URL of the source video. Overrides the video socket."}),
            "num_steps": ("INT", {"default": 50, "min": 1, "max": 100, "tooltip": "50 is the documented high-quality transformation setting."}),
            "guidance": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 30.0, "step": 0.1}),
            "use_fal_upload": ("BOOLEAN", {"default": True, "tooltip": "Upload the video socket to FAL's CDN and send a URL. On by default — inline video usually exceeds the 10 MiB body cap."}),
        }
        optional.update(_api_widgets(ENV_COSMOS))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _video_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/VideoGeneration"

    async def generate(self, prompt, seconds, fps, resolution, aspect_ratio, seed,
                       video=None, video_url="", num_steps=50, guidance=4.5, use_fal_upload=True,
                       api_key_env=ENV_COSMOS, timeout=DEFAULT_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        config = f"cosmos3-nano/v2v | {resolution}p {aspect_ratio} | {seconds}s @{fps}fps | steps={num_steps} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _video_error("TensorScale Cosmos3 Nano V2V skipped: prompt is blank.", config, log, unique_id)
        if video is None and _is_blank(video_url):
            return _video_error(
                "TensorScale Cosmos3 Nano V2V skipped: connect a video or set video_url.",
                config, log, unique_id,
            )

        try:
            if not _is_blank(video_url):
                source = video_url.strip()
                push_node_status(unique_id, f"Source video: URL {source}", log)
            else:
                push_node_status(unique_id, "Encoding source video...", log)
                _, source = _media_reference("video", video, use_fal_upload)
        except Exception as e:
            logger.exception("TensorScale Cosmos3 Nano v2v: source encoding failed")
            return _video_error(f"ERROR preparing source video: {e}", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "video": source,
            "seconds": float(seconds),
            "fps": int(fps),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_steps": int(num_steps),
            "guidance": float(guidance),
            "seed": int(seed),
        }

        try:
            path, request_id, _ = await _post_binary(
                PATH_COSMOS_V2V, payload, "video/mp4", api_key_env, timeout, "cosmos_v2v", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale Cosmos3 Nano v2v failed")
            return _video_error(f"ERROR: {e}", config, log, unique_id)

        headline = f"OK (cosmos3-nano/v2v): {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (VideoFromFile(path), path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Hunyuan Image 3 Fast
# ---------------------------------------------------------------------------


class Soze_TensorScaleHunyuanImage3:
    """TensorScale — Hunyuan Image 3 Fast text-to-image and image editing.

    Endpoint: POST /v2/hunyuan-image-3-instruct/fast

    Leave both image slots empty for text-to-image; connect one or two
    references (the documented maximum) to edit or blend. `bot_task` switches
    the model's front-end behaviour: `image` generates directly, `recaption`
    rewrites the prompt first, `think` reasons before generating, and
    `think_recaption` does both.
    """

    MAX_IMAGES = 2

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "What to generate, or the edit to apply to the references."}),
            "bot_task": (HUNYUAN_BOT_TASK_CHOICES, {"default": "image"}),
            "seed": ("INT", {"default": 43, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "image_1": ("IMAGE", {"tooltip": "Reference image 1 (API max is 2)."}),
            "image_2": ("IMAGE", {"tooltip": "Reference image 2."}),
            "image_url_1": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS URL. Overrides image_1."}),
            "image_url_2": ("STRING", {"default": "", "multiline": False, "tooltip": "Public HTTPS URL. Overrides image_2."}),
            "use_fal_upload": ("BOOLEAN", {"default": False, "tooltip": "Upload connected images to FAL's CDN and send URLs instead of inline base64."}),
            "image_format": (IMAGE_ENCODE_CHOICES, {"default": "PNG"}),
        }
        optional.update(_api_widgets(ENV_HUNYUAN, default_timeout=600))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _image_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/ImageGeneration"

    async def generate(self, prompt, bot_task, seed,
                       image_1=None, image_2=None, image_url_1="", image_url_2="",
                       use_fal_upload=False, image_format="PNG",
                       api_key_env=ENV_HUNYUAN, timeout=600, unique_id=None):
        log = EventLog()
        config = f"hunyuan-image-3-instruct/fast | task={bot_task} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _image_error("TensorScale Hunyuan Image 3 skipped: prompt is blank.", config, log, unique_id)

        images = []
        try:
            for url_value, tensor, label in (
                (image_url_1, image_1, "image_1"),
                (image_url_2, image_2, "image_2"),
            ):
                if not _is_blank(url_value):
                    images.append(url_value.strip())
                    push_node_status(unique_id, f"{label}: using URL {url_value.strip()}", log)
                elif tensor is not None:
                    _, value = _media_reference("image", tensor, use_fal_upload, image_format)
                    images.append(value)
                    push_node_status(unique_id, f"{label}: attached", log)
        except Exception as e:
            logger.exception("TensorScale Hunyuan Image 3: reference encoding failed")
            return _image_error(f"ERROR preparing references: {e}", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "images": images,  # empty list = text-to-image, per the schema
            "bot_task": bot_task,
            "seed": int(seed),
        }

        try:
            path, request_id, _ = await _post_binary(
                PATH_HUNYUAN_IMAGE3, payload, "image/jpeg", api_key_env, timeout, "hunyuan3", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale Hunyuan Image 3 failed")
            return _image_error(f"ERROR: {e}", config, log, unique_id)

        try:
            tensor = _file_to_image_tensor(path)
        except Exception as e:
            logger.exception("TensorScale Hunyuan Image 3: decode failed (%s)", path)
            return _image_error(f"ERROR decoding returned image: {e!r}", config, log, unique_id, request_id)

        headline = f"OK (hunyuan-image-3): {tensor.shape[2]}x{tensor.shape[1]} -> {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (tensor, path, request_id, config, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# SenseNova U1.5
# ---------------------------------------------------------------------------


class Soze_TensorScaleSenseNovaU15:
    """TensorScale — SenseNova U1.5 text-to-image.

    Endpoint: POST /v2/sensenova-u1.5/t2i

    The current service profile is pinned to the "golden" 2048x2048 / 50-step
    request: `size` must be 2048x2048, `cfg_scale` must be 4.0, and
    `timestep_shift` must be 3.0. Those three are exposed anyway so the node
    keeps working if TensorScale widens the profile — leave them alone unless
    the docs say otherwise.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
        }
        optional = {
            "size": ("STRING", {"default": "2048x2048", "multiline": False, "tooltip": "Must be 2048x2048 in the current service profile."}),
            "cfg_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Must be 4.0 in the current service profile."}),
            "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Must be 3.0 in the current service profile."}),
        }
        optional.update(_api_widgets(ENV_SENSENOVA, default_timeout=DEFAULT_READ_TIMEOUT))
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES, RETURN_NAMES = _image_returns()
    FUNCTION = "generate"
    CATEGORY = "TensorScale/ImageGeneration"

    async def generate(self, prompt, seed, size="2048x2048", cfg_scale=4.0, timestep_shift=3.0,
                       api_key_env=ENV_SENSENOVA, timeout=DEFAULT_READ_TIMEOUT, unique_id=None):
        log = EventLog()
        effective_size = size.strip() if not _is_blank(size) else "2048x2048"
        config = f"sensenova-u1.5/t2i | {effective_size} | cfg={cfg_scale} | shift={timestep_shift} | seed={seed}"
        log.add(f"Config: {config}")

        if _is_blank(prompt):
            return _image_error("TensorScale SenseNova U1.5 skipped: prompt is blank.", config, log, unique_id)

        payload = {
            "prompt": prompt.strip(),
            "size": effective_size,
            "seed": int(seed),
            "cfg_scale": float(cfg_scale),
            "timestep_shift": float(timestep_shift),
        }

        try:
            path, request_id, _ = await _post_binary(
                PATH_SENSENOVA_T2I, payload, "image/png", api_key_env, timeout, "sensenova", log, unique_id
            )
        except Exception as e:
            logger.exception("TensorScale SenseNova U1.5 failed")
            return _image_error(f"ERROR: {e}", config, log, unique_id)

        try:
            tensor = _file_to_image_tensor(path)
        except Exception as e:
            logger.exception("TensorScale SenseNova U1.5: decode failed (%s)", path)
            return _image_error(f"ERROR decoding returned image: {e!r}", config, log, unique_id, request_id)

        headline = f"OK (sensenova-u1.5): {tensor.shape[2]}x{tensor.shape[1]} -> {os.path.basename(path)}"
        push_node_status(unique_id, headline, log)
        return (tensor, path, request_id, config, finalize_status(headline, log))
