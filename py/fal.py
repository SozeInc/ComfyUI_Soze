import logging
from datetime import datetime

from comfy_api_nodes.util import download_url_to_video_output

from .fal_utils import ApiHandler, ImageUtils

logger = logging.getLogger(__name__)


class _EventLog:
    """Accumulates timestamped events during a node run so the final status
    output can include a full transcript of what happened."""

    def __init__(self):
        self.events: list[str] = []

    def add(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {text}")

    def render(self) -> str:
        return "\n".join(self.events)


def _push_node_status(unique_id, text: str, event_log: _EventLog | None = None) -> None:
    """Push a status string to the node body widget (the area native API nodes use)
    and append it to the run's event log. Silently no-ops on the widget side if
    the running ComfyUI doesn't expose the API."""
    if event_log is not None:
        event_log.add(text)
    if unique_id is None:
        return
    try:
        from server import PromptServer
        server = PromptServer.instance
        if hasattr(server, "send_progress_text"):
            server.send_progress_text(text, unique_id)
        else:
            server.send_sync("progress_text", {"node_id": unique_id, "text": text})
    except Exception:
        # Status display is best-effort — never let it break the node.
        logger.debug("Failed to push node status for %s", unique_id, exc_info=True)


def _build_config_string(speed: str, resolution: str, duration: str, aspect_ratio: str, generate_audio: bool) -> str:
    """Return a compact run-config tag, e.g. 'Std-480-15s-169-Audio'."""
    speed_tag = "Fast" if speed == "fast" else "Std"
    res_tag = resolution.rstrip("p") if resolution else "Auto"
    dur_tag = "Auto" if duration == "auto" or not duration else f"{duration}s"
    ar_tag = "Auto" if aspect_ratio == "auto" or not aspect_ratio else aspect_ratio.replace(":", "")
    audio_tag = "Audio" if generate_audio else "NoAudio"
    return f"{speed_tag}-{res_tag}-{dur_tag}-{ar_tag}-{audio_tag}"


def _finalize_status(headline: str, event_log: _EventLog) -> str:
    """Combine a one-line headline with the full event log for the status output."""
    log_section = event_log.render()
    if log_section:
        return f"{headline}\n\nEvents:\n{log_section}"
    return headline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SPEED_CHOICES = ["standard", "fast"]
DURATION_CHOICES = ["auto", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
ASPECT_CHOICES = ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
RESOLUTION_CHOICES = ["480p", "720p", "1080p"]
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
SEED_DEFAULT = 0  # 0 = let FAL choose; any positive int is forwarded.


def _seedance_endpoint(speed: str, kind: str) -> str:
    """Build the Seedance 2.0 endpoint id for (speed, kind)."""
    if speed == "fast":
        return f"bytedance/seedance-2.0/fast/{kind}"
    return f"bytedance/seedance-2.0/{kind}"


def _validate_resolution(speed: str, resolution: str) -> None:
    if speed == "fast" and resolution == "1080p":
        raise ValueError(
            "1080p is not supported by the fast Seedance endpoints. "
            "Choose 480p or 720p, or set speed to 'standard'."
        )


def _is_blank(value) -> bool:
    """True for None or whitespace-only strings — treated as 'not connected'."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id):
    args = {
        "prompt": prompt,
        "resolution": resolution,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "generate_audio": bool(generate_audio),
    }
    if isinstance(seed, int) and seed > 0:
        args["seed"] = seed
    if not _is_blank(end_user_id):
        args["end_user_id"] = end_user_id.strip()
    return args


async def _call_seedance(endpoint: str, arguments: dict, unique_id=None, event_log: _EventLog | None = None):
    if event_log is None:
        event_log = _EventLog()
    _push_node_status(unique_id, f"Submitting to {endpoint}...", event_log)
    try:
        result = ApiHandler.submit_and_get_result(endpoint, arguments)
    except Exception as e:
        logger.exception("Seedance 2 generation failed (endpoint=%s)", endpoint)
        headline = f"ERROR ({endpoint}): {e!r}"
        _push_node_status(unique_id, headline, event_log)
        return (None, "", _finalize_status(headline, event_log))

    video_info = result.get("video") or {}
    video_url = video_info.get("url")
    if not video_url:
        headline = f"ERROR ({endpoint}): API did not return a video URL. Response: {result!r}"
        _push_node_status(unique_id, headline, event_log)
        return (None, "", _finalize_status(headline, event_log))

    seed_used = int(result.get("seed", 0) or 0)
    logger.info("Seedance 2 (%s) video URL: %s (seed=%s)", endpoint, video_url, seed_used)
    _push_node_status(unique_id, f"Downloading video (seed={seed_used})...", event_log)
    try:
        video = await download_url_to_video_output(video_url)
    except Exception as e:
        logger.exception("Seedance 2 video download failed (url=%s)", video_url)
        headline = f"ERROR downloading video ({endpoint}): {e!r}"
        _push_node_status(unique_id, headline, event_log)
        return (None, video_url, _finalize_status(headline, event_log))
    headline = f"OK ({endpoint}): seed={seed_used}, url={video_url}"
    _push_node_status(unique_id, headline, event_log)
    return (video, video_url, _finalize_status(headline, event_log))


def _upload_image_or_raise(image_tensor, label: str) -> str:
    url = ImageUtils.upload_image(image_tensor[0:1])
    if not url:
        raise RuntimeError(f"Failed to upload {label} to FAL.")
    return url


# ---------------------------------------------------------------------------
# Seedance 2.0 — image-to-video
# ---------------------------------------------------------------------------


class Soze_FALSeedance2ImageToVideo:
    """ByteDance Seedance 2.0 image-to-video. `speed` switches between the
    standard endpoint (480p/720p/1080p) and the fast endpoint (480p/720p)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "speed": (SPEED_CHOICES, {"default": "standard", "tooltip": "'fast' uses the fast/* endpoint and disallows 1080p."}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "resolution": (RESOLUTION_CHOICES, {"default": "720p", "tooltip": "1080p is only valid when speed=standard."}),
                "duration": (DURATION_CHOICES, {"default": "auto"}),
                "aspect_ratio": (ASPECT_CHOICES, {"default": "auto"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": SEED_DEFAULT, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = let FAL choose."}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "Optional: target last frame."}),
                "end_user_id": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "config", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, speed, prompt, image, resolution, duration, aspect_ratio,
                       generate_audio, seed, end_image=None, end_user_id="", unique_id=None):
        config = _build_config_string(speed, resolution, duration, aspect_ratio, generate_audio)
        event_log = _EventLog()
        event_log.add(f"Config: {config}")

        # None-tolerance: if the caller didn't actually wire up the required inputs,
        # silently no-op rather than raise. Matches the rest of the Soze node suite.
        if image is None or _is_blank(prompt):
            headline = "Seedance image-to-video skipped: missing required input(s)."
            logger.info(headline)
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        try:
            _validate_resolution(speed, resolution)
            args = _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id)
            _push_node_status(unique_id, "Uploading start image...", event_log)
            args["image_url"] = _upload_image_or_raise(image, "start image")
            if end_image is not None:
                _push_node_status(unique_id, "Uploading end image...", event_log)
                args["end_image_url"] = _upload_image_or_raise(end_image, "end image")
        except Exception as e:
            logger.exception("Seedance image-to-video setup failed")
            headline = f"ERROR: {e!r}"
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        video, video_url, status = await _call_seedance(
            _seedance_endpoint(speed, "image-to-video"), args,
            unique_id=unique_id, event_log=event_log,
        )
        return (video, video_url, config, status)


# ---------------------------------------------------------------------------
# Seedance 2.0 — reference-to-video
# ---------------------------------------------------------------------------


def _ref_input_types() -> dict:
    required = {
        "speed": (SPEED_CHOICES, {"default": "standard", "tooltip": "'fast' uses the fast/* endpoint and disallows 1080p."}),
        "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite uploaded inputs in the prompt as @Image1..@Image9, @Video1..@Video3, @Audio1..@Audio3."}),
        "resolution": (RESOLUTION_CHOICES, {"default": "720p", "tooltip": "1080p is only valid when speed=standard."}),
        "duration": (DURATION_CHOICES, {"default": "auto"}),
        "aspect_ratio": (ASPECT_CHOICES, {"default": "auto"}),
        "generate_audio": ("BOOLEAN", {"default": True}),
        "seed": ("INT", {"default": SEED_DEFAULT, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = let FAL choose."}),
    }
    optional: dict = {}
    for i in range(1, MAX_REFERENCE_IMAGES + 1):
        optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Cite as @Image{i} in the prompt."})
    for i in range(1, MAX_REFERENCE_VIDEOS + 1):
        optional[f"video_{i}"] = ("VIDEO", {"tooltip": f"Cite as @Video{i} in the prompt. Uploaded to FAL via upload_file."})
    for i in range(1, MAX_REFERENCE_AUDIOS + 1):
        optional[f"audio_{i}"] = ("AUDIO", {"tooltip": f"Cite as @Audio{i} in the prompt. Uploaded to FAL via upload_file."})
    optional["end_user_id"] = ("STRING", {"default": "", "multiline": False})
    return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}


def _collect_ordered(kwargs: dict, prefix: str, count: int) -> list[tuple[int, object]]:
    """Pull e.g. image_1..image_N from kwargs in slot order, dropping missing ones."""
    return [(i, kwargs.get(f"{prefix}_{i}")) for i in range(1, count + 1)]


class Soze_FALSeedance2ReferenceToVideo:
    """ByteDance Seedance 2.0 reference-to-video. `speed` switches between the
    standard endpoint (480p/720p/1080p) and the fast endpoint (480p/720p).

    Up to 9 reference images, 3 videos, and 3 audios — each on its own typed
    input slot. Anything connected gets uploaded to FAL via upload_file. Cite
    them in `prompt` as @Image1..@Image9, @Video1..@Video3, @Audio1..@Audio3.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _ref_input_types()

    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "config", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, speed, prompt, resolution, duration, aspect_ratio,
                       generate_audio, seed, end_user_id="", unique_id=None, **slots):
        config = _build_config_string(speed, resolution, duration, aspect_ratio, generate_audio)
        event_log = _EventLog()
        event_log.add(f"Config: {config}")

        if _is_blank(prompt):
            headline = "Seedance reference-to-video skipped: prompt is blank."
            logger.info(headline)
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        try:
            _validate_resolution(speed, resolution)
            args = _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id)

            _push_node_status(unique_id, "Uploading reference images...", event_log)
            image_urls, image_conn, image_fail = self._upload_slots(
                slots, "image", MAX_REFERENCE_IMAGES, ImageUtils.upload_image,
                transform=lambda img: img[0:1],
            )
            if image_urls:
                args["image_urls"] = image_urls

            _push_node_status(unique_id, "Uploading reference videos...", event_log)
            video_urls, video_conn, video_fail = self._upload_slots(
                slots, "video", MAX_REFERENCE_VIDEOS, ImageUtils.upload_video,
            )
            if video_urls:
                args["video_urls"] = video_urls

            _push_node_status(unique_id, "Uploading reference audio...", event_log)
            audio_urls, audio_conn, audio_fail = self._upload_slots(
                slots, "audio", MAX_REFERENCE_AUDIOS, ImageUtils.upload_audio,
            )
            if audio_urls:
                args["audio_urls"] = audio_urls
        except Exception as e:
            logger.exception("Seedance reference-to-video setup failed")
            headline = f"ERROR: {e!r}"
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        connected_total = image_conn + video_conn + audio_conn
        upload_failures = image_fail + video_fail + audio_fail
        event_log.add(
            f"Uploads: image {len(image_urls)}/{image_conn}, "
            f"video {len(video_urls)}/{video_conn}, audio {len(audio_urls)}/{audio_conn}"
        )

        if connected_total == 0:
            headline = "Seedance reference-to-video skipped: no reference inputs connected."
            logger.info(headline)
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        if not (image_urls or video_urls or audio_urls):
            # Inputs were connected but every upload failed — surface that distinctly.
            details = "; ".join(upload_failures) if upload_failures else "no upload error captured"
            headline = (
                f"Seedance reference-to-video failed: {connected_total} input(s) connected "
                f"but all uploads failed. Details: {details}"
            )
            logger.error(headline)
            _push_node_status(unique_id, headline, event_log)
            return (None, "", config, _finalize_status(headline, event_log))

        if upload_failures:
            event_log.add(f"Partial uploads — {len(upload_failures)} failed: " + "; ".join(upload_failures))

        video_out, video_url, status = await _call_seedance(
            _seedance_endpoint(speed, "reference-to-video"), args,
            unique_id=unique_id, event_log=event_log,
        )
        return (video_out, video_url, config, status)

    @staticmethod
    def _upload_slots(slots: dict, prefix: str, count: int, uploader, transform=None):
        """Upload connected slots in order. Returns (urls, connected_count, failures)
        where failures is a list of human-readable strings — one per slot whose upload
        returned no URL or raised."""
        urls: list[str] = []
        connected = 0
        failures: list[str] = []
        for slot_idx, value in _collect_ordered(slots, prefix, count):
            if value is None:
                continue
            connected += 1
            try:
                payload = transform(value) if transform else value
                url = uploader(payload)
            except Exception as e:
                logger.exception("Seedance: %s_%d upload raised", prefix, slot_idx)
                failures.append(f"{prefix}_{slot_idx}: {e!r}")
                continue
            if url:
                urls.append(url)
            else:
                logger.warning("Seedance: %s_%d returned no URL; check FAL config and earlier log entries", prefix, slot_idx)
                failures.append(f"{prefix}_{slot_idx}: uploader returned None (see ComfyUI log for cause)")
        return urls, connected, failures


# ---------------------------------------------------------------------------
# Veo 3.1 (existing)
# ---------------------------------------------------------------------------


class Veo31RefImgVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 8}),
            },
            "optional": {
                "generate_audio": ("BOOLEAN", {"default": True}),
                "auto_fix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    async def generate_video(
        self,
        images,
        prompt,
        resolution,
        duration,
        generate_audio=True,
        auto_fix=True,
    ):
        image_urls = []
        for i in range(images.shape[0]):
            url = ImageUtils.upload_image(images[i])
            if url:
                image_urls.append(url)

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "resolution": resolution,
            "duration": duration,
            "generate_audio": generate_audio,
            "auto_fix": auto_fix,
        }

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/veo3.1/reference-to-video", arguments
            )
            video_url = result["video"]["url"]
            logger.info("Veo3.1 Video URL: %s", video_url)
            return (await download_url_to_video_output(video_url),)
        except Exception:
            logger.exception("Veo3.1 video generation failed")
            raise
