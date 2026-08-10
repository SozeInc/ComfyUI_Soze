import io
import logging
import os
import tempfile

import numpy as np
import requests
import torch
from comfy.utils import common_upscale
from comfy_api_nodes.util import download_url_to_video_output
from PIL import Image

from .fal_utils import ApiHandler, FalConfig, ImageUtils
from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = (10, 600)


def _build_config_string(speed: str, resolution: str, duration: str, aspect_ratio: str, generate_audio: bool) -> str:
    """Return a compact run-config tag, e.g. 'Std-480-15s-169-Audio'."""
    speed_tag = "Fast" if speed == "fast" else "Std"
    res_tag = resolution.rstrip("p") if resolution else "Auto"
    dur_tag = "Auto" if duration == "auto" or not duration else f"{duration}s"
    ar_tag = "Auto" if aspect_ratio == "auto" or not aspect_ratio else aspect_ratio.replace(":", "")
    audio_tag = "Audio" if generate_audio else "NoAudio"
    return f"{speed_tag}-{res_tag}-{dur_tag}-{ar_tag}-{audio_tag}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SPEED_CHOICES = ["standard", "fast"]
DURATION_CHOICES = ["auto", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
ASPECT_CHOICES = ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
RESOLUTION_CHOICES = ["480p", "720p", "1080p"]
BITRATE_MODE_CHOICES = ["standard", "high"]
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


def _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id,
                       bitrate_mode="standard"):
    args = {
        "prompt": prompt,
        "resolution": resolution,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "generate_audio": bool(generate_audio),
    }
    if bitrate_mode in BITRATE_MODE_CHOICES:
        args["bitrate_mode"] = bitrate_mode
    if isinstance(seed, int) and seed > 0:
        args["seed"] = seed
    if not _is_blank(end_user_id):
        args["end_user_id"] = end_user_id.strip()
    return args


async def _call_seedance(endpoint: str, arguments: dict, unique_id=None, event_log: EventLog | None = None):
    if event_log is None:
        event_log = EventLog()
    push_node_status(unique_id, f"Submitting to {endpoint}...", event_log)
    try:
        result = ApiHandler.submit_and_get_result(endpoint, arguments)
    except Exception as e:
        logger.exception("Seedance 2 generation failed (endpoint=%s)", endpoint)
        headline = f"ERROR ({endpoint}): {e!r}"
        push_node_status(unique_id, headline, event_log)
        return (None, "", finalize_status(headline, event_log))

    video_info = result.get("video") or {}
    video_url = video_info.get("url")
    if not video_url:
        headline = f"ERROR ({endpoint}): API did not return a video URL. Response: {result!r}"
        push_node_status(unique_id, headline, event_log)
        return (None, "", finalize_status(headline, event_log))

    seed_used = int(result.get("seed", 0) or 0)
    logger.info("Seedance 2 (%s) video URL: %s (seed=%s)", endpoint, video_url, seed_used)
    push_node_status(unique_id, f"Downloading video (seed={seed_used})...", event_log)
    try:
        video = await download_url_to_video_output(video_url)
    except Exception as e:
        logger.exception("Seedance 2 video download failed (url=%s)", video_url)
        headline = f"ERROR downloading video ({endpoint}): {e!r}"
        push_node_status(unique_id, headline, event_log)
        return (None, video_url, finalize_status(headline, event_log))
    headline = f"OK ({endpoint}): seed={seed_used}, url={video_url}"
    push_node_status(unique_id, headline, event_log)
    return (video, video_url, finalize_status(headline, event_log))


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
                "bitrate_mode": (BITRATE_MODE_CHOICES, {"default": "standard", "tooltip": "Output bitrate mode. 'high' requests a higher-quality, larger-file encode; 'standard' uses the default bitrate."}),
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
                       generate_audio, seed, bitrate_mode="standard", end_image=None,
                       end_user_id="", unique_id=None):
        config = _build_config_string(speed, resolution, duration, aspect_ratio, generate_audio)
        event_log = EventLog()
        event_log.add(f"Config: {config}")

        # None-tolerance: if the caller didn't actually wire up the required inputs,
        # silently no-op rather than raise. Matches the rest of the Soze node suite.
        if image is None or _is_blank(prompt):
            headline = "Seedance image-to-video skipped: missing required input(s)."
            logger.info(headline)
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

        try:
            _validate_resolution(speed, resolution)
            args = _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id, bitrate_mode)
            push_node_status(unique_id, "Uploading start image...", event_log)
            args["image_url"] = _upload_image_or_raise(image, "start image")
            if end_image is not None:
                push_node_status(unique_id, "Uploading end image...", event_log)
                args["end_image_url"] = _upload_image_or_raise(end_image, "end image")
        except Exception as e:
            logger.exception("Seedance image-to-video setup failed")
            headline = f"ERROR: {e!r}"
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

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
        "bitrate_mode": (BITRATE_MODE_CHOICES, {"default": "standard", "tooltip": "Output bitrate mode. 'high' requests a higher-quality, larger-file encode; 'standard' uses the default bitrate."}),
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
                       generate_audio, seed, bitrate_mode="standard", end_user_id="",
                       unique_id=None, **slots):
        config = _build_config_string(speed, resolution, duration, aspect_ratio, generate_audio)
        event_log = EventLog()
        event_log.add(f"Config: {config}")

        if _is_blank(prompt):
            headline = "Seedance reference-to-video skipped: prompt is blank."
            logger.info(headline)
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

        try:
            _validate_resolution(speed, resolution)
            args = _build_common_args(prompt, resolution, duration, aspect_ratio, generate_audio, seed, end_user_id, bitrate_mode)

            push_node_status(unique_id, "Uploading reference images...", event_log)
            image_urls, image_conn, image_fail = self._upload_slots(
                slots, "image", MAX_REFERENCE_IMAGES, ImageUtils.upload_image,
                transform=lambda img: img[0:1],
            )
            if image_urls:
                args["image_urls"] = image_urls

            push_node_status(unique_id, "Uploading reference videos...", event_log)
            video_urls, video_conn, video_fail = self._upload_slots(
                slots, "video", MAX_REFERENCE_VIDEOS, ImageUtils.upload_video,
            )
            if video_urls:
                args["video_urls"] = video_urls

            push_node_status(unique_id, "Uploading reference audio...", event_log)
            audio_urls, audio_conn, audio_fail = self._upload_slots(
                slots, "audio", MAX_REFERENCE_AUDIOS, ImageUtils.upload_audio,
            )
            if audio_urls:
                args["audio_urls"] = audio_urls
        except Exception as e:
            logger.exception("Seedance reference-to-video setup failed")
            headline = f"ERROR: {e!r}"
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

        connected_total = image_conn + video_conn + audio_conn
        upload_failures = image_fail + video_fail + audio_fail
        event_log.add(
            f"Uploads: image {len(image_urls)}/{image_conn}, "
            f"video {len(video_urls)}/{video_conn}, audio {len(audio_urls)}/{audio_conn}"
        )

        if connected_total == 0:
            headline = "Seedance reference-to-video skipped: no reference inputs connected."
            logger.info(headline)
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

        if not (image_urls or video_urls or audio_urls):
            # Inputs were connected but every upload failed — surface that distinctly.
            details = "; ".join(upload_failures) if upload_failures else "no upload error captured"
            headline = (
                f"Seedance reference-to-video failed: {connected_total} input(s) connected "
                f"but all uploads failed. Details: {details}"
            )
            logger.error(headline)
            push_node_status(unique_id, headline, event_log)
            return (None, "", config, finalize_status(headline, event_log))

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


# ---------------------------------------------------------------------------
# OpenAI GPT Image 2 — edit (via FAL)
# ---------------------------------------------------------------------------


GPT_IMAGE_2_EDIT_ENDPOINT = "openai/gpt-image-2/edit"
GPT_IMAGE_2_T2I_ENDPOINT = "openai/gpt-image-2/text-to-image"
GPT_IMAGE_2_SIZE_CHOICES = [
    "auto",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
GPT_IMAGE_2_QUALITY_CHOICES = ["auto", "low", "medium", "high"]
GPT_IMAGE_2_FORMAT_CHOICES = ["png", "jpeg", "webp"]
MAX_GPT_IMAGE_2_INPUTS = 10


def _mask_to_pil(mask_tensor) -> Image.Image:
    """Convert a ComfyUI MASK tensor [B,H,W] (or [H,W]) into an L-mode PIL image
    of the first frame. Values are scaled 0..255."""
    if mask_tensor.dim() == 3:
        m = mask_tensor[0]
    else:
        m = mask_tensor
    arr = (m.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _upload_pil(pil_image, suffix: str = ".png", fmt: str = "PNG"):
    """Save a PIL image to a temp file and upload it to FAL. Returns the URL or None."""
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        pil_image.save(temp_path, format=fmt)
        return FalConfig().get_client().upload_file(temp_path)
    except Exception as e:
        logger.error("Error uploading PIL image: %s", e)
        return None
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _download_image_to_tensor(url: str):
    """Download a URL (or decode a data: URI) into a [1, H, W, 3] float32 tensor."""
    if url.startswith("data:"):
        import base64
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


def _batch_align_images(tensors):
    """Cat a list of [1,H,W,3] tensors, resizing later ones to match the first
    via common_upscale (matches the rest of the Soze pack's batching convention)."""
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


class Soze_FALGPTImage2Edit:
    """OpenAI GPT Image 2 image edit, via FAL (endpoint: openai/gpt-image-2/edit).

    Connect 1-10 reference images (image_1..image_10). Each image is uploaded to
    FAL and its URL is passed in `image_urls`. Optionally provide a MASK to send
    as `mask_url`. The result image(s) are downloaded and returned as an IMAGE
    batch alongside their URLs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (GPT_IMAGE_2_SIZE_CHOICES, {"default": "auto", "tooltip": "'auto' infers size from input images."}),
            "quality": (GPT_IMAGE_2_QUALITY_CHOICES, {"default": "high"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "output_format": (GPT_IMAGE_2_FORMAT_CHOICES, {"default": "png"}),
        }
        optional = {}
        for i in range(1, MAX_GPT_IMAGE_2_INPUTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i}. Overrides image_batch when connected."})
        optional["image_batch"] = ("IMAGE", {"tooltip": "Optional IMAGE batch. Each frame is uploaded as a reference. Ignored if any image_N slot is connected."})
        optional["mask"] = ("MASK", {"tooltip": "Optional mask. Sent as mask_url."})
        optional["sync_mode"] = ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "status")
    FUNCTION = "edit"
    CATEGORY = "FAL/ImageGeneration"

    def edit(self, prompt, image_size, quality, num_images, output_format,
             image_batch=None, mask=None, sync_mode=False, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: {image_size} | {quality} | n={num_images} | {output_format}")

        if _is_blank(prompt):
            headline = "GPT Image 2 edit skipped: prompt is blank."
            logger.info(headline)
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        # 1) Individual image_N slots take precedence over image_batch.
        slot_pairs: list[tuple[str, "torch.Tensor"]] = []
        for i in range(1, MAX_GPT_IMAGE_2_INPUTS + 1):
            tensor = slots.get(f"image_{i}")
            if tensor is not None:
                slot_pairs.append((f"image_{i}", tensor[0:1]))

        # 2) Otherwise, fan out the image_batch.
        if slot_pairs:
            if image_batch is not None:
                push_node_status(unique_id, "Note: image_batch is ignored because individual image_N slot(s) are connected.", log)
            source_label = f"{len(slot_pairs)} individual image slot(s)"
        elif image_batch is not None and image_batch.shape[0] > 0:
            for i in range(image_batch.shape[0]):
                slot_pairs.append((f"batch[{i}]", image_batch[i:i+1]))
            source_label = f"image_batch ({image_batch.shape[0]} frame(s))"
        else:
            headline = "GPT Image 2 edit skipped: no reference images connected (no image_N slot and no image_batch)."
            logger.info(headline)
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        # Upload reference images.
        push_node_status(unique_id, f"Uploading {len(slot_pairs)} reference image(s) from {source_label}...", log)
        image_urls = []
        upload_failures = []
        for label, tensor in slot_pairs:
            try:
                url = ImageUtils.upload_image(tensor)
            except Exception as e:
                logger.exception("GPT Image 2: %s upload raised", label)
                upload_failures.append(f"{label}: {e!r}")
                continue
            if url:
                image_urls.append(url)
            else:
                upload_failures.append(f"{label}: uploader returned None (see ComfyUI log)")

        if not image_urls:
            details = "; ".join(upload_failures) or "no upload error captured"
            headline = f"GPT Image 2 edit failed: every reference image upload failed. {details}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        # Optional mask.
        mask_url = None
        if mask is not None:
            push_node_status(unique_id, "Uploading mask...", log)
            try:
                mask_pil = _mask_to_pil(mask)
                mask_url = _upload_pil(mask_pil)
            except Exception as e:
                logger.exception("GPT Image 2: mask upload raised")
                push_node_status(unique_id, f"Mask upload failed, continuing without mask: {e!r}", log)
                mask_url = None
            if mask_url is None:
                push_node_status(unique_id, "Note: mask uploader returned no URL; proceeding without mask.", log)

        arguments = {
            "prompt": prompt.strip(),
            "image_urls": image_urls,
            "image_size": image_size,
            "quality": quality,
            "num_images": int(num_images),
            "output_format": output_format,
        }
        if mask_url:
            arguments["mask_url"] = mask_url
        if sync_mode:
            arguments["sync_mode"] = True

        push_node_status(unique_id, f"Submitting to {GPT_IMAGE_2_EDIT_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(GPT_IMAGE_2_EDIT_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("GPT Image 2 edit failed")
            headline = f"ERROR ({GPT_IMAGE_2_EDIT_ENDPOINT}): {e!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        out_files = result.get("images") or []
        result_urls = []
        for entry in out_files:
            if isinstance(entry, dict):
                u = entry.get("url")
                if u:
                    result_urls.append(u)
            elif isinstance(entry, str):
                result_urls.append(entry)

        if not result_urls:
            headline = f"ERROR ({GPT_IMAGE_2_EDIT_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        push_node_status(unique_id, f"Downloading {len(result_urls)} result image(s)...", log)
        tensors = []
        for i, u in enumerate(result_urls):
            try:
                tensors.append(_download_image_to_tensor(u))
            except Exception as e:
                logger.exception("GPT Image 2: failed to load result image %s", u)
                push_node_status(unique_id, f"Skip result {i+1}: {e!r}", log)

        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, finalize_status(headline, log))

        image_batch = _batch_align_images(tensors)

        if upload_failures:
            push_node_status(unique_id, f"Note: {len(upload_failures)} reference upload(s) failed: " + "; ".join(upload_failures), log)

        urls_str = "\n".join(result_urls)
        headline = f"OK ({GPT_IMAGE_2_EDIT_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch.shape[2]}x{image_batch.shape[1]}"
        push_node_status(unique_id, headline, log)
        return (image_batch, urls_str, len(tensors), finalize_status(headline, log))


class Soze_FALGPTImage2:
    """OpenAI GPT Image 2 via FAL — text-to-image, or image edit when reference
    images are connected.

      - No images connected  -> openai/gpt-image-2/text-to-image (prompt only)
      - Any image connected   -> openai/gpt-image-2/edit (prompt + image_urls,
                                  with an optional mask)

    Connect references via the individual image_N slots and/or an image_batch.
    The result image(s) are downloaded and returned as an IMAGE batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (GPT_IMAGE_2_SIZE_CHOICES, {"default": "auto", "tooltip": "'auto' infers size (from input images in edit mode)."}),
            "quality": (GPT_IMAGE_2_QUALITY_CHOICES, {"default": "high"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "output_format": (GPT_IMAGE_2_FORMAT_CHOICES, {"default": "png"}),
        }
        optional = {}
        for i in range(1, MAX_GPT_IMAGE_2_INPUTS + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i}. Connecting any image switches to edit mode. Overrides image_batch."})
        optional["image_batch"] = ("IMAGE", {"tooltip": "Optional IMAGE batch. Each frame is uploaded as a reference. Ignored if any image_N slot is connected."})
        optional["mask"] = ("MASK", {"tooltip": "Optional mask (edit mode only). Sent as mask_url."})
        optional["sync_mode"] = ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/ImageGeneration"

    def generate(self, prompt, image_size, quality, num_images, output_format,
                 image_batch=None, mask=None, sync_mode=False, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: {image_size} | {quality} | n={num_images} | {output_format}")

        if _is_blank(prompt):
            headline = "GPT Image 2 skipped: prompt is blank."
            logger.info(headline)
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        # Gather reference images: individual image_N slots take precedence,
        # otherwise fan out image_batch.
        slot_pairs: list[tuple[str, "torch.Tensor"]] = []
        for i in range(1, MAX_GPT_IMAGE_2_INPUTS + 1):
            tensor = slots.get(f"image_{i}")
            if tensor is not None:
                slot_pairs.append((f"image_{i}", tensor[0:1]))

        if slot_pairs:
            if image_batch is not None:
                push_node_status(unique_id, "Note: image_batch is ignored because individual image_N slot(s) are connected.", log)
            source_label = f"{len(slot_pairs)} individual image slot(s)"
        elif image_batch is not None and image_batch.shape[0] > 0:
            for i in range(image_batch.shape[0]):
                slot_pairs.append((f"batch[{i}]", image_batch[i:i+1]))
            source_label = f"image_batch ({image_batch.shape[0]} frame(s))"
        else:
            slot_pairs = []
            source_label = None

        edit_mode = len(slot_pairs) > 0

        arguments = {
            "prompt": prompt.strip(),
            "image_size": image_size,
            "quality": quality,
            "num_images": int(num_images),
            "output_format": output_format,
        }
        if sync_mode:
            arguments["sync_mode"] = True

        upload_failures = []
        if edit_mode:
            endpoint = GPT_IMAGE_2_EDIT_ENDPOINT
            push_node_status(unique_id, f"Edit mode: uploading {len(slot_pairs)} reference image(s) from {source_label}...", log)
            image_urls = []
            for label, tensor in slot_pairs:
                try:
                    url = ImageUtils.upload_image(tensor)
                except Exception as e:
                    logger.exception("GPT Image 2: %s upload raised", label)
                    upload_failures.append(f"{label}: {e!r}")
                    continue
                if url:
                    image_urls.append(url)
                else:
                    upload_failures.append(f"{label}: uploader returned None (see ComfyUI log)")

            if not image_urls:
                details = "; ".join(upload_failures) or "no upload error captured"
                headline = f"GPT Image 2 edit failed: every reference image upload failed. {details}"
                push_node_status(unique_id, headline, log)
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

            arguments["image_urls"] = image_urls

            # Optional mask (edit mode only).
            if mask is not None:
                push_node_status(unique_id, "Uploading mask...", log)
                try:
                    mask_url = _upload_pil(_mask_to_pil(mask))
                except Exception as e:
                    logger.exception("GPT Image 2: mask upload raised")
                    push_node_status(unique_id, f"Mask upload failed, continuing without mask: {e!r}", log)
                    mask_url = None
                if mask_url:
                    arguments["mask_url"] = mask_url
                else:
                    push_node_status(unique_id, "Note: mask uploader returned no URL; proceeding without mask.", log)
        else:
            endpoint = GPT_IMAGE_2_T2I_ENDPOINT
            push_node_status(unique_id, "Text-to-image mode (no reference images connected).", log)

        push_node_status(unique_id, f"Submitting to {endpoint}...", log)
        try:
            result = ApiHandler.submit_and_get_result(endpoint, arguments)
        except Exception as e:
            logger.exception("GPT Image 2 failed")
            headline = f"ERROR ({endpoint}): {e!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        out_files = result.get("images") or []
        result_urls = []
        for entry in out_files:
            if isinstance(entry, dict):
                u = entry.get("url")
                if u:
                    result_urls.append(u)
            elif isinstance(entry, str):
                result_urls.append(entry)

        if not result_urls:
            headline = f"ERROR ({endpoint}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, finalize_status(headline, log))

        push_node_status(unique_id, f"Downloading {len(result_urls)} result image(s)...", log)
        tensors = []
        for i, u in enumerate(result_urls):
            try:
                tensors.append(_download_image_to_tensor(u))
            except Exception as e:
                logger.exception("GPT Image 2: failed to load result image %s", u)
                push_node_status(unique_id, f"Skip result {i+1}: {e!r}", log)

        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, finalize_status(headline, log))

        images_out = _batch_align_images(tensors)

        if upload_failures:
            push_node_status(unique_id, f"Note: {len(upload_failures)} reference upload(s) failed: " + "; ".join(upload_failures), log)

        mode_tag = "edit" if edit_mode else "t2i"
        urls_str = "\n".join(result_urls)
        headline = f"OK ({endpoint}, {mode_tag}): {len(tensors)}/{len(result_urls)} image(s), size={images_out.shape[2]}x{images_out.shape[1]}"
        push_node_status(unique_id, headline, log)
        return (images_out, urls_str, len(tensors), finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Kling O1 — shared helpers for all four flavors
# ---------------------------------------------------------------------------


KLING_TIER_CHOICES = ["standard", "pro"]
KLING_DURATION_CHOICES = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
KLING_ASPECT_CHOICES = ["auto", "16:9", "9:16", "1:1"]
KLING_SHOT_TYPE_CHOICES = ["customize", "intelligent"]
# o3 caps reference images at 4 across all kinds (down from o1's 7).
KLING_MAX_IMAGES = 4


def _kling_endpoint(tier: str, kind: str) -> str:
    """Build the Kling O3 endpoint id.

    fal.ai's URL convention for Kling O3 is symmetric between tiers:
        Standard: fal-ai/kling-video/o3/standard/{kind}
        Pro:      fal-ai/kling-video/o3/pro/{kind}

    kind: 'image-to-video' | 'reference-to-video' | 'video-to-video/edit' | 'video-to-video/reference'
    """
    return f"fal-ai/kling-video/o3/{tier}/{kind}"


def _extract_fal_error_detail(exc: Exception) -> str:
    """Best-effort extraction of fal.ai's actual error body + request id from an exception.

    fal_client's exceptions carry a `.response` (requests.Response). The body is usually
    a small JSON like `{"detail": "Internal Server Error"}` and the headers carry the
    request id, both of which are far more useful than the raw repr."""
    parts = []
    response = getattr(exc, "response", None)
    if response is not None:
        body = ""
        try:
            body = response.text
        except Exception:
            try:
                body = response.content.decode("utf-8", "replace")
            except Exception:
                body = ""
        if body:
            parts.append(f"body={body.strip()[:500]}")
        try:
            req_id = response.headers.get("x-fal-request-id")
            if req_id:
                parts.append(f"x-fal-request-id={req_id}")
        except Exception:
            pass
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    if not parts:
        return repr(exc)
    return f"{type(exc).__name__}: " + " | ".join(parts)


async def _call_kling(endpoint: str, arguments: dict, unique_id=None, event_log: EventLog | None = None):
    """Submit to a Kling O3 endpoint and download the resulting video.

    Returns (video, video_url, status_string). All four Kling endpoints share
    the same output shape: {"video": {"url": ..., ...}}.
    """
    if event_log is None:
        event_log = EventLog()
    push_node_status(unique_id, f"Submitting to {endpoint}...", event_log)
    try:
        result = ApiHandler.submit_and_get_result(endpoint, arguments)
    except Exception as e:
        logger.exception("Kling O3 generation failed (endpoint=%s)", endpoint)
        headline = f"ERROR ({endpoint}): {_extract_fal_error_detail(e)}"
        push_node_status(unique_id, headline, event_log)
        return (None, "", finalize_status(headline, event_log))

    video_info = result.get("video") or {}
    video_url = video_info.get("url")
    if not video_url:
        headline = f"ERROR ({endpoint}): API did not return a video URL. Response: {result!r}"
        push_node_status(unique_id, headline, event_log)
        return (None, "", finalize_status(headline, event_log))

    logger.info("Kling O3 (%s) video URL: %s", endpoint, video_url)
    push_node_status(unique_id, "Downloading video...", event_log)
    try:
        video = await download_url_to_video_output(video_url)
    except Exception as e:
        logger.exception("Kling O3 video download failed (url=%s)", video_url)
        headline = f"ERROR downloading video ({endpoint}): {e!r}"
        push_node_status(unique_id, headline, event_log)
        return (None, video_url, finalize_status(headline, event_log))
    headline = f"OK ({endpoint}): url={video_url}"
    push_node_status(unique_id, headline, event_log)
    return (video, video_url, finalize_status(headline, event_log))


def _upload_kling_image_slots(slots, max_count, image_batch, unique_id, log):
    """Collect connected image_1..image_N slots OR an image_batch and upload each.

    image_N slots take precedence over image_batch (matching the GPT Image 2
    node's convention). Returns (urls, failures, source_label).
    """
    slot_pairs: list[tuple[str, "torch.Tensor"]] = []
    for i in range(1, max_count + 1):
        tensor = slots.get(f"image_{i}")
        if tensor is not None:
            slot_pairs.append((f"image_{i}", tensor[0:1]))

    if slot_pairs:
        if image_batch is not None:
            push_node_status(unique_id, "Note: image_batch is ignored because individual image_N slot(s) are connected.", log)
        source_label = f"{len(slot_pairs)} individual image slot(s)"
    elif image_batch is not None and image_batch.shape[0] > 0:
        # Cap batch frames at the API's max so we don't waste uploads.
        n = min(image_batch.shape[0], max_count)
        if image_batch.shape[0] > max_count:
            push_node_status(unique_id, f"image_batch has {image_batch.shape[0]} frames; using first {max_count} (API max).", log)
        for i in range(n):
            slot_pairs.append((f"batch[{i}]", image_batch[i:i+1]))
        source_label = f"image_batch ({n} frame(s))"
    else:
        return [], [], "no images"

    push_node_status(unique_id, f"Uploading {len(slot_pairs)} reference image(s) from {source_label}...", log)
    urls = []
    failures = []
    for label, tensor in slot_pairs:
        try:
            url = ImageUtils.upload_image(tensor)
        except Exception as e:
            logger.exception("Kling: %s upload raised", label)
            failures.append(f"{label}: {e!r}")
            continue
        if url:
            urls.append(url)
        else:
            failures.append(f"{label}: uploader returned None (see ComfyUI log)")
    return urls, failures, source_label


def _kling_image_input_dict(max_count: int = KLING_MAX_IMAGES) -> dict:
    """Build an `optional` dict containing image_1..image_N + image_batch."""
    optional: dict = {}
    for i in range(1, max_count + 1):
        optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image slot {i}. Overrides image_batch when connected. Cite as @Image{i} in the prompt."})
    optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {max_count} frames). Ignored if any image_N slot is connected."})
    return optional


def _try_upload_image(image_tensor, label, unique_id, log):
    """Upload a single IMAGE tensor (taking the first frame) with error capture."""
    if image_tensor is None:
        return None
    try:
        url = ImageUtils.upload_image(image_tensor[0:1])
    except Exception as e:
        logger.exception("Kling: %s upload raised", label)
        push_node_status(unique_id, f"{label} upload failed: {e!r}", log)
        return None
    if not url:
        push_node_status(unique_id, f"{label} upload returned no URL.", log)
    return url


# ---------------------------------------------------------------------------
# Kling O3 — Reference Image-to-Video (Reference I2V)
# ---------------------------------------------------------------------------


class Soze_FALKlingO3ReferenceToVideo:
    """Kling O3 reference-to-video. Connect up to 4 reference images (or an
    image_batch), an optional start/end frame, and a prompt that cites them as
    @Image1..@Image4. `tier` switches between Standard and Pro endpoints."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "tier": (KLING_TIER_CHOICES, {"default": "standard"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite uploaded images as @Image1..@Image4 in the prompt."}),
            "duration": (KLING_DURATION_CHOICES, {"default": "5"}),
            "aspect_ratio": (KLING_ASPECT_CHOICES, {"default": "16:9"}),
            "shot_type": (KLING_SHOT_TYPE_CHOICES, {"default": "customize"}),
            "generate_audio": ("BOOLEAN", {"default": False}),
        }
        optional = {
            "start_image": ("IMAGE", {"tooltip": "Optional first frame."}),
            "end_image": ("IMAGE", {"tooltip": "Optional last frame."}),
        }
        optional.update(_kling_image_input_dict())
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, tier, prompt, duration, aspect_ratio, shot_type, generate_audio,
                       start_image=None, end_image=None, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: kling-o3 {tier} reference-to-video | duration={duration}s | aspect={aspect_ratio} | shot={shot_type} | audio={generate_audio}")

        if _is_blank(prompt):
            headline = "Kling O3 reference-to-video skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        urls, failures, _ = _upload_kling_image_slots(slots, KLING_MAX_IMAGES, image_batch, unique_id, log)
        start_url = _try_upload_image(start_image, "start_image", unique_id, log)
        end_url = _try_upload_image(end_image, "end_image", unique_id, log)

        if not urls and not start_url and not end_url:
            details = "; ".join(failures) if failures else "no reference images, no start/end frames"
            headline = f"Kling O3 reference-to-video skipped: nothing to send ({details})"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        endpoint = _kling_endpoint(tier, "reference-to-video")
        arguments = {
            "prompt": prompt.strip(),
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "shot_type": shot_type,
            "generate_audio": bool(generate_audio),
        }
        if urls:
            arguments["image_urls"] = urls
        if start_url:
            arguments["start_image_url"] = start_url
        if end_url:
            arguments["end_image_url"] = end_url
        if failures:
            push_node_status(unique_id, f"Partial uploads — {len(failures)} failed: " + "; ".join(failures), log)

        return await _call_kling(endpoint, arguments, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Kling O3 — First-Last Frame Video (FLFV) = image-to-video with start/end
# ---------------------------------------------------------------------------


class Soze_FALKlingO3FirstLastFrameVideo:
    """Kling O3 first-frame to last-frame video (FLFV) — the o3 image-to-video
    endpoint with optional end frame. `tier` switches Standard / Pro.

    Note: the o3 image-to-video API uses `image_url` for the start frame
    (not `start_image_url` like o1)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tier": (KLING_TIER_CHOICES, {"default": "standard"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Use @Image1 for start frame, @Image2 for end frame in the prompt."}),
                "start_image": ("IMAGE", {"tooltip": "Start frame. Required."}),
                "duration": (KLING_DURATION_CHOICES, {"default": "5"}),
                "shot_type": (KLING_SHOT_TYPE_CHOICES, {"default": "customize"}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "Optional end frame."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, tier, prompt, start_image, duration, shot_type, generate_audio,
                       end_image=None, unique_id=None):
        log = EventLog()
        log.add(f"Config: kling-o3 {tier} FLFV | duration={duration}s | shot={shot_type} | audio={generate_audio}")

        if start_image is None or _is_blank(prompt):
            headline = "Kling O3 FLFV skipped: prompt or start_image missing."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        try:
            push_node_status(unique_id, "Uploading start image...", log)
            args = {
                "prompt": prompt.strip(),
                "image_url": _upload_image_or_raise(start_image, "start image"),
                "duration": duration,
                "shot_type": shot_type,
                "generate_audio": bool(generate_audio),
            }
            if end_image is not None:
                push_node_status(unique_id, "Uploading end image...", log)
                args["end_image_url"] = _upload_image_or_raise(end_image, "end image")
        except Exception as e:
            logger.exception("Kling O3 FLFV setup failed")
            headline = f"ERROR: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        endpoint = _kling_endpoint(tier, "image-to-video")
        return await _call_kling(endpoint, args, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Kling O3 — Edit Video (video-to-video/edit)
# ---------------------------------------------------------------------------


class Soze_FALKlingO3EditVideo:
    """Kling O3 edit-video: transform a source video with natural language while
    preserving original motion and camera angles. Optional reference images can
    style the output. `tier` switches Standard / Pro."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "tier": (KLING_TIER_CHOICES, {"default": "standard"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite reference images as @Image1..@Image4."}),
            "video": ("VIDEO", {"tooltip": "Source video to edit. 3-10s, 720-2160px, max 200MB, mp4/mov."}),
            "shot_type": (KLING_SHOT_TYPE_CHOICES, {"default": "customize"}),
        }
        optional = {
            "keep_audio": ("BOOLEAN", {"default": True, "tooltip": "Preserve the original audio track."}),
        }
        optional.update(_kling_image_input_dict())
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, tier, prompt, video, shot_type, keep_audio=True, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: kling-o3 {tier} edit-video | shot={shot_type} | keep_audio={keep_audio}")

        if video is None or _is_blank(prompt):
            headline = "Kling O3 edit-video skipped: prompt or video missing."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        try:
            push_node_status(unique_id, "Uploading source video...", log)
            video_url = ImageUtils.upload_video(video)
            if not video_url:
                raise RuntimeError("Source video upload returned no URL.")
        except Exception as e:
            logger.exception("Kling O3 edit-video: video upload failed")
            headline = f"ERROR uploading source video: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        urls, failures, _ = _upload_kling_image_slots(slots, KLING_MAX_IMAGES, image_batch, unique_id, log)

        endpoint = _kling_endpoint(tier, "video-to-video/edit")
        arguments = {
            "prompt": prompt.strip(),
            "video_url": video_url,
            "keep_audio": bool(keep_audio),
            "shot_type": shot_type,
        }
        if urls:
            arguments["image_urls"] = urls
        if failures:
            push_node_status(unique_id, f"Partial reference uploads — {len(failures)} failed: " + "; ".join(failures), log)

        return await _call_kling(endpoint, arguments, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Kling O3 — Reference Video-to-Video (video-to-video/reference)
# ---------------------------------------------------------------------------


class Soze_FALKlingO3ReferenceVideoToVideo:
    """Kling O3 reference video-to-video: reshape a source video using reference
    images for style/identity consistency. `tier` switches Standard / Pro."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "tier": (KLING_TIER_CHOICES, {"default": "standard"}),
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite reference images as @Image1..@Image4."}),
            "video": ("VIDEO", {"tooltip": "Source video. 3-10s, 720-2160px, max 200MB, mp4/mov."}),
            "duration": (KLING_DURATION_CHOICES, {"default": "5"}),
            "aspect_ratio": (KLING_ASPECT_CHOICES, {"default": "auto"}),
            "shot_type": (KLING_SHOT_TYPE_CHOICES, {"default": "customize"}),
        }
        optional = {
            "keep_audio": ("BOOLEAN", {"default": True, "tooltip": "Preserve the original audio track."}),
        }
        optional.update(_kling_image_input_dict())
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, tier, prompt, video, duration, aspect_ratio, shot_type, keep_audio=True, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: kling-o3 {tier} reference-V2V | duration={duration}s | aspect={aspect_ratio} | shot={shot_type} | keep_audio={keep_audio}")

        if video is None or _is_blank(prompt):
            headline = "Kling O3 reference-V2V skipped: prompt or video missing."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        try:
            push_node_status(unique_id, "Uploading source video...", log)
            video_url = ImageUtils.upload_video(video)
            if not video_url:
                raise RuntimeError("Source video upload returned no URL.")
        except Exception as e:
            logger.exception("Kling O3 reference-V2V: video upload failed")
            headline = f"ERROR uploading source video: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        urls, failures, _ = _upload_kling_image_slots(slots, KLING_MAX_IMAGES, image_batch, unique_id, log)

        endpoint = _kling_endpoint(tier, "video-to-video/reference")
        arguments = {
            "prompt": prompt.strip(),
            "video_url": video_url,
            "keep_audio": bool(keep_audio),
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "shot_type": shot_type,
        }
        if urls:
            arguments["image_urls"] = urls
        if failures:
            push_node_status(unique_id, f"Partial reference uploads — {len(failures)} failed: " + "; ".join(failures), log)

        return await _call_kling(endpoint, arguments, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Topaz — Video Upscale (via FAL)
# ---------------------------------------------------------------------------


TOPAZ_UPSCALE_VIDEO_ENDPOINT = "fal-ai/topaz/upscale/video"
TOPAZ_MODEL_CHOICES = [
    "Proteus",
    "Artemis HQ", "Artemis MQ", "Artemis LQ",
    "Nyx", "Nyx Fast", "Nyx XL", "Nyx HF",
    "Gaia HQ", "Gaia CG", "Gaia 2",
    "Starlight Precise 1", "Starlight Precise 2", "Starlight Precise 2.5",
    "Starlight HQ", "Starlight Mini", "Starlight Sharp",
    "Starlight Fast 1", "Starlight Fast 2",
]


class Soze_FALTopazUpscaleVideo:
    """Topaz video upscale via FAL (endpoint: fal-ai/topaz/upscale/video).

    Upscales a source VIDEO by `upscale_factor`. Optional frame interpolation
    (set target_fps > 0), artifact/noise/halo/grain controls, detail recovery,
    and H.264 output. Enhancement sliders left at -1 use the model's default.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # -1.0 sentinel on the float sliders => omit the field (use model default).
        model_slider = {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "Source video to upscale."}),
                "model": (TOPAZ_MODEL_CHOICES, {"default": "Proteus"}),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25, "tooltip": "e.g. 2.0 doubles width and height."}),
            },
            "optional": {
                "target_fps": ("INT", {"default": 0, "min": 0, "max": 120, "step": 1, "tooltip": "0 = off. If > 0, enables frame interpolation to this FPS."}),
                "compression": ("FLOAT", {**model_slider, "tooltip": "Compression artifact removal (0-1). -1 = model default."}),
                "noise": ("FLOAT", {**model_slider, "tooltip": "Noise reduction (0-1). -1 = model default."}),
                "halo": ("FLOAT", {**model_slider, "tooltip": "Halo reduction (0-1). -1 = model default."}),
                "grain": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 0.1, "step": 0.005, "tooltip": "Film grain amount (0-0.1). -1 = model default."}),
                "recover_detail": ("FLOAT", {**model_slider, "tooltip": "Recover original detail (0-1). -1 = model default."}),
                "H264_output": ("BOOLEAN", {"default": False, "tooltip": "Output H.264 instead of the default H.265."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "upscale"
    CATEGORY = "FAL/VideoGeneration"

    async def upscale(self, video, model, upscale_factor, target_fps=0,
                      compression=-1.0, noise=-1.0, halo=-1.0, grain=-1.0,
                      recover_detail=-1.0, H264_output=False, unique_id=None):
        log = EventLog()
        log.add(f"Config: topaz upscale | model={model} | x{upscale_factor} | target_fps={target_fps} | H264={H264_output}")

        if video is None:
            headline = "Topaz upscale skipped: no video connected."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        try:
            push_node_status(unique_id, "Uploading source video...", log)
            video_url = ImageUtils.upload_video(video)
            if not video_url:
                raise RuntimeError("Source video upload returned no URL.")
        except Exception as e:
            logger.exception("Topaz upscale: video upload failed")
            headline = f"ERROR uploading source video: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        arguments = {
            "video_url": video_url,
            "model": model,
            "upscale_factor": float(upscale_factor),
            "H264_output": bool(H264_output),
        }
        # Frame interpolation only when a positive target_fps is provided.
        if isinstance(target_fps, int) and target_fps > 0:
            arguments["target_fps"] = int(target_fps)
        # Enhancement sliders: only send when the user overrode the -1 sentinel.
        for name, value in (
            ("compression", compression),
            ("noise", noise),
            ("halo", halo),
            ("grain", grain),
            ("recover_detail", recover_detail),
        ):
            if value is not None and float(value) >= 0.0:
                arguments[name] = float(value)

        return await _call_kling(TOPAZ_UPSCALE_VIDEO_ENDPOINT, arguments, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# MiniMax H3 — Reference-to-Video (via FAL)
# ---------------------------------------------------------------------------


MINIMAX_H3_REF_ENDPOINT = "minimax/h3/reference-to-video"
MINIMAX_H3_RESOLUTION_CHOICES = ["2K", "768P"]
MINIMAX_H3_ASPECT_CHOICES = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]


def _upload_typed_ref_slots(slots, prefix, count, uploader):
    """Upload connected {prefix}_1..{prefix}_N slots in order. Returns (urls, failures)."""
    urls = []
    failures = []
    for i, val in _collect_ordered(slots, prefix, count):
        if val is None:
            continue
        try:
            u = uploader(val)
        except Exception as e:
            logger.exception("MiniMax H3: %s_%d upload raised", prefix, i)
            failures.append(f"{prefix}_{i}: {e!r}")
            continue
        if u:
            urls.append(u)
        else:
            failures.append(f"{prefix}_{i}: uploader returned None (check FAL config)")
    return urls, failures


class Soze_FALMiniMaxH3ReferenceToVideo:
    """MiniMax H3 reference-to-video via FAL (endpoint: minimax/h3/reference-to-video).

    Cite uploaded references in the prompt as 'Image 1'..'Image 9',
    'Video 1'..'Video 3', 'Audio 1'..'Audio 3'. Connect images via image_1..
    image_9 (individual slots) or image_batch; videos/audios via their own slots.
    Everything is uploaded to FAL, then the resulting MP4 is downloaded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite uploaded refs as 'Image 1'..'Image 9', 'Video 1'..'Video 3', 'Audio 1'..'Audio 3'."}),
            "duration": ("INT", {"default": 5, "min": 5, "max": 15, "step": 1, "tooltip": "Clip length in seconds (5-15)."}),
            "resolution": (MINIMAX_H3_RESOLUTION_CHOICES, {"default": "2K"}),
            "aspect_ratio": (MINIMAX_H3_ASPECT_CHOICES, {"default": "adaptive", "tooltip": "'adaptive' matches the reference frames."}),
        }
        optional = {}
        for i in range(1, MAX_REFERENCE_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image {i} (cite as 'Image {i}'). Overrides image_batch when connected."})
        optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {MAX_REFERENCE_IMAGES}). Ignored if any image_N slot is connected."})
        for i in range(1, MAX_REFERENCE_VIDEOS + 1):
            optional[f"video_{i}"] = ("VIDEO", {"tooltip": f"Reference video {i} (cite as 'Video {i}'). 2-15s each, combined <= 15s."})
        for i in range(1, MAX_REFERENCE_AUDIOS + 1):
            optional[f"audio_{i}"] = ("AUDIO", {"tooltip": f"Reference audio {i} (cite as 'Audio {i}'). Requires >= 1 image or video."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_url", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, duration, resolution, aspect_ratio,
                       image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: minimax-h3 ref2video | {resolution} | {aspect_ratio} | {duration}s")

        if _is_blank(prompt):
            headline = "MiniMax H3 reference-to-video skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        # Upload references. Images use the shared slot/batch helper; video/audio
        # use their own typed slots.
        push_node_status(unique_id, "Uploading references...", log)
        image_urls, image_failures, _ = _upload_kling_image_slots(slots, MAX_REFERENCE_IMAGES, image_batch, unique_id, log)
        video_urls, video_failures = _upload_typed_ref_slots(slots, "video", MAX_REFERENCE_VIDEOS, ImageUtils.upload_video)
        audio_urls, audio_failures = _upload_typed_ref_slots(slots, "audio", MAX_REFERENCE_AUDIOS, ImageUtils.upload_audio)

        failures = image_failures + video_failures + audio_failures
        total_refs = len(image_urls) + len(video_urls) + len(audio_urls)
        if total_refs == 0 and failures:
            headline = "MiniMax H3 reference-to-video failed: reference(s) connected but all uploads failed. " + "; ".join(failures)
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        if audio_urls and not (image_urls or video_urls):
            push_node_status(unique_id, "Warning: reference audio requires >= 1 image or video; the API may reject this.", log)

        arguments = {
            "prompt": prompt.strip(),
            "duration": int(duration),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        if image_urls:
            arguments["reference_image_urls"] = image_urls
        if video_urls:
            arguments["reference_video_urls"] = video_urls
        if audio_urls:
            arguments["reference_audio_urls"] = audio_urls

        if failures:
            push_node_status(unique_id, f"Partial uploads — {len(failures)} failed: " + "; ".join(failures), log)
        log.add(f"Uploaded refs: {len(image_urls)} image, {len(video_urls)} video, {len(audio_urls)} audio")

        return await _call_kling(MINIMAX_H3_REF_ENDPOINT, arguments, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# ByteDance Seedream v5 Lite — Edit + Text-to-Image (via FAL)
# ---------------------------------------------------------------------------


SEEDREAM5_EDIT_ENDPOINT = "fal-ai/bytedance/seedream/v5/lite/edit"
SEEDREAM5_T2I_ENDPOINT = "fal-ai/bytedance/seedream/v5/lite/text-to-image"
SEEDREAM5_IMAGE_SIZE_CHOICES = [
    "auto_2K",
    "auto_3K",
    "auto_4K",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
SEEDREAM5_EDIT_MAX_IMAGES = 10  # Seedream v5 edit accepts up to 10 input images.


def _seedream5_collect_results(result):
    """Pull (image_urls, seed) out of a Seedream response."""
    out_files = result.get("images") or []
    result_urls = []
    for entry in out_files:
        if isinstance(entry, dict):
            u = entry.get("url")
            if u:
                result_urls.append(u)
        elif isinstance(entry, str):
            result_urls.append(entry)
    seed_used = int(result.get("seed", 0) or 0)
    return result_urls, seed_used


def _seedream5_decode_results(result_urls, unique_id, log):
    """Download each result image URL into a [1,H,W,3] tensor."""
    push_node_status(unique_id, f"Downloading {len(result_urls)} result image(s)...", log)
    tensors = []
    for i, u in enumerate(result_urls):
        try:
            tensors.append(_download_image_to_tensor(u))
        except Exception as e:
            logger.exception("Seedream v5: failed to load result image %s", u)
            push_node_status(unique_id, f"Skip result {i+1}: {e!r}", log)
    return tensors


class Soze_FALSeedream5LiteEdit:
    """ByteDance Seedream v5 Lite — image edit (endpoint: fal-ai/bytedance/seedream/v5/lite/edit).

    Connect 1-10 input images via image_1..image_10 (individual slots take
    precedence) or feed an image_batch. The result image(s) are returned as an
    IMAGE batch alongside their URLs and the seed used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (SEEDREAM5_IMAGE_SIZE_CHOICES, {"default": "auto_2K"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "max_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "If >1, enables multi-image generation."}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
        }
        optional = {}
        for i in range(1, SEEDREAM5_EDIT_MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Input image slot {i}. Overrides image_batch when connected."})
        optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {SEEDREAM5_EDIT_MAX_IMAGES} frames). Ignored if any image_N slot is connected."})
        optional["sync_mode"] = ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "seed", "status")
    FUNCTION = "edit"
    CATEGORY = "FAL/ImageGeneration"

    def edit(self, prompt, image_size, num_images, max_images, enable_safety_checker,
             image_batch=None, sync_mode=False, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: seedream-v5-lite edit | size={image_size} | n={num_images} | max_images={max_images} | safety={enable_safety_checker}")

        if _is_blank(prompt):
            headline = "Seedream v5 Lite edit skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        # Reuse the Kling slot helper — same image_N / image_batch override convention.
        urls, failures, _ = _upload_kling_image_slots(slots, SEEDREAM5_EDIT_MAX_IMAGES, image_batch, unique_id, log)
        if not urls:
            details = "; ".join(failures) if failures else "no input images connected"
            headline = f"Seedream v5 Lite edit skipped: {details}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        arguments = {
            "prompt": prompt.strip(),
            "image_urls": urls,
            "image_size": image_size,
            "num_images": int(num_images),
            "max_images": int(max_images),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if sync_mode:
            arguments["sync_mode"] = True

        push_node_status(unique_id, f"Submitting to {SEEDREAM5_EDIT_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(SEEDREAM5_EDIT_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("Seedream v5 Lite edit failed")
            headline = f"ERROR ({SEEDREAM5_EDIT_ENDPOINT}): {_extract_fal_error_detail(e)}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        result_urls, seed_used = _seedream5_collect_results(result)
        if not result_urls:
            headline = f"ERROR ({SEEDREAM5_EDIT_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, seed_used, finalize_status(headline, log))

        tensors = _seedream5_decode_results(result_urls, unique_id, log)
        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, seed_used, finalize_status(headline, log))

        image_batch_out = _batch_align_images(tensors)
        if failures:
            push_node_status(unique_id, f"Note: {len(failures)} input upload(s) failed: " + "; ".join(failures), log)

        urls_str = "\n".join(result_urls)
        headline = f"OK ({SEEDREAM5_EDIT_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}, seed={seed_used}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), seed_used, finalize_status(headline, log))


class Soze_FALSeedream5LiteTextToImage:
    """ByteDance Seedream v5 Lite — text-to-image (endpoint: fal-ai/bytedance/seedream/v5/lite/text-to-image)."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (SEEDREAM5_IMAGE_SIZE_CHOICES, {"default": "auto_2K"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "max_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "If >1, enables multi-image generation."}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
        }
        optional = {
            "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."}),
            "return_byteplus_urls": ("BOOLEAN", {"default": False, "tooltip": "If True, returns ByteDance-trusted URLs that expire in 24 hours."}),
        }
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "seed", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/ImageGeneration"

    def generate(self, prompt, image_size, num_images, max_images, enable_safety_checker,
                 sync_mode=False, return_byteplus_urls=False, unique_id=None):
        log = EventLog()
        log.add(f"Config: seedream-v5-lite t2i | size={image_size} | n={num_images} | max_images={max_images} | safety={enable_safety_checker}")

        if _is_blank(prompt):
            headline = "Seedream v5 Lite text-to-image skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        arguments = {
            "prompt": prompt.strip(),
            "image_size": image_size,
            "num_images": int(num_images),
            "max_images": int(max_images),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if sync_mode:
            arguments["sync_mode"] = True
        if return_byteplus_urls:
            arguments["return_byteplus_urls"] = True

        push_node_status(unique_id, f"Submitting to {SEEDREAM5_T2I_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(SEEDREAM5_T2I_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("Seedream v5 Lite text-to-image failed")
            headline = f"ERROR ({SEEDREAM5_T2I_ENDPOINT}): {_extract_fal_error_detail(e)}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        result_urls, seed_used = _seedream5_collect_results(result)
        if not result_urls:
            headline = f"ERROR ({SEEDREAM5_T2I_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, seed_used, finalize_status(headline, log))

        tensors = _seedream5_decode_results(result_urls, unique_id, log)
        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, seed_used, finalize_status(headline, log))

        image_batch_out = _batch_align_images(tensors)
        urls_str = "\n".join(result_urls)
        headline = f"OK ({SEEDREAM5_T2I_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}, seed={seed_used}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), seed_used, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# ByteDance Seedream v5 Pro — Edit (via FAL)
# ---------------------------------------------------------------------------


SEEDREAM5_PRO_EDIT_ENDPOINT = "fal-ai/bytedance/seedream/v5/pro/edit"
# Pro edit presets per the FAL schema (auto_1K/auto_2K; no auto_3K/4K).
SEEDREAM5_PRO_IMAGE_SIZE_CHOICES = [
    "auto_2K",
    "auto_1K",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
SEEDREAM5_PRO_FORMAT_CHOICES = ["jpeg", "png"]
SEEDREAM5_PRO_EDIT_MAX_IMAGES = 10  # Seedream v5 Pro edit accepts up to 10 input images.


class Soze_FALSeedream5ProEdit:
    """ByteDance Seedream v5 Pro — image edit (endpoint: fal-ai/bytedance/seedream/v5/pro/edit).

    Connect 1-10 input images via image_1..image_10 (individual slots take
    precedence) or feed an image_batch. The result image(s) are returned as an
    IMAGE batch alongside their URLs and the seed used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (SEEDREAM5_PRO_IMAGE_SIZE_CHOICES, {"default": "auto_2K"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1, "tooltip": "Separate generations to run (1-6)."}),
            "output_format": (SEEDREAM5_PRO_FORMAT_CHOICES, {"default": "jpeg"}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
        }
        optional = {}
        for i in range(1, SEEDREAM5_PRO_EDIT_MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Input image slot {i}. Overrides image_batch when connected."})
        optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {SEEDREAM5_PRO_EDIT_MAX_IMAGES} frames). Ignored if any image_N slot is connected."})
        optional["sync_mode"] = ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "seed", "status")
    FUNCTION = "edit"
    CATEGORY = "FAL/ImageGeneration"

    def edit(self, prompt, image_size, num_images, output_format, enable_safety_checker,
             image_batch=None, sync_mode=False, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: seedream-v5-pro edit | size={image_size} | n={num_images} | fmt={output_format} | safety={enable_safety_checker}")

        if _is_blank(prompt):
            headline = "Seedream v5 Pro edit skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        # Reuse the shared slot helper — same image_N / image_batch override convention.
        urls, failures, _ = _upload_kling_image_slots(slots, SEEDREAM5_PRO_EDIT_MAX_IMAGES, image_batch, unique_id, log)
        if not urls:
            details = "; ".join(failures) if failures else "no input images connected"
            headline = f"Seedream v5 Pro edit skipped: {details}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        arguments = {
            "prompt": prompt.strip(),
            "image_urls": urls,
            "image_size": image_size,
            "num_images": int(num_images),
            "output_format": output_format,
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if sync_mode:
            arguments["sync_mode"] = True

        push_node_status(unique_id, f"Submitting to {SEEDREAM5_PRO_EDIT_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(SEEDREAM5_PRO_EDIT_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("Seedream v5 Pro edit failed")
            headline = f"ERROR ({SEEDREAM5_PRO_EDIT_ENDPOINT}): {_extract_fal_error_detail(e)}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        result_urls, seed_used = _seedream5_collect_results(result)
        if not result_urls:
            headline = f"ERROR ({SEEDREAM5_PRO_EDIT_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, seed_used, finalize_status(headline, log))

        tensors = _seedream5_decode_results(result_urls, unique_id, log)
        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, seed_used, finalize_status(headline, log))

        image_batch_out = _batch_align_images(tensors)
        if failures:
            push_node_status(unique_id, f"Note: {len(failures)} input upload(s) failed: " + "; ".join(failures), log)

        urls_str = "\n".join(result_urls)
        headline = f"OK ({SEEDREAM5_PRO_EDIT_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}, seed={seed_used}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), seed_used, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# ByteDance Seedream v4.5 — Edit + Text-to-Image (via FAL)
# ---------------------------------------------------------------------------


SEEDREAM45_EDIT_ENDPOINT = "fal-ai/bytedance/seedream/v4.5/edit"
SEEDREAM45_T2I_ENDPOINT = "fal-ai/bytedance/seedream/v4.5/text-to-image"
# v4.5 lacks auto_3K compared to v5 Lite; otherwise identical preset list.
SEEDREAM45_IMAGE_SIZE_CHOICES = [
    "auto_2K",
    "auto_4K",
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]
SEEDREAM45_EDIT_MAX_IMAGES = 10  # v4.5 accepts up to 10 input images.
SEEDREAM45_SEED_DEFAULT = 0  # 0 == "let FAL choose" (we omit the param).


class Soze_FALSeedream45Edit:
    """ByteDance Seedream v4.5 — image edit (endpoint: fal-ai/bytedance/seedream/v4.5/edit).

    Connect 1-10 input images via image_1..image_10 (individual slots take
    precedence) or feed an image_batch. The result image(s) are returned as an
    IMAGE batch alongside their URLs and the seed used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (SEEDREAM45_IMAGE_SIZE_CHOICES, {"default": "auto_2K"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "max_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "If >1, enables multi-image generation."}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": SEEDREAM45_SEED_DEFAULT, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = let FAL choose."}),
        }
        optional = {}
        for i in range(1, SEEDREAM45_EDIT_MAX_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Input image slot {i}. Overrides image_batch when connected."})
        optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {SEEDREAM45_EDIT_MAX_IMAGES} frames). Ignored if any image_N slot is connected."})
        optional["sync_mode"] = ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "seed", "status")
    FUNCTION = "edit"
    CATEGORY = "FAL/ImageGeneration"

    def edit(self, prompt, image_size, num_images, max_images, enable_safety_checker, seed,
             image_batch=None, sync_mode=False, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: seedream-v4.5 edit | size={image_size} | n={num_images} | max_images={max_images} | safety={enable_safety_checker} | seed={seed or 'auto'}")

        if _is_blank(prompt):
            headline = "Seedream v4.5 edit skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        urls, failures, _ = _upload_kling_image_slots(slots, SEEDREAM45_EDIT_MAX_IMAGES, image_batch, unique_id, log)
        if not urls:
            details = "; ".join(failures) if failures else "no input images connected"
            headline = f"Seedream v4.5 edit skipped: {details}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        arguments = {
            "prompt": prompt.strip(),
            "image_urls": urls,
            "image_size": image_size,
            "num_images": int(num_images),
            "max_images": int(max_images),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if isinstance(seed, int) and seed > 0:
            arguments["seed"] = seed
        if sync_mode:
            arguments["sync_mode"] = True

        push_node_status(unique_id, f"Submitting to {SEEDREAM45_EDIT_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(SEEDREAM45_EDIT_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("Seedream v4.5 edit failed")
            headline = f"ERROR ({SEEDREAM45_EDIT_ENDPOINT}): {_extract_fal_error_detail(e)}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        result_urls, seed_used = _seedream5_collect_results(result)
        if not result_urls:
            headline = f"ERROR ({SEEDREAM45_EDIT_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, seed_used, finalize_status(headline, log))

        tensors = _seedream5_decode_results(result_urls, unique_id, log)
        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, seed_used, finalize_status(headline, log))

        image_batch_out = _batch_align_images(tensors)
        if failures:
            push_node_status(unique_id, f"Note: {len(failures)} input upload(s) failed: " + "; ".join(failures), log)

        urls_str = "\n".join(result_urls)
        headline = f"OK ({SEEDREAM45_EDIT_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}, seed={seed_used}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), seed_used, finalize_status(headline, log))


class Soze_FALSeedream45TextToImage:
    """ByteDance Seedream v4.5 — text-to-image (endpoint: fal-ai/bytedance/seedream/v4.5/text-to-image)."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "image_size": (SEEDREAM45_IMAGE_SIZE_CHOICES, {"default": "auto_2K"}),
            "num_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            "max_images": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "If >1, enables multi-image generation."}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
            "seed": ("INT", {"default": SEEDREAM45_SEED_DEFAULT, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = let FAL choose."}),
        }
        optional = {
            "sync_mode": ("BOOLEAN", {"default": False, "tooltip": "If True, FAL returns images as data URIs instead of public URLs."}),
        }
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "image_urls", "image_count", "seed", "status")
    FUNCTION = "generate"
    CATEGORY = "FAL/ImageGeneration"

    def generate(self, prompt, image_size, num_images, max_images, enable_safety_checker, seed,
                 sync_mode=False, unique_id=None):
        log = EventLog()
        log.add(f"Config: seedream-v4.5 t2i | size={image_size} | n={num_images} | max_images={max_images} | safety={enable_safety_checker} | seed={seed or 'auto'}")

        if _is_blank(prompt):
            headline = "Seedream v4.5 text-to-image skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        arguments = {
            "prompt": prompt.strip(),
            "image_size": image_size,
            "num_images": int(num_images),
            "max_images": int(max_images),
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if isinstance(seed, int) and seed > 0:
            arguments["seed"] = seed
        if sync_mode:
            arguments["sync_mode"] = True

        push_node_status(unique_id, f"Submitting to {SEEDREAM45_T2I_ENDPOINT}...", log)
        try:
            result = ApiHandler.submit_and_get_result(SEEDREAM45_T2I_ENDPOINT, arguments)
        except Exception as e:
            logger.exception("Seedream v4.5 text-to-image failed")
            headline = f"ERROR ({SEEDREAM45_T2I_ENDPOINT}): {_extract_fal_error_detail(e)}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, 0, finalize_status(headline, log))

        result_urls, seed_used = _seedream5_collect_results(result)
        if not result_urls:
            headline = f"ERROR ({SEEDREAM45_T2I_ENDPOINT}): API returned no images. Response: {result!r}"
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "", 0, seed_used, finalize_status(headline, log))

        tensors = _seedream5_decode_results(result_urls, unique_id, log)
        if not tensors:
            headline = "ERROR: every result image failed to load."
            push_node_status(unique_id, headline, log)
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), "\n".join(result_urls), 0, seed_used, finalize_status(headline, log))

        image_batch_out = _batch_align_images(tensors)
        urls_str = "\n".join(result_urls)
        headline = f"OK ({SEEDREAM45_T2I_ENDPOINT}): {len(tensors)}/{len(result_urls)} image(s), size={image_batch_out.shape[2]}x{image_batch_out.shape[1]}, seed={seed_used}"
        push_node_status(unique_id, headline, log)
        return (image_batch_out, urls_str, len(tensors), seed_used, finalize_status(headline, log))


