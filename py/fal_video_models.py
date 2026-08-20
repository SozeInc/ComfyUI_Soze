"""Additional FAL reference/image-to-video model nodes.

Each model returns {"video": {"url": ...}}, so they reuse fal.py's shared upload
helpers and the _call_kling submit+download helper (which also carries the
transient-error retry). Reference images come in via image_1..image_N slots
(individual slots win) or an image_batch, and are uploaded to FAL.

Models covered here (the ones not already in fal.py):
  - Seedance 2.5 reference-to-video          (bytedance/seedance-2.5/reference-to-video)
  - Gemini Omni Flash                        (google/gemini-omni-flash/reference-to-video)
  - Kling V3 Turbo Standard image-to-video   (fal-ai/kling-video/v3/turbo/standard/image-to-video)
  - Grok Imagine reference-to-video          (xai/grok-imagine-video/reference-to-video)
  - Happy Horse 1.1 reference-to-video       (alibaba/happy-horse/v1.1/reference-to-video)
  - Kling O3 4K reference-to-video           (fal-ai/kling-video/o3/4k/reference-to-video)
  - PixVerse C1 reference-to-video           (fal-ai/pixverse/c1/reference-to-video)
  - Kling O1 Pro reference-to-video          (fal-ai/kling-video/o1/reference-to-video)
  - Kling O1 Standard reference-to-video     (fal-ai/kling-video/o1/standard/reference-to-video)
  - Vidu Q3 reference-to-video mix           (fal-ai/vidu/q3/reference-to-video/mix)
  - Vidu Q1 reference-to-video               (fal-ai/vidu/q1/reference-to-video)
  - Mirage Avatar X reference-to-video       (mirage-api/avatar-x/reference-to-video)
"""

import logging

from .fal import (
    _call_kling,
    _upload_kling_image_slots,
    _upload_typed_ref_slots,
    _is_blank,
    MAX_REFERENCE_VIDEOS,
    MAX_REFERENCE_AUDIOS,
)
from .fal_utils import ImageUtils
from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_image_slots(optional, max_images, cite="Image"):
    """Add image_1..image_N slots + an image_batch to an optional-inputs dict."""
    for i in range(1, max_images + 1):
        optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"Reference image {i} (cite as @{cite}{i} in the prompt)."})
    optional["image_batch"] = ("IMAGE", {"tooltip": f"Optional IMAGE batch (capped at {max_images}). Ignored if any image_N slot is connected."})
    return optional


def _upload_single_image(img):
    """Upload the first frame of an IMAGE tensor to FAL. Returns URL or None."""
    if img is None:
        return None
    try:
        return ImageUtils.upload_image(img[0:1])
    except Exception:
        logger.exception("Single image upload failed")
        return None


async def _run_image_ref_video(endpoint, base_args, image_key, slots, image_batch,
                               max_images, unique_id, log, model_name, require_image=True,
                               extra_urls=None):
    """Upload image_N/image_batch, set them under `image_key`, and submit.

    `extra_urls` is an optional dict of {arg_key: url_or_list} already prepared
    (e.g. start/end frames) to merge into the request.
    """
    push_node_status(unique_id, "Uploading reference images...", log)
    image_urls, failures, _ = _upload_kling_image_slots(slots, max_images, image_batch, unique_id, log)

    if require_image and not image_urls:
        details = "; ".join(failures) if failures else "no reference images connected"
        headline = f"{model_name} skipped: {details}"
        push_node_status(unique_id, headline, log)
        return (None, "", finalize_status(headline, log))

    args = dict(base_args)
    if image_urls:
        args[image_key] = image_urls
    if extra_urls:
        for k, v in extra_urls.items():
            if v:
                args[k] = v
    if failures:
        push_node_status(unique_id, f"Partial uploads — {len(failures)} failed: " + "; ".join(failures), log)
    return await _call_kling(endpoint, args, unique_id=unique_id, event_log=log)


VIDEO_RETURN_TYPES = ("VIDEO", "STRING", "STRING")
VIDEO_RETURN_NAMES = ("video", "video_url", "status")


# ---------------------------------------------------------------------------
# Seedance 2.5 — reference-to-video
# ---------------------------------------------------------------------------


SEEDANCE25_ENDPOINT = "bytedance/seedance-2.5/reference-to-video"
SEEDANCE25_RESOLUTION = ["480p", "720p", "1080p"]
SEEDANCE25_DURATION = ["auto"] + [str(i) for i in range(4, 31)]
SEEDANCE25_ASPECT = ["auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
SEEDANCE25_MAX_IMAGES = 9


class Soze_FALSeedance25ReferenceToVideo:
    """Dreamina Seedance 2.5 reference-to-video via FAL. Cite refs as @Image1..,
    @Video1.., @Audio1.. in the prompt. (API allows up to 30 images / 10 videos /
    10 audio; this node exposes 9 / 3 / 3 slots.)"""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite refs as @Image1.., @Video1.., @Audio1.."}),
            "resolution": (SEEDANCE25_RESOLUTION, {"default": "720p"}),
            "duration": (SEEDANCE25_DURATION, {"default": "auto", "tooltip": "Seconds, or 'auto'."}),
            "aspect_ratio": (SEEDANCE25_ASPECT, {"default": "auto"}),
            "generate_audio": ("BOOLEAN", {"default": True}),
            "bitrate_mode": (["standard", "high"], {"default": "standard", "tooltip": "'high' = higher-quality, larger file."}),
        }
        optional = {}
        _add_image_slots(optional, SEEDANCE25_MAX_IMAGES)
        for i in range(1, MAX_REFERENCE_VIDEOS + 1):
            optional[f"video_{i}"] = ("VIDEO", {"tooltip": f"Reference video {i} (cite as @Video{i})."})
        for i in range(1, MAX_REFERENCE_AUDIOS + 1):
            optional[f"audio_{i}"] = ("AUDIO", {"tooltip": f"Reference audio {i} (cite as @Audio{i}). Requires >=1 image or video."})
        optional["end_user_id"] = ("STRING", {"default": "", "multiline": False})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, resolution, duration, aspect_ratio, generate_audio,
                       bitrate_mode, end_user_id="", image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: seedance-2.5 ref2video | {resolution} | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "Seedance 2.5 skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        push_node_status(unique_id, "Uploading references...", log)
        image_urls, img_fail, _ = _upload_kling_image_slots(slots, SEEDANCE25_MAX_IMAGES, image_batch, unique_id, log)
        video_urls, vid_fail = _upload_typed_ref_slots(slots, "video", MAX_REFERENCE_VIDEOS, ImageUtils.upload_video)
        audio_urls, aud_fail = _upload_typed_ref_slots(slots, "audio", MAX_REFERENCE_AUDIOS, ImageUtils.upload_audio)

        args = {
            "prompt": prompt.strip(),
            "resolution": resolution,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "generate_audio": bool(generate_audio),
            "bitrate_mode": bitrate_mode,
        }
        if image_urls:
            args["image_urls"] = image_urls
        if video_urls:
            args["video_urls"] = video_urls
        if audio_urls:
            args["audio_urls"] = audio_urls
        if not _is_blank(end_user_id):
            args["end_user_id"] = end_user_id.strip()

        failures = img_fail + vid_fail + aud_fail
        if failures:
            push_node_status(unique_id, f"Partial uploads — {len(failures)} failed: " + "; ".join(failures), log)
        return await _call_kling(SEEDANCE25_ENDPOINT, args, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Gemini Omni Flash — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALGeminiOmniFlashReferenceToVideo:
    """Google Gemini Omni Flash reference-to-video via FAL."""

    ENDPOINT = "google/gemini-omni-flash/reference-to-video"
    MAX_IMAGES = 9

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Reference images by tag, e.g. <IMAGE_REF_0>."}),
            "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
            "duration": ("INT", {"default": 8, "min": 3, "max": 10, "step": 1, "tooltip": "Seconds (3-10)."}),
        }
        optional = {}
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, aspect_ratio, duration, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: gemini-omni-flash | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "Gemini Omni Flash skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {"prompt": prompt.strip(), "aspect_ratio": aspect_ratio, "duration": int(duration)}
        return await _run_image_ref_video(self.ENDPOINT, base, "image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Gemini Omni Flash", require_image=True)


# ---------------------------------------------------------------------------
# Kling V3 Turbo Standard — image-to-video (single first frame)
# ---------------------------------------------------------------------------


class Soze_FALKlingV3TurboStandardImageToVideo:
    """Kling 3.0 Turbo Standard image-to-video via FAL (single first-frame image)."""

    ENDPOINT = "fal-ai/kling-video/v3/turbo/standard/image-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prompt (< 2500 chars)."}),
                "image": ("IMAGE", {"tooltip": "First-frame reference image."}),
                "duration": (["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], {"default": "5"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, image, duration, unique_id=None):
        log = EventLog()
        log.add(f"Config: kling-v3-turbo-standard i2v | {duration}s")
        if image is None:
            headline = "Kling V3 Turbo skipped: no image connected."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        push_node_status(unique_id, "Uploading first-frame image...", log)
        image_url = _upload_single_image(image)
        if not image_url:
            headline = "Kling V3 Turbo failed: image upload returned no URL."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        args = {"image_url": image_url, "duration": duration}
        if not _is_blank(prompt):
            args["prompt"] = prompt.strip()
        return await _call_kling(self.ENDPOINT, args, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Grok Imagine — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALGrokImagineReferenceToVideo:
    """xAI Grok Imagine reference-to-video via FAL. Cite images as @Image1.."""

    ENDPOINT = "xai/grok-imagine-video/reference-to-video"
    MAX_IMAGES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite images as @Image1.. in order."}),
            "duration": ("INT", {"default": 8, "min": 1, "max": 10, "step": 1, "tooltip": "Seconds (1-10)."}),
            "resolution": (["480p", "720p"], {"default": "480p"}),
            "aspect_ratio": (["16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16"], {"default": "16:9"}),
        }
        optional = {}
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, duration, resolution, aspect_ratio, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: grok-imagine ref2video | {resolution} | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "Grok Imagine skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {"prompt": prompt.strip(), "duration": int(duration), "resolution": resolution, "aspect_ratio": aspect_ratio}
        return await _run_image_ref_video(self.ENDPOINT, base, "reference_image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Grok Imagine", require_image=True)


# ---------------------------------------------------------------------------
# Happy Horse 1.1 — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALHappyHorse11ReferenceToVideo:
    """Alibaba Happy Horse 1.1 reference-to-video via FAL. Subjects map to
    character1..character9 in image_urls order."""

    ENDPOINT = "alibaba/happy-horse/v1.1/reference-to-video"
    MAX_IMAGES = 9

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Refer to subjects as character1..character9 (order matches images). Max 2500 chars."}),
            "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "9:21", "5:4", "4:5"], {"default": "16:9"}),
            "resolution": (["720p", "1080p"], {"default": "1080p"}),
            "duration": (["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], {"default": "5"}),
            "enable_safety_checker": ("BOOLEAN", {"default": True}),
        }
        optional = {"seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "tooltip": "0 = random."})}
        _add_image_slots(optional, cls.MAX_IMAGES, cite="character")
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, aspect_ratio, resolution, duration, enable_safety_checker,
                       seed=0, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: happy-horse-1.1 ref2video | {resolution} | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "Happy Horse skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": duration,
            "enable_safety_checker": bool(enable_safety_checker),
        }
        if isinstance(seed, int) and seed > 0:
            base["seed"] = int(seed)
        return await _run_image_ref_video(self.ENDPOINT, base, "image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Happy Horse", require_image=True)


# ---------------------------------------------------------------------------
# Kling O3 4K — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALKlingO34KReferenceToVideo:
    """Kling O3 native-4K reference-to-video via FAL. Cite references as
    @Image1..; optional start/end frames. (Advanced 'elements' not exposed.)"""

    ENDPOINT = "fal-ai/kling-video/o3/4k/reference-to-video"
    MAX_IMAGES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite references as @Image1.."}),
            "duration": (["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"], {"default": "5"}),
            "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            "shot_type": (["customize", "intelligent"], {"default": "customize"}),
            "generate_audio": ("BOOLEAN", {"default": False}),
        }
        optional = {
            "start_image": ("IMAGE", {"tooltip": "Optional first-frame image."}),
            "end_image": ("IMAGE", {"tooltip": "Optional last-frame image."}),
        }
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, duration, aspect_ratio, shot_type, generate_audio,
                       start_image=None, end_image=None, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: kling-o3-4k ref2video | {aspect_ratio} | {duration}s | audio={generate_audio}")
        base = {
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "shot_type": shot_type,
            "generate_audio": bool(generate_audio),
        }
        if not _is_blank(prompt):
            base["prompt"] = prompt.strip()
        extra = {}
        if start_image is not None:
            push_node_status(unique_id, "Uploading start frame...", log)
            extra["start_image_url"] = _upload_single_image(start_image)
        if end_image is not None:
            push_node_status(unique_id, "Uploading end frame...", log)
            extra["end_image_url"] = _upload_single_image(end_image)
        # Reference images optional here (start/end frame alone is valid).
        return await _run_image_ref_video(self.ENDPOINT, base, "image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Kling O3 4K",
                                          require_image=False, extra_urls=extra)


# ---------------------------------------------------------------------------
# PixVerse C1 — reference-to-video (structured image_references)
# ---------------------------------------------------------------------------


class Soze_FALPixVerseC1ReferenceToVideo:
    """PixVerse C1 reference-to-video via FAL. Each connected image becomes a
    reference named ref1..refN (type 'subject'); cite them as @ref1.. in the prompt."""

    ENDPOINT = "fal-ai/pixverse/c1/reference-to-video"
    MAX_IMAGES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite references as @ref1.. (limited to 2048 bytes)."}),
            "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "2:3", "3:2", "21:9"], {"default": "16:9"}),
            "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
            "duration": ("INT", {"default": 5, "min": 1, "max": 15, "step": 1, "tooltip": "Seconds (1-15)."}),
            "generate_audio": ("BOOLEAN", {"default": False, "tooltip": "Maps to generate_audio_switch (BGM, SFX, dialogue)."}),
        }
        optional = {"seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = random."})}
        _add_image_slots(optional, cls.MAX_IMAGES, cite="ref")
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, aspect_ratio, resolution, duration, generate_audio,
                       seed=0, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: pixverse-c1 ref2video | {resolution} | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "PixVerse C1 skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        push_node_status(unique_id, "Uploading reference images...", log)
        image_urls, failures, _ = _upload_kling_image_slots(slots, self.MAX_IMAGES, image_batch, unique_id, log)
        if not image_urls:
            details = "; ".join(failures) if failures else "no reference images connected"
            headline = f"PixVerse C1 skipped: {details}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        image_references = [
            {"type": "subject", "image_url": u, "ref_name": f"ref{i + 1}"}
            for i, u in enumerate(image_urls)
        ]
        args = {
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration": int(duration),
            "generate_audio_switch": bool(generate_audio),
            "image_references": image_references,
        }
        if isinstance(seed, int) and seed > 0:
            args["seed"] = int(seed)
        if failures:
            push_node_status(unique_id, f"Partial uploads — {len(failures)} failed: " + "; ".join(failures), log)
        return await _call_kling(self.ENDPOINT, args, unique_id=unique_id, event_log=log)


# ---------------------------------------------------------------------------
# Kling O1 — reference-to-video (Pro + Standard share one schema)
# ---------------------------------------------------------------------------


class _KlingO1Base:
    MAX_IMAGES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Cite references as @Image1.. in order."}),
            "duration": (["3", "4", "5", "6", "7", "8", "9", "10"], {"default": "5"}),
            "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
        }
        optional = {}
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, duration, aspect_ratio, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: {self.MODEL_NAME} ref2video | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = f"{self.MODEL_NAME} skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {"prompt": prompt.strip(), "duration": duration, "aspect_ratio": aspect_ratio}
        return await _run_image_ref_video(self.ENDPOINT, base, "image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, self.MODEL_NAME, require_image=True)


class Soze_FALKlingO1ProReferenceToVideo(_KlingO1Base):
    """Kling O1 Pro reference-to-video via FAL. Cite references as @Image1.."""
    ENDPOINT = "fal-ai/kling-video/o1/reference-to-video"
    MODEL_NAME = "Kling O1 Pro"


class Soze_FALKlingO1StandardReferenceToVideo(_KlingO1Base):
    """Kling O1 Standard reference-to-video via FAL. Cite references as @Image1.."""
    ENDPOINT = "fal-ai/kling-video/o1/standard/reference-to-video"
    MODEL_NAME = "Kling O1 Standard"


# ---------------------------------------------------------------------------
# Vidu Q3 (mix) — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALViduQ3ReferenceToVideoMix:
    """Vidu Q3 reference-to-video (mix) via FAL. 1-4 reference images."""

    ENDPOINT = "fal-ai/vidu/q3/reference-to-video/mix"
    MAX_IMAGES = 4

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Max 2000 characters."}),
            "duration": ("INT", {"default": 5, "min": 1, "max": 16, "step": 1, "tooltip": "Seconds (1-16)."}),
            "aspect_ratio": (["16:9", "9:16", "4:3", "3:4", "1:1"], {"default": "16:9"}),
            "resolution": (["360p", "540p", "720p", "1080p"], {"default": "720p"}),
            "audio": ("BOOLEAN", {"default": True, "tooltip": "Generate audio with the video."}),
        }
        optional = {"seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = random."})}
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, duration, aspect_ratio, resolution, audio,
                       seed=0, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: vidu-q3-mix ref2video | {resolution} | {aspect_ratio} | {duration}s")
        if _is_blank(prompt):
            headline = "Vidu Q3 skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {
            "prompt": prompt.strip(),
            "duration": int(duration),
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "audio": bool(audio),
        }
        if isinstance(seed, int) and seed > 0:
            base["seed"] = int(seed)
        return await _run_image_ref_video(self.ENDPOINT, base, "reference_image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Vidu Q3", require_image=True)


# ---------------------------------------------------------------------------
# Vidu Q1 — reference-to-video
# ---------------------------------------------------------------------------


class Soze_FALViduQ1ReferenceToVideo:
    """Vidu Q1 reference-to-video via FAL. Up to 7 reference images."""

    ENDPOINT = "fal-ai/vidu/q1/reference-to-video"
    MAX_IMAGES = 7

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Max 1500 characters."}),
            "aspect_ratio": (["16:9", "9:16", "1:1"], {"default": "16:9"}),
            "movement_amplitude": (["auto", "small", "medium", "large"], {"default": "auto"}),
            "bgm": ("BOOLEAN", {"default": False, "tooltip": "Add background music."}),
        }
        optional = {"seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "0 = random."})}
        _add_image_slots(optional, cls.MAX_IMAGES)
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, prompt, aspect_ratio, movement_amplitude, bgm,
                       seed=0, image_batch=None, unique_id=None, **slots):
        log = EventLog()
        log.add(f"Config: vidu-q1 ref2video | {aspect_ratio} | move={movement_amplitude}")
        if _is_blank(prompt):
            headline = "Vidu Q1 skipped: prompt is blank."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        base = {
            "prompt": prompt.strip(),
            "aspect_ratio": aspect_ratio,
            "movement_amplitude": movement_amplitude,
            "bgm": bool(bgm),
        }
        if isinstance(seed, int) and seed > 0:
            base["seed"] = int(seed)
        return await _run_image_ref_video(self.ENDPOINT, base, "reference_image_urls", slots, image_batch,
                                          self.MAX_IMAGES, unique_id, log, "Vidu Q1", require_image=True)


# ---------------------------------------------------------------------------
# Mirage Avatar X — audio-driven reference-to-video
# ---------------------------------------------------------------------------


MIRAGE_AVATARS = [
    "None",
    "Ayesha", "Ayesha (16:9)", "Farhan", "Farhan (16:9)", "Giulia", "Giulia (16:9)",
    "Jasmine", "Jasmine (16:9)", "Luke", "Luke (16:9)", "Maya", "Maya (16:9)",
    "Michael", "Michael (16:9)", "Neha", "Neha (16:9)", "Tariq", "Tariq (16:9)",
    "Valerie", "Valerie (16:9)",
]


class Soze_FALMirageAvatarXReferenceToVideo:
    """Mirage Avatar X via FAL — audio-driven talking avatar. The video follows
    the driving AUDIO. Provide a reference image OR video, or pick a stock avatar
    (a connected image/video reference overrides the avatar; video overrides image)."""

    ENDPOINT = "mirage-api/avatar-x/reference-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Driving audio (3-180s). The generated video lip-syncs to this."}),
            },
            "optional": {
                "avatar": (MIRAGE_AVATARS, {"default": "None", "tooltip": "Stock avatar; overridden by any image/video reference."}),
                "image_reference": ("IMAGE", {"tooltip": "Optional reference image (9:16 or 16:9). Ignored if a video reference is provided."}),
                "video_reference": ("VIDEO", {"tooltip": "Optional reference video (takes precedence over the image)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = VIDEO_RETURN_TYPES
    RETURN_NAMES = VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = "FAL/VideoGeneration"

    async def generate(self, audio, avatar="None", image_reference=None, video_reference=None, unique_id=None):
        log = EventLog()
        log.add(f"Config: mirage-avatar-x | avatar={avatar}")
        if audio is None:
            headline = "Mirage Avatar X skipped: no driving audio connected."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        push_node_status(unique_id, "Uploading driving audio...", log)
        try:
            audio_url = ImageUtils.upload_audio(audio)
        except Exception as e:
            logger.exception("Mirage Avatar X: audio upload failed")
            headline = f"ERROR uploading audio: {e!r}"
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))
        if not audio_url:
            headline = "Mirage Avatar X failed: audio upload returned no URL."
            push_node_status(unique_id, headline, log)
            return (None, "", finalize_status(headline, log))

        args = {"audio_url": audio_url}

        # video_reference takes precedence over image_reference.
        if video_reference is not None:
            push_node_status(unique_id, "Uploading reference video...", log)
            try:
                v = ImageUtils.upload_video(video_reference)
            except Exception:
                logger.exception("Mirage Avatar X: video upload failed")
                v = None
            if v:
                args["video_reference_url"] = v
        elif image_reference is not None:
            push_node_status(unique_id, "Uploading reference image...", log)
            u = _upload_single_image(image_reference)
            if u:
                args["image_reference_url"] = u

        if "video_reference_url" not in args and "image_reference_url" not in args and avatar and avatar != "None":
            args["avatar"] = avatar

        return await _call_kling(self.ENDPOINT, args, unique_id=unique_id, event_log=log)
