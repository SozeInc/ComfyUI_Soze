"""Get a single frame from a video as an IMAGE.

Accepts either an IMAGE batch ([B,H,W,C], B = frames — how ComfyUI usually
passes video between nodes) OR a ComfyUI VIDEO object (decoded to frames via its
components API). `frame_number = -1` returns the last frame.
"""

import logging

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


def _video_to_frames(video):
    """Best-effort extraction of the frame tensor ([B,H,W,C]) from a VIDEO."""
    if video is None:
        return None
    # Standard ComfyUI API: VideoInput.get_components().images
    try:
        comps = video.get_components()
        imgs = getattr(comps, "images", None)
        if imgs is None and isinstance(comps, dict):
            imgs = comps.get("images")
        if imgs is not None and hasattr(imgs, "shape"):
            return imgs
    except Exception:
        logger.exception("Get Frame: video.get_components() failed")
    # Fallbacks seen across versions.
    for attr in ("images", "frames"):
        val = getattr(video, attr, None)
        if val is not None and hasattr(val, "shape"):
            return val
    for meth in ("get_images", "get_frames"):
        fn = getattr(video, meth, None)
        if callable(fn):
            try:
                val = fn()
                if val is not None and hasattr(val, "shape"):
                    return val
            except Exception:
                pass
    return None


class Soze_GetFrame:
    """Pick one frame out of a video (IMAGE batch or VIDEO) as an IMAGE.

    frame_number: 0-based index; -1 returns the last frame. Out-of-range
    indices are clamped to the valid range and noted in the status. If both
    inputs are connected, `images` takes precedence.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_number": ("INT", {
                    "default": 0, "min": -1, "max": 0xFFFFFFF, "step": 1,
                    "tooltip": "0-based frame index. -1 = last frame.",
                }),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Video frames as an IMAGE batch."}),
                "video": ("VIDEO", {"tooltip": "A VIDEO object (decoded to frames). Used if 'images' is not connected."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "frame_index", "total_frames", "status")
    FUNCTION = "get_frame"
    CATEGORY = "image"

    def get_frame(self, frame_number, images=None, video=None, unique_id=None):
        log = EventLog()

        # Resolve frames: prefer a connected IMAGE batch, else decode the VIDEO.
        frames = images
        source = "images"
        if frames is None and video is not None:
            frames = _video_to_frames(video)
            source = "video"
            if frames is None:
                headline = "ERROR: could not extract frames from the VIDEO input (unsupported VIDEO type)."
                push_node_status(unique_id, headline, log)
                raise ValueError(headline)

        if frames is None or not hasattr(frames, "shape") or frames.shape[0] == 0:
            headline = "Skipped: no frames provided (connect 'images' or 'video')."
            push_node_status(unique_id, headline, log)
            return (None, 0, 0, finalize_status(headline, log))

        total = frames.shape[0]

        # Resolve the requested index.
        if frame_number == -1:
            idx = total - 1
        else:
            idx = int(frame_number)

        clamped = False
        if idx < 0:
            idx = 0
            clamped = True
        elif idx >= total:
            idx = total - 1
            clamped = True

        frame = frames[idx:idx + 1]  # keep a [1,H,W,C] batch of one

        h, w = frame.shape[1], frame.shape[2]
        headline = f"OK: frame {idx + 1}/{total} ({w}x{h}) from {source}"
        if frame_number == -1:
            headline += " [last]"
        if clamped:
            headline += f" [requested {frame_number} clamped to {idx}]"
        push_node_status(unique_id, headline, log)
        return (frame, idx, total, finalize_status(headline, log))
