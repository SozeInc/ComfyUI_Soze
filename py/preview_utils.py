"""Shared image-preview plumbing for image-loading nodes.

ComfyUI's frontend automatically renders a thumbnail strip at the bottom of any
OUTPUT_NODE whose executed result contains `ui.images` (a list of
{"filename", "subfolder", "type"} entries pointing at files in the temp dir).

`enable_image_preview(*classes)` mutates each given node class so that:
  - OUTPUT_NODE is set to True, and
  - its FUNCTION method is wrapped to save every IMAGE output frame to the
    temp directory and attach `ui.images`, while passing the original result
    tuple through unchanged.

This is centralized on purpose: the loader functions have many early-return
branches (skip/error cases), and wrapping the whole call covers all of them
without editing each return site.
"""

import os
import uuid
import logging

import numpy as np
from PIL import Image

import folder_paths

logger = logging.getLogger(__name__)

# Cap how many frames we materialize per execution so a huge batch can't flood
# the temp dir or stall the canvas.
MAX_PREVIEW_FRAMES = 64

_WRAPPED_FLAG = "_soze_preview_wrapped"


def _to_numpy(value):
    """Best-effort convert a torch.Tensor / array-like to a numpy array."""
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _frames_from_value(value, is_list):
    """Yield single [H, W, C] float frames from an IMAGE output value.

    Handles: a [B,H,W,C] tensor, an [H,W,C] tensor, or (when is_list) a list of
    such tensors. None entries are skipped.
    """
    frames = []
    if value is None:
        return frames
    items = value if is_list else [value]
    if not isinstance(items, (list, tuple)):
        items = [items]
    for item in items:
        if item is None:
            continue
        try:
            arr = _to_numpy(item)
        except Exception:
            continue
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            continue
        for frame in arr:
            frames.append(frame)
    return frames


def _save_frames_to_temp(frames):
    """Save frames as PNGs in the temp dir; return ComfyUI ui.images entries."""
    if not frames:
        return []
    temp_dir = folder_paths.get_temp_directory()
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except OSError:
        return []

    prefix = f"soze_prev_{uuid.uuid4().hex[:12]}"
    results = []
    for i, frame in enumerate(frames[:MAX_PREVIEW_FRAMES]):
        try:
            a = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]  # grayscale -> 2D for PIL
            img = Image.fromarray(a)
            file = f"{prefix}_{i:03d}.png"
            img.save(os.path.join(temp_dir, file), compress_level=4)
            results.append({"filename": file, "subfolder": "", "type": "temp"})
        except Exception:
            logger.exception("Soze preview: failed to save frame %d", i)
            continue
    return results


def _collect_preview_entries(cls, result_tuple):
    """Pull every IMAGE output out of the result tuple and save previews."""
    return_types = getattr(cls, "RETURN_TYPES", ()) or ()
    output_is_list = getattr(cls, "OUTPUT_IS_LIST", None)

    all_frames = []
    for idx, rtype in enumerate(return_types):
        if rtype != "IMAGE":
            continue
        if idx >= len(result_tuple):
            continue
        is_list = bool(output_is_list[idx]) if (output_is_list and idx < len(output_is_list)) else False
        all_frames.extend(_frames_from_value(result_tuple[idx], is_list))
        if len(all_frames) >= MAX_PREVIEW_FRAMES:
            break

    return _save_frames_to_temp(all_frames)


def enable_image_preview(*classes):
    """Mutate each class so its IMAGE outputs render as a node thumbnail strip."""
    for cls in classes:
        if cls is None:
            continue
        if getattr(cls, _WRAPPED_FLAG, False):
            continue
        func_name = getattr(cls, "FUNCTION", None)
        if not func_name or not hasattr(cls, func_name):
            logger.warning("Soze preview: %s has no FUNCTION to wrap", getattr(cls, "__name__", cls))
            continue

        original = getattr(cls, func_name)

        def make_wrapper(orig, node_cls):
            def wrapper(self, *args, **kwargs):
                result = orig(self, *args, **kwargs)

                # If the node already returns a ui/result dict, only enrich it
                # with images when it doesn't already provide them.
                if isinstance(result, dict):
                    ui = result.setdefault("ui", {})
                    if "images" not in ui:
                        result_tuple = result.get("result")
                        if isinstance(result_tuple, (list, tuple)):
                            try:
                                entries = _collect_preview_entries(node_cls, result_tuple)
                                if entries:
                                    ui["images"] = entries
                            except Exception:
                                logger.exception("Soze preview: enrich failed for %s", node_cls.__name__)
                    return result

                if not isinstance(result, (list, tuple)):
                    return result  # unexpected shape — leave untouched

                try:
                    entries = _collect_preview_entries(node_cls, result)
                except Exception:
                    logger.exception("Soze preview: build failed for %s", node_cls.__name__)
                    entries = []

                if not entries:
                    # Nothing to preview (e.g. skip/None case) — return as-is so
                    # behavior is identical to the un-wrapped node.
                    return result

                return {"ui": {"images": entries}, "result": tuple(result)}

            return wrapper

        setattr(cls, func_name, make_wrapper(original, cls))
        cls.OUTPUT_NODE = True
        setattr(cls, _WRAPPED_FLAG, True)
        logger.info("Soze preview: enabled for %s", getattr(cls, "__name__", cls))
