"""Linked-chain JSON builder.

Build JSON objects in ComfyUI by chaining typed name/value pair nodes:

    StringPair("name", "Alice")  ──chain──>
    IntPair("age", 30)           ──chain──>
    ArrayPair("tags", "a", "b")  ──chain──>
    Generate JSON                 ──> '{"name":"Alice","age":30,"tags":["a","b"]}'

Each pair node takes an optional `chain_in` (the previous chain, or unconnected
if this is the head) and emits `chain_out`. The Generate JSON node serializes
the final chain into a properly formatted JSON string.

Implementation notes:
- The chain flowing between nodes is a list of (key, value) tuples, passed via
  a custom `JSON_PAIRS` type. Nodes never mutate the input chain — they return
  a new list with their pair appended.
- The value inputs on pair nodes use `forceInput: True` so the user MUST wire
  in a value (no inline widget). Names stay as plain widget fields.
- The Array Pair node accepts any input type on its 10 value slots via the
  community "*" wildcard trick.
- Duplicate keys are allowed; the LAST value wins (standard dict behavior).
  Generate JSON reports duplicates in its status output.
"""

import ast
import base64
import io
import json
import logging
import os

import folder_paths
import numpy as np
import torch
from PIL import Image

from .status_utils import push_node_status

logger = logging.getLogger(__name__)


JSON_IMAGE_MODES = ["data_uri", "filepath"]
JSON_IMAGE_FORMATS = ["png", "jpeg", "webp"]


JSON_PAIRS_TYPE = "JSON_PAIRS"
MAX_ARRAY_ITEMS = 10


class _AnyType(str):
    """ComfyUI 'accepts anything' trick — used for the array pair value slots.

    ComfyUI validates connections by comparing type strings with `!=`. If both
    sides report `!=` False, the connection is allowed. Returning False from
    __ne__ makes this type accept any source type.
    """

    def __ne__(self, other):
        return False


ANY = _AnyType("*")


def _append_pair(chain_in, key, value):
    """Return a NEW chain list with (key, value) appended. Never mutates input.

    Blank keys are silently skipped so an unconfigured pair node is harmless.
    """
    if not key or not str(key).strip():
        return (list(chain_in) if chain_in else [],)
    base = list(chain_in) if chain_in else []
    base.append((str(key), value))
    return (base,)


def _pairs_to_object(chain):
    """Collapse a chain (list of (key, value) tuples) into a dict.

    Later keys overwrite earlier ones — same rule the JSON Generate node uses.
    Returns (obj_dict, duplicate_keys_list).
    """
    obj = {}
    duplicate_keys = []
    for k, v in (chain or []):
        if k in obj:
            duplicate_keys.append(k)
        obj[k] = _coerce_json_value(v)
    return obj, duplicate_keys


def _coerce_json_value(v):
    """Coerce a runtime value to a JSON-friendly primitive.

    Handles the common ComfyUI types we'd see flowing through the Array Pair's
    wildcard slots: str/int/float/bool/None, lists, dicts. Anything else falls
    back to `str(v)`.
    """
    if isinstance(v, bool):  # NOTE: bool is a subclass of int — check first
        return v
    if isinstance(v, (str, int, float)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_coerce_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce_json_value(x) for k, x in v.items()}
    return str(v)


# ---------------------------------------------------------------------------
# Scalar Pair Nodes
# ---------------------------------------------------------------------------


class Soze_JSONStringPair:
    """Adds a `name: string-value` pair to the JSON chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
            "optional": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, value, chain_in=None):
        return _append_pair(chain_in, name, str(value))


# Bulk-pair authoring: same shape as the single String Pair, but ten of them
# packed into one node. Keeps the widget surface manageable when you'd
# otherwise have a long chain of single-pair nodes.
MAX_BULK_STRING_PAIRS = 10


class Soze_JSONStringPairs10X:
    """Adds up to 10 `name: string-value` pairs to the JSON chain in one node.

    Both the names AND values are STRING widgets (no forceInput), so you can
    type them directly — handy for authoring lots of static keys quickly. Any
    row whose `name_N` is blank is skipped, so a half-filled node is fine.

    The output is a normal JSON_PAIRS chain. Two common patterns:

    * **Siblings** — connect the node's `chain_out` into the next pair node's
      `chain_in` (or straight into JSON Generate). The 10 pairs land at the
      same nesting level as whatever's already in the chain.

    * **Sub-values of a property** — feed `chain_out` into a JSON Object Pair's
      `nested_chain` input. The Object Pair wraps the 10 pairs under one named
      property. This is what you want when you'd like e.g. a `meta: { ... }`
      block with 10 keys under it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        for i in range(1, MAX_BULK_STRING_PAIRS + 1):
            required[f"name_{i}"] = ("STRING", {"default": "", "multiline": False})
            required[f"value_{i}"] = ("STRING", {"default": "", "multiline": False})
        return {
            "required": required,
            "optional": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, chain_in=None, **kwargs):
        # Build the chain inline (rather than calling _append_pair 10 times)
        # so we copy the input list only once.
        base = list(chain_in) if chain_in else []
        for i in range(1, MAX_BULK_STRING_PAIRS + 1):
            name = kwargs.get(f"name_{i}", "")
            if not name or not str(name).strip():
                continue
            value = kwargs.get(f"value_{i}", "")
            base.append((str(name), str(value)))
        return (base,)


class Soze_JSONIntPair:
    """Adds a `name: integer-value` pair to the JSON chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "value": ("INT", {"default": 0, "min": -0x7FFFFFFFFFFFFFFF, "max": 0x7FFFFFFFFFFFFFFF, "step": 1, "forceInput": True}),
            },
            "optional": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, value, chain_in=None):
        return _append_pair(chain_in, name, int(value))


class Soze_JSONFloatPair:
    """Adds a `name: float-value` pair to the JSON chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "value": ("FLOAT", {"default": 0.0, "min": -1e18, "max": 1e18, "step": 0.001, "forceInput": True}),
            },
            "optional": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, value, chain_in=None):
        return _append_pair(chain_in, name, float(value))


class Soze_JSONBoolPair:
    """Adds a `name: boolean-value` pair to the JSON chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "value": ("BOOLEAN", {"default": False, "forceInput": True}),
            },
            "optional": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, value, chain_in=None):
        return _append_pair(chain_in, name, bool(value))


# ---------------------------------------------------------------------------
# Array Pair Node — accepts any type via 10 input slots
# ---------------------------------------------------------------------------


class Soze_JSONArrayPair:
    """Adds a `name: [v1, v2, ...]` array pair to the JSON chain.

    Values come from up to 10 typed inputs (`value_1`..`value_10`); only
    connected slots are included, in slot order. The slots accept any type
    (STRING / INT / FLOAT / BOOLEAN — or anything coerce-able to one).
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"value_{i}": (ANY, {"forceInput": True}) for i in range(1, MAX_ARRAY_ITEMS + 1)}
        optional["chain_in"] = (JSON_PAIRS_TYPE,)
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, chain_in=None, **slots):
        values = []
        for i in range(1, MAX_ARRAY_ITEMS + 1):
            v = slots.get(f"value_{i}")
            if v is None:
                continue
            values.append(_coerce_json_value(v))
        return _append_pair(chain_in, name, values)


# ---------------------------------------------------------------------------
# Object Pair Node — nest a sub-chain under one property key
# ---------------------------------------------------------------------------


class Soze_JSONObjectPair:
    """Add a `name: { ...nested pairs... }` pair to the JSON chain.

    Build a separate side-chain of pair nodes (any mix of String/Int/Float/
    Bool/Array/Image/Object pair nodes) and connect its `chain_out` to this
    node's `nested_chain` input. Those pairs are collapsed into a single
    nested object and attached to the outer chain under `name`.

    Object Pair nodes can themselves be nested — the `nested_chain` of one
    Object Pair can contain another Object Pair — so you can build trees of
    arbitrary depth.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "nested_chain": (JSON_PAIRS_TYPE, {"tooltip": "Side-chain whose pairs become the nested object's keys/values."}),
                "chain_in": (JSON_PAIRS_TYPE, {"tooltip": "Outer chain this property is appended to."}),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, nested_chain=None, chain_in=None):
        nested_obj, _dupes = _pairs_to_object(nested_chain)
        return _append_pair(chain_in, name, nested_obj)


# ---------------------------------------------------------------------------
# Raw JSON Pair Node — embed a JSON-encoded string as a property value
# ---------------------------------------------------------------------------


def _parse_json_value(text, strict=True):
    """Parse a JSON string into a Python value.

    Strips optional ```json``` fenced-code wrappers (which the existing
    JSON parser nodes also tolerate). When `strict=False`, falls back to
    ast.literal_eval for Python-style dicts/lists with single quotes.
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.startswith("```json"):
        s = s.replace("```json", "").replace("```", "").strip()
    elif s.startswith("```"):
        s = s.replace("```", "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        if strict:
            raise
        try:
            return ast.literal_eval(s)
        except Exception:
            raise


class Soze_JSONRawPair:
    """Add a `name: <parsed-json>` pair to the chain, where the value is parsed
    from a JSON string instead of authored via the pair-chain.

    Use this to bridge the output of `JSON Value Parser (Soze)` (a JSON-encoded
    string) into a new builder chain — e.g. extract a nested `Story` object
    from one file and re-attach it under any property name in a new file.

    The parsed result can be an object, array, or scalar; it's embedded as-is.
    With `strict=False`, Python-style literals (single-quoted dicts / lists)
    are also accepted.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "json_string": ("STRING", {"default": "", "multiline": True, "forceInput": True, "tooltip": "JSON to embed. Object, array, or scalar."}),
            },
            "optional": {
                "strict": ("BOOLEAN", {"default": True, "tooltip": "If False, falls back to ast.literal_eval for Python-style dicts/lists."}),
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, json_string, strict=True, chain_in=None):
        parsed = _parse_json_value(json_string, strict=bool(strict))
        return _append_pair(chain_in, name, parsed)


# ---------------------------------------------------------------------------
# Generate JSON — terminal node that serializes the chain
# ---------------------------------------------------------------------------


class Soze_JSONGenerate:
    """Serialize a chain of name/value pairs into a JSON string.

    Connect the last pair node's `chain_out` here. Outputs the JSON text and a
    status string summarizing the result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chain_in": (JSON_PAIRS_TYPE,),
            },
            "optional": {
                "pretty": ("BOOLEAN", {"default": True, "tooltip": "Pretty-print with the given indent. If False, output is compact."}),
                "indent": ("INT", {"default": 2, "min": 0, "max": 10, "step": 1}),
                "sort_keys": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("json", "status")
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"
    OUTPUT_NODE = True

    def build(self, chain_in, pretty=True, indent=2, sort_keys=False, unique_id=None):
        obj, duplicate_keys = _pairs_to_object(chain_in)

        try:
            if pretty:
                text = json.dumps(obj, indent=int(indent), ensure_ascii=False, sort_keys=bool(sort_keys))
            else:
                text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=bool(sort_keys))
        except Exception as e:
            logger.exception("Soze_JSONGenerate: serialization failed")
            status = f"ERROR: {e!r}"
            push_node_status(unique_id, status)
            return ("", status)

        dupes_note = ""
        if duplicate_keys:
            dupes_note = f" (overwrote duplicate keys: {sorted(set(duplicate_keys))})"
        status = f"OK: {len(obj)} key(s), {len(text)} chars{dupes_note}"
        push_node_status(unique_id, status)
        return (text, status)


# ---------------------------------------------------------------------------
# Image Pair Nodes — encode a ComfyUI IMAGE as a JSON-friendly string value
# ---------------------------------------------------------------------------


def _frame_to_pil(frame_tensor):
    """A ComfyUI IMAGE frame [H,W,C] (float32 in [0,1]) -> PIL Image."""
    arr = (frame_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr, mode="RGB")


def _pil_save_kwargs(fmt: str, quality: int):
    """Return (PIL save_format, save_kwargs) for a chosen format/quality."""
    if fmt == "jpeg":
        return "JPEG", {"quality": int(quality)}
    if fmt == "webp":
        return "WEBP", {"quality": int(quality)}
    return "PNG", {}


def _pil_to_data_uri(pil_image, fmt: str, quality: int) -> str:
    """Encode a PIL image as a `data:image/<fmt>;base64,...` URI."""
    save_format, save_kwargs = _pil_save_kwargs(fmt, quality)
    if save_format == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format=save_format, **save_kwargs)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "jpeg" else f"image/{fmt}"
    return f"data:{mime};base64,{encoded}"


def _resolve_save_target(save_path: str, fmt: str, append_suffix: bool) -> str:
    """Resolve `save_path` to an absolute target file path.

    - Empty save_path defaults to `json_image.<ext>` in the ComfyUI output dir.
    - Adds the correct extension if missing.
    - Relative paths resolve under ComfyUI's output directory; absolute paths
      are accepted as-is.
    - If `append_suffix` and the target exists, appends an incrementing
      `_00001` counter (matching ComfyUI's save-image convention).
    """
    if not save_path or not save_path.strip():
        save_path = "json_image"
    ext = "jpg" if fmt == "jpeg" else fmt
    if not save_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        save_path = f"{save_path}.{ext}"

    if os.path.isabs(save_path):
        target = os.path.normpath(save_path)
    else:
        target = os.path.normpath(os.path.join(folder_paths.get_output_directory(), save_path))

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if append_suffix:
        base, ext_part = os.path.splitext(target)
        counter = 1
        while True:
            candidate = f"{base}_{counter:05d}{ext_part}"
            if not os.path.exists(candidate):
                target = candidate
                break
            counter += 1
    return target


def _pil_to_file(pil_image, save_path: str, fmt: str, quality: int, append_suffix: bool) -> str:
    """Save a PIL image to disk and return the absolute path written."""
    target = _resolve_save_target(save_path, fmt, append_suffix)
    save_format, save_kwargs = _pil_save_kwargs(fmt, quality)
    if save_format == "JPEG" and pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    pil_image.save(target, format=save_format, **save_kwargs)
    return target


def _encode_image_frame(frame_tensor, mode, fmt, quality, save_path, append_suffix):
    """Dispatch a single frame to the right encoder based on `mode`."""
    pil = _frame_to_pil(frame_tensor)
    if mode == "data_uri":
        return _pil_to_data_uri(pil, fmt, quality)
    if mode == "filepath":
        return _pil_to_file(pil, save_path, fmt, quality, append_suffix)
    raise ValueError(f"Unknown JSON image mode: {mode!r}")


class Soze_JSONImagePair:
    """Add a `name: <encoded-image>` pair to the JSON chain. Uses the first
    frame of the batch.

    Modes:
    - `data_uri`: emits `data:image/<fmt>;base64,...` — self-contained but big.
    - `filepath`: saves the image to disk and emits the absolute file path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE", {"forceInput": True}),
                "mode": (JSON_IMAGE_MODES, {"default": "data_uri"}),
                "format": (JSON_IMAGE_FORMATS, {"default": "png"}),
            },
            "optional": {
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Only used for jpeg / webp."}),
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "filepath mode: target path (relative to ComfyUI output dir, or absolute). Extension is added if missing."}),
                "append_suffix": ("BOOLEAN", {"default": True, "tooltip": "filepath mode: append _00001 if the target already exists."}),
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, image, mode, format, jpeg_quality=95, save_path="", append_suffix=True, chain_in=None):
        if image is None or image.shape[0] == 0:
            return _append_pair(chain_in, name, None)
        value = _encode_image_frame(image[0], mode, format, jpeg_quality, save_path, append_suffix)
        return _append_pair(chain_in, name, value)


class Soze_JSONImageEncoder:
    """Encode an IMAGE into a STRING that can be stored as a JSON value.

    Useful when you need the encoded image at a node-input boundary instead of
    bundled with a key (e.g. to feed into `JSON String Pair`'s `value` socket,
    or into any text node). Uses the first frame of the batch.

    Modes:
    - `data_uri`: emits `data:image/<fmt>;base64,...` — self-contained but
      large (a 1MP PNG is ~1.5 MB of base64 text).
    - `filepath`: saves the image to disk and emits its absolute path —
      tiny string, but the JSON is no longer self-contained.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "mode": (JSON_IMAGE_MODES, {"default": "data_uri"}),
                "format": (JSON_IMAGE_FORMATS, {"default": "png"}),
            },
            "optional": {
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Only used for jpeg / webp."}),
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "filepath mode: target path (relative to ComfyUI output dir, or absolute). Extension is added if missing."}),
                "append_suffix": ("BOOLEAN", {"default": True, "tooltip": "filepath mode: append _00001 if the target already exists."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("encoded",)
    FUNCTION = "encode"
    CATEGORY = "Soze Nodes/JSON"

    def encode(self, image, mode, format, jpeg_quality=95, save_path="", append_suffix=True):
        if image is None or image.shape[0] == 0:
            return ("",)
        return (_encode_image_frame(image[0], mode, format, jpeg_quality, save_path, append_suffix),)


# ---------------------------------------------------------------------------
# Inverse: STRING -> IMAGE
# ---------------------------------------------------------------------------


def _decode_one_image_value(encoded):
    """Decode a single string into a [1, H, W, 3] float32 IMAGE tensor.

    Recognized inputs:
      * `data:image/<fmt>;base64,...`  — base64 data URI
      * `http(s)://...`                — fetch via HTTP(S)
      * anything else                  — treated as a file path on disk
    """
    s = str(encoded).strip()
    if not s:
        raise ValueError("Empty encoded value")
    if s.startswith("data:"):
        _, _, b64 = s.partition(",")
        raw = base64.b64decode(b64)
        pil = Image.open(io.BytesIO(raw))
    elif s.lower().startswith(("http://", "https://")):
        import requests  # local import — only needed for the URL branch
        resp = requests.get(s, timeout=(10, 60), stream=True)
        resp.raise_for_status()
        pil = Image.open(io.BytesIO(resp.content))
    else:
        pil = Image.open(s)
    pil = pil.convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _parse_image_string_list(encoded):
    """Return a list of strings from a flexible input.

    Accepts:
      * a JSON-encoded array of strings  ->  parsed as JSON
      * newline-separated values          ->  split on newlines (handles the
                                              FAL nodes' `image_urls` output)
      * a single value                    ->  one-element list
    """
    s = str(encoded).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            pass
    if "\n" in s:
        return [line.strip() for line in s.split("\n") if line.strip()]
    return [s]


def _align_decoded_tensors(tensors):
    """Concatenate [1,H,W,3] tensors into a [B,H,W,3] batch, resizing later
    frames to match the first when sizes differ. Uses ComfyUI's common_upscale
    when available, falling back to PIL.BILINEAR otherwise."""
    if not tensors:
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    base = tensors[0]
    base_h, base_w = base.shape[1], base.shape[2]
    aligned = [base]
    for t in tensors[1:]:
        if t.shape[1] != base_h or t.shape[2] != base_w:
            try:
                from comfy.utils import common_upscale
                t = common_upscale(t.movedim(-1, 1), base_w, base_h, "bilinear", "center").movedim(1, -1)
            except Exception:
                # Fallback when running outside ComfyUI (tests / scripts).
                arr = (t.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
                pil = Image.fromarray(arr).resize((base_w, base_h), Image.BILINEAR)
                arr = np.array(pil).astype(np.float32) / 255.0
                t = torch.from_numpy(arr)[None,]
        aligned.append(t)
    return torch.cat(aligned, dim=0)


class Soze_JSONImageDecoder:
    """Inverse of `JSON Image Encoder`. Takes a STRING and returns an IMAGE.

    Accepts whatever the encoder (or `JSON Image Array Pair`, or the FAL
    `image_urls` STRING outputs) might produce:

    * a single `data:image/<fmt>;base64,...` URI
    * a single file path (absolute or relative to the current working dir)
    * a single `http(s)://...` URL
    * a JSON-encoded array of any of the above (e.g. from Image Array Pair)
    * newline-separated values (e.g. the FAL nodes' `image_urls` output)

    All frames are stacked into one IMAGE batch. Frames of differing sizes are
    bilinear-resized to match the first frame, matching the rest of the Soze
    pack's batching convention.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoded": ("STRING", {"default": "", "multiline": True, "tooltip": "Data URI, file path, URL, JSON array, or newline-separated list."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "count", "status")
    FUNCTION = "decode"
    CATEGORY = "Soze Nodes/JSON"

    def decode(self, encoded, unique_id=None):
        if not encoded or not str(encoded).strip():
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            status = "Skipped: empty encoded input."
            push_node_status(unique_id, status)
            return (blank, 0, status)

        values = _parse_image_string_list(encoded)
        if not values:
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            status = "Skipped: no values parsed."
            push_node_status(unique_id, status)
            return (blank, 0, status)

        push_node_status(unique_id, f"Decoding {len(values)} value(s)...")
        tensors = []
        failures = []
        for i, v in enumerate(values):
            try:
                tensors.append(_decode_one_image_value(v))
            except Exception as e:
                logger.exception("JSONImageDecoder: failed to decode value %d", i)
                failures.append(f"value[{i}]: {e!r}")

        if not tensors:
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            details = "; ".join(failures) or "no decoder error captured"
            status = f"ERROR: every value failed to decode. {details}"
            push_node_status(unique_id, status)
            return (blank, 0, status)

        image_batch = _align_decoded_tensors(tensors)
        status = f"OK: decoded {len(tensors)}/{len(values)} value(s), size={image_batch.shape[2]}x{image_batch.shape[1]}"
        if failures:
            status += " | failures: " + "; ".join(failures)
        push_node_status(unique_id, status)
        return (image_batch, len(tensors), status)


class Soze_JSONImageArrayPair:
    """Add a `name: [<encoded-image>, ...]` array pair to the JSON chain — one
    entry per frame in the IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE", {"forceInput": True}),
                "mode": (JSON_IMAGE_MODES, {"default": "data_uri"}),
                "format": (JSON_IMAGE_FORMATS, {"default": "png"}),
            },
            "optional": {
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Only used for jpeg / webp."}),
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "filepath mode: target path. Each frame gets its own _NNNNN suffix."}),
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, image, mode, format, jpeg_quality=95, save_path="", chain_in=None):
        if image is None or image.shape[0] == 0:
            return _append_pair(chain_in, name, [])
        values = []
        for i in range(image.shape[0]):
            # In filepath mode, always append a per-frame counter so we never
            # overwrite earlier frames.
            values.append(
                _encode_image_frame(image[i], mode, format, jpeg_quality, save_path, append_suffix=True)
            )
        return _append_pair(chain_in, name, values)


# ---------------------------------------------------------------------------
# Audio Pair / Encoder / Decoder — same shape as the Image family
# ---------------------------------------------------------------------------


JSON_AUDIO_MODES = ["data_uri", "filepath"]
# Encoder always emits MP3. Encoding via torchaudio requires the ffmpeg backend
# (which ComfyUI normally has). The decoder is more permissive and accepts
# whatever torchaudio.load() can read on the user's machine.
JSON_AUDIO_ENCODE_FORMAT = "mp3"
JSON_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")


def _audio_dict_to_frames(audio_dict):
    """Yield each frame in a ComfyUI AUDIO dict as a [C, S] tensor + sample_rate.

    ComfyUI's AUDIO type is `{"waveform": tensor[B, C, S], "sample_rate": int}`.
    For B == 1 (the usual case) you get one frame; B > 1 yields each in order.
    """
    waveform = audio_dict["waveform"]
    sample_rate = int(audio_dict["sample_rate"])
    if waveform.dim() == 2:
        # Already [C, S] — treat as single frame.
        yield waveform, sample_rate
        return
    for i in range(waveform.shape[0]):
        yield waveform[i], sample_rate


def _torchaudio_save(target, waveform_cs, sample_rate):
    """Save a single [C, S] waveform to a path or BytesIO as MP3.

    Wraps torchaudio with two clearer error cases: torchaudio missing, or the
    underlying backend (typically ffmpeg) can't encode MP3.
    """
    try:
        import torchaudio
    except ImportError as e:
        raise RuntimeError(
            "torchaudio is required for the JSON Audio Encoder/Decoder. "
            "Install it with: pip install torchaudio"
        ) from e
    try:
        torchaudio.save(target, waveform_cs.cpu(), int(sample_rate), format=JSON_AUDIO_ENCODE_FORMAT)
    except (RuntimeError, OSError) as e:
        raise RuntimeError(
            f"MP3 encoding failed (torchaudio.save): {e}. "
            f"This usually means torchaudio's ffmpeg backend is missing — "
            f"install ffmpeg (e.g. `conda install -c conda-forge ffmpeg` or your OS package manager) "
            f"and ensure torchaudio>=2.1."
        ) from e


def _save_audio_bytes(waveform_cs, sample_rate):
    """Serialize a single [C, S] waveform to MP3 bytes."""
    buf = io.BytesIO()
    _torchaudio_save(buf, waveform_cs, sample_rate)
    return buf.getvalue()


def _audio_frame_to_data_uri(waveform_cs, sample_rate):
    raw = _save_audio_bytes(waveform_cs, sample_rate)
    encoded = base64.b64encode(raw).decode("ascii")
    # audio/mpeg is the IANA-registered MIME for MP3.
    return f"data:audio/mpeg;base64,{encoded}"


def _resolve_audio_save_target(save_path, append_suffix):
    """Resolve `save_path` to an absolute MP3 target."""
    if not save_path or not save_path.strip():
        save_path = "json_audio"
    if not save_path.lower().endswith(JSON_AUDIO_EXTENSIONS):
        save_path = f"{save_path}.{JSON_AUDIO_ENCODE_FORMAT}"

    if os.path.isabs(save_path):
        target = os.path.normpath(save_path)
    else:
        target = os.path.normpath(os.path.join(folder_paths.get_output_directory(), save_path))

    parent = os.path.dirname(target)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if append_suffix:
        base, ext_part = os.path.splitext(target)
        counter = 1
        while True:
            candidate = f"{base}_{counter:05d}{ext_part}"
            if not os.path.exists(candidate):
                target = candidate
                break
            counter += 1
    return target


def _audio_frame_to_file(waveform_cs, sample_rate, save_path, append_suffix):
    """Save a single [C, S] waveform to disk as MP3; return absolute path written."""
    target = _resolve_audio_save_target(save_path, append_suffix)
    _torchaudio_save(target, waveform_cs, sample_rate)
    return target


def _encode_audio_frame(waveform_cs, sample_rate, mode, save_path, append_suffix):
    if mode == "data_uri":
        return _audio_frame_to_data_uri(waveform_cs, sample_rate)
    if mode == "filepath":
        return _audio_frame_to_file(waveform_cs, sample_rate, save_path, append_suffix)
    raise ValueError(f"Unknown JSON audio mode: {mode!r}")


def _decode_one_audio_value(encoded):
    """Decode a string into an AUDIO dict {waveform: [1, C, S], sample_rate}."""
    try:
        import torchaudio
    except ImportError as e:
        raise RuntimeError(
            "torchaudio is required for the JSON Audio Decoder. "
            "Install it with: pip install torchaudio"
        ) from e

    s = str(encoded).strip()
    if not s:
        raise ValueError("Empty encoded value")

    if s.startswith("data:"):
        _, _, b64 = s.partition(",")
        raw = base64.b64decode(b64)
        waveform, sample_rate = torchaudio.load(io.BytesIO(raw))
    elif s.lower().startswith(("http://", "https://")):
        import requests
        resp = requests.get(s, timeout=(10, 60), stream=True)
        resp.raise_for_status()
        waveform, sample_rate = torchaudio.load(io.BytesIO(resp.content))
    else:
        waveform, sample_rate = torchaudio.load(s)

    # waveform: [C, S] -> [1, C, S] to match the ComfyUI AUDIO convention.
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _align_decoded_audios(audio_dicts):
    """Stack multiple AUDIO dicts into one batched AUDIO dict.

    - Resamples all to the first audio's sample rate (torchaudio.functional.resample).
    - Skips frames whose channel count doesn't match the first.
    - Pads shorter waveforms with zeros to match the longest length.
    """
    if not audio_dicts:
        return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
    if len(audio_dicts) == 1:
        return audio_dicts[0]

    try:
        import torchaudio.functional as AF
    except ImportError:
        # Fall back to the first audio if we can't resample.
        return audio_dicts[0]

    base_sr = audio_dicts[0]["sample_rate"]
    base_w = audio_dicts[0]["waveform"]
    base_w = base_w[0] if base_w.dim() == 3 else base_w
    base_channels = base_w.shape[0]

    aligned = [base_w]
    skipped_channels = 0
    for d in audio_dicts[1:]:
        w = d["waveform"]
        if w.dim() == 3:
            w = w[0]
        if d["sample_rate"] != base_sr:
            w = AF.resample(w, d["sample_rate"], base_sr)
        if w.shape[0] != base_channels:
            skipped_channels += 1
            continue
        aligned.append(w)

    max_samples = max(w.shape[1] for w in aligned)
    padded = []
    for w in aligned:
        if w.shape[1] < max_samples:
            pad = torch.zeros((w.shape[0], max_samples - w.shape[1]), dtype=w.dtype)
            w = torch.cat([w, pad], dim=1)
        padded.append(w)

    stacked = torch.stack(padded, dim=0)  # [N, C, S]
    return {"waveform": stacked, "sample_rate": base_sr}


class Soze_JSONAudioPair:
    """Add a `name: <encoded-audio>` pair to the JSON chain. Uses the first
    frame of the AUDIO batch.

    Modes (same as the image counterpart):
    - `data_uri`: emits `data:audio/mpeg;base64,...` — self-contained, MP3-compressed.
    - `filepath`: saves the audio to disk as MP3 and emits the absolute file path.

    Encoding requires torchaudio with an ffmpeg backend (which ComfyUI normally has).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "audio": ("AUDIO", {"forceInput": True}),
                "mode": (JSON_AUDIO_MODES, {"default": "data_uri"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "filepath mode: target path (relative to ComfyUI output dir, or absolute). `.mp3` is added if missing."}),
                "append_suffix": ("BOOLEAN", {"default": True, "tooltip": "filepath mode: append _00001 if the target already exists."}),
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, audio, mode, save_path="", append_suffix=True, chain_in=None):
        if audio is None:
            return _append_pair(chain_in, name, None)
        waveform, sr = next(_audio_dict_to_frames(audio))
        value = _encode_audio_frame(waveform, sr, mode, save_path, append_suffix)
        return _append_pair(chain_in, name, value)


class Soze_JSONAudioEncoder:
    """Encode an AUDIO into a STRING (MP3) that can be stored as a JSON value.

    Uses the first frame of the batch. Use the array variant if you need
    multiple frames encoded as a JSON array.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"forceInput": True}),
                "mode": (JSON_AUDIO_MODES, {"default": "data_uri"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "", "multiline": False}),
                "append_suffix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("encoded",)
    FUNCTION = "encode"
    CATEGORY = "Soze Nodes/JSON"

    def encode(self, audio, mode, save_path="", append_suffix=True):
        if audio is None:
            return ("",)
        waveform, sr = next(_audio_dict_to_frames(audio))
        return (_encode_audio_frame(waveform, sr, mode, save_path, append_suffix),)


class Soze_JSONAudioArrayPair:
    """Add a `name: [<encoded-audio>, ...]` MP3 array pair to the JSON chain —
    one entry per frame in the AUDIO batch. (ComfyUI AUDIO has shape
    [B, C, S]; this iterates over the B dimension.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "multiline": False}),
                "audio": ("AUDIO", {"forceInput": True}),
                "mode": (JSON_AUDIO_MODES, {"default": "data_uri"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "", "multiline": False, "tooltip": "filepath mode: each frame gets its own _NNNNN suffix."}),
                "chain_in": (JSON_PAIRS_TYPE,),
            },
        }

    RETURN_TYPES = (JSON_PAIRS_TYPE,)
    RETURN_NAMES = ("chain_out",)
    FUNCTION = "build"
    CATEGORY = "Soze Nodes/JSON"

    def build(self, name, audio, mode, save_path="", chain_in=None):
        if audio is None:
            return _append_pair(chain_in, name, [])
        values = []
        for waveform, sr in _audio_dict_to_frames(audio):
            # filepath mode: always counter-suffix per frame to avoid overwrites.
            values.append(_encode_audio_frame(waveform, sr, mode, save_path, append_suffix=True))
        return _append_pair(chain_in, name, values)


class Soze_JSONAudioDecoder:
    """Inverse of `JSON Audio Encoder`. Takes a STRING and returns an AUDIO.

    Accepts whatever the encoder (or `JSON Audio Array Pair`, or other STRING
    outputs) might produce:

    * a single `data:audio/mpeg;base64,...` URI (from the encoder)
    * a single `data:audio/<wav|flac|...>;base64,...` URI (legacy/external)
    * a single file path (absolute or relative to current working dir)
    * a single `http(s)://...` URL
    * a JSON-encoded array of any of the above
    * newline-separated values

    Multiple values are stacked into one AUDIO batch. Differing sample rates
    are resampled to match the first frame. Frames whose channel count
    doesn't match the first are skipped (status records this). Shorter
    waveforms are zero-padded to match the longest length.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "encoded": ("STRING", {"default": "", "multiline": True, "tooltip": "Data URI, file path, URL, JSON array, or newline-separated list."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "count", "status")
    FUNCTION = "decode"
    CATEGORY = "Soze Nodes/JSON"

    def decode(self, encoded, unique_id=None):
        blank = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
        if not encoded or not str(encoded).strip():
            status = "Skipped: empty encoded input."
            push_node_status(unique_id, status)
            return (blank, 0, status)

        values = _parse_image_string_list(encoded)  # same parser handles all list forms
        if not values:
            status = "Skipped: no values parsed."
            push_node_status(unique_id, status)
            return (blank, 0, status)

        push_node_status(unique_id, f"Decoding {len(values)} audio value(s)...")
        decoded = []
        failures = []
        for i, v in enumerate(values):
            try:
                decoded.append(_decode_one_audio_value(v))
            except Exception as e:
                logger.exception("JSONAudioDecoder: failed to decode value %d", i)
                failures.append(f"value[{i}]: {e!r}")

        if not decoded:
            details = "; ".join(failures) or "no decoder error captured"
            status = f"ERROR: every value failed to decode. {details}"
            push_node_status(unique_id, status)
            return (blank, 0, status)

        result = _align_decoded_audios(decoded)
        wf = result["waveform"]
        n_frames, n_channels, n_samples = wf.shape[0], wf.shape[1], wf.shape[2]
        sr = result["sample_rate"]
        duration_sec = n_samples / sr if sr else 0.0
        status = (
            f"OK: decoded {len(decoded)}/{len(values)} value(s), "
            f"{n_frames} frame(s) x {n_channels}ch, {duration_sec:.2f}s @ {sr}Hz"
        )
        if failures:
            status += " | failures: " + "; ".join(failures)
        push_node_status(unique_id, status)
        return (result, len(decoded), status)
