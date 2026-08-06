"""Enum-switch model loaders.

Each node takes a string `enum_value` and 10 (compare_text, model) pairs.
Only the model in the matching slot is actually loaded — the others are
strings on disk paths, never instantiated. If no case matches, a fallback
`default_model` is used (or, for the LoRA variant, the inputs pass through
unchanged with no LoRA applied).

Matching is case-insensitive + whitespace-trimmed by default; toggle the
booleans to make it strict.

Variants:
  - Soze_EnumSwitchCheckpointLoader        -> (MODEL, CLIP, VAE)
  - Soze_EnumSwitchUpscaleModelLoader      -> (UPSCALE_MODEL)
  - Soze_EnumSwitchLoraLoader              -> (MODEL, CLIP)  (passthrough on no match)
  - Soze_EnumSwitchDiffusionModelLoader    -> (MODEL)
"""

import folder_paths

from .status_utils import EventLog, push_node_status, finalize_status


MAX_ENUM_SWITCH_CASES = 10


def _select_case(enum_value, kwargs, max_cases, case_sensitive=False, strip=True):
    """Return (matched_index, matched_label).

    Index is 1-based for a hit, 0 for no match. Blank compare_text_N entries
    are skipped (so an unconnected/empty widget can't accidentally match).
    """
    key = enum_value if isinstance(enum_value, str) else str(enum_value or "")
    if strip:
        key = key.strip()
    key_cmp = key if case_sensitive else key.casefold()
    for i in range(1, max_cases + 1):
        label = kwargs.get(f"compare_text_{i}", "")
        if label is None:
            continue
        label_str = label if isinstance(label, str) else str(label)
        if strip:
            label_str = label_str.strip()
        if not label_str:
            continue
        label_cmp = label_str if case_sensitive else label_str.casefold()
        if label_cmp == key_cmp:
            return i, label_str
    return 0, None


# Special enum value meaning "do nothing this run" — active only when the
# node's allow_none toggle is on. The Copy-Names button prepends "none" to the
# clipboard array when allow_none is enabled.
NONE_SENTINEL = "none"


def _allow_none_widget():
    return ("BOOLEAN", {
        "default": False,
        "tooltip": "If enabled, an enum_value of 'none' (the first item in the "
                   "copied compare-text array) skips loading: the LoRA loader "
                   "passes (model, clip) through unchanged; the other loaders "
                   "output nothing for that run.",
    })


def _is_none_request(enum_value, allow_none, strip=True):
    """True when allow_none is on and enum_value is the 'none' sentinel.

    Always matched case-insensitively (the sentinel is a fixed magic word),
    honoring only the strip_whitespace setting.
    """
    if not allow_none:
        return False
    v = enum_value if isinstance(enum_value, str) else str(enum_value or "")
    if strip:
        v = v.strip()
    return v.casefold() == NONE_SENTINEL


# Blank sentinel placed first in every model dropdown so unused slots default
# to "no model" instead of force-picking the first real file.
BLANK_MODEL_CHOICE = ""


def _common_optional(folder_filenames, include_default=True):
    """Build the shared optional-input block: switches + 10 (compare, model) pairs.

    `folder_filenames` is the dropdown list (e.g. checkpoints, loras...).
    A blank entry is prepended to every model dropdown and used as the default.
    """
    choices = [BLANK_MODEL_CHOICE] + list(folder_filenames)
    opt = {
        "case_sensitive": ("BOOLEAN", {"default": False, "tooltip": "If True, 'A' != 'a'."}),
        "strip_whitespace": ("BOOLEAN", {"default": True, "tooltip": "Trim whitespace before comparing."}),
    }
    if include_default:
        opt["default_model"] = (choices, {
            "default": BLANK_MODEL_CHOICE,
            "tooltip": "Loaded when no compare_text_N matches enum_value. Blank = none.",
        })
    for i in range(1, MAX_ENUM_SWITCH_CASES + 1):
        opt[f"compare_text_{i}"] = ("STRING", {
            "default": "",
            "tooltip": f"Case {i} label. Blank = skip this case.",
        })
        opt[f"model_{i}"] = (choices, {
            "default": BLANK_MODEL_CHOICE,
            "tooltip": f"Model loaded when compare_text_{i} matches enum_value. Blank = unused.",
        })
    return opt


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class Soze_EnumSwitchCheckpointLoader:
    """Enum-switch loader: picks one of 10 checkpoints by string match and loads it."""

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = folder_paths.get_filename_list("checkpoints")
        opt = _common_optional(ckpts, include_default=True)
        opt["allow_none"] = _allow_none_widget()
        return {
            "required": {
                "enum_value": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "String to match against each compare_text_N.",
                }),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name", "matched_index", "status")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, enum_value, default_model=None, case_sensitive=False,
             strip_whitespace=True, allow_none=False, unique_id=None, **kwargs):
        log = EventLog()
        if _is_none_request(enum_value, allow_none, strip=strip_whitespace):
            headline = "none requested; skipping checkpoint load (MODEL/CLIP/VAE are None)."
            push_node_status(unique_id, headline, log)
            return (None, None, None, "", 0, finalize_status(headline, log))
        idx, label = _select_case(enum_value, kwargs, MAX_ENUM_SWITCH_CASES,
                                  case_sensitive=case_sensitive,
                                  strip=strip_whitespace)
        if idx:
            chosen = kwargs.get(f"model_{idx}")
            push_node_status(unique_id, f"matched case {idx} ({label!r}) -> {chosen}", log)
        else:
            chosen = default_model
            push_node_status(unique_id, f"no match for {enum_value!r}; using default {chosen!r}", log)

        if not chosen:
            headline = "ERROR: no checkpoint selected (matched slot is blank, or no match and no default_model)."
            push_node_status(unique_id, headline, log)
            raise ValueError(headline)

        import comfy.sd
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", chosen)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = out[0], out[1], out[2]
        headline = f"OK: loaded {chosen}"
        push_node_status(unique_id, headline, log)
        return (model, clip, vae, chosen, idx, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Upscale model
# ---------------------------------------------------------------------------


class Soze_EnumSwitchUpscaleModelLoader:
    """Enum-switch loader: picks one of 10 upscale models by string match and loads it."""

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("upscale_models")
        opt = _common_optional(models, include_default=False)
        opt["allow_none"] = _allow_none_widget()
        return {
            "required": {
                "enum_value": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "String to match against each compare_text_N.",
                }),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("UPSCALE_MODEL", "STRING", "INT", "STRING")
    RETURN_NAMES = ("UPSCALE_MODEL", "model_name", "matched_index", "status")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, enum_value, case_sensitive=False,
             strip_whitespace=True, allow_none=False, unique_id=None, **kwargs):
        log = EventLog()
        if _is_none_request(enum_value, allow_none, strip=strip_whitespace):
            headline = "none requested; skipping upscale model load (UPSCALE_MODEL is None)."
            push_node_status(unique_id, headline, log)
            return (None, "", 0, finalize_status(headline, log))
        idx, label = _select_case(enum_value, kwargs, MAX_ENUM_SWITCH_CASES,
                                  case_sensitive=case_sensitive,
                                  strip=strip_whitespace)
        if idx:
            chosen = kwargs.get(f"model_{idx}")
            push_node_status(unique_id, f"matched case {idx} ({label!r}) -> {chosen}", log)
        else:
            chosen = None
            push_node_status(unique_id, f"no match for {enum_value!r}", log)

        if not chosen:
            headline = "ERROR: no upscale model selected (no match, or matched slot is blank)."
            push_node_status(unique_id, headline, log)
            raise ValueError(headline)

        # Delegate to ComfyUI's built-in upscale loader to stay version-compat.
        try:
            from comfy_extras.nodes_upscale_model import UpscaleModelLoader
        except ImportError as e:
            headline = f"ERROR: UpscaleModelLoader unavailable: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        (model_obj,) = UpscaleModelLoader().load_model(chosen)
        headline = f"OK: loaded {chosen}"
        push_node_status(unique_id, headline, log)
        return (model_obj, chosen, idx, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class Soze_EnumSwitchLoraLoader:
    """Enum-switch LoRA loader: picks one of 10 LoRAs by string match and
    applies it to (model, clip). Passthrough if no case matches."""

    def __init__(self):
        # Reuse the built-in LoraLoader's per-instance cache.
        self._delegate = None

    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        opt = _common_optional(loras, include_default=False)
        # LoRA-specific extras
        opt["strength_model"] = ("FLOAT", {
            "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
            "tooltip": "How strongly to modify the diffusion model. Negative values are allowed.",
        })
        opt["strength_clip"] = ("FLOAT", {
            "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
            "tooltip": "How strongly to modify the CLIP model. Negative values are allowed.",
        })
        opt["allow_none"] = _allow_none_widget()
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "Base CLIP model the LoRA will be applied to."}),
                "enum_value": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "String to match against each compare_text_N.",
                }),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "INT", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_name", "matched_index", "status")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, model, clip, enum_value, case_sensitive=False,
             strip_whitespace=True, strength_model=1.0, strength_clip=1.0,
             allow_none=False, unique_id=None, **kwargs):
        log = EventLog()
        if _is_none_request(enum_value, allow_none, strip=strip_whitespace):
            headline = "none requested; passthrough (no LoRA applied)."
            push_node_status(unique_id, headline, log)
            return (model, clip, "", 0, finalize_status(headline, log))
        idx, label = _select_case(enum_value, kwargs, MAX_ENUM_SWITCH_CASES,
                                  case_sensitive=case_sensitive,
                                  strip=strip_whitespace)
        if idx == 0:
            headline = f"No match for {enum_value!r}; passthrough (no LoRA applied)."
            push_node_status(unique_id, headline, log)
            return (model, clip, "", 0, finalize_status(headline, log))

        chosen = kwargs.get(f"model_{idx}")
        if not chosen:
            headline = f"Matched case {idx} but no LoRA selected; passthrough."
            push_node_status(unique_id, headline, log)
            return (model, clip, "", idx, finalize_status(headline, log))

        if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
            headline = f"Matched case {idx} ({label!r}) but both strengths are 0; passthrough."
            push_node_status(unique_id, headline, log)
            return (model, clip, chosen, idx, finalize_status(headline, log))

        push_node_status(unique_id, f"matched case {idx} ({label!r}) -> {chosen}", log)

        try:
            from nodes import LoraLoader
        except ImportError as e:
            headline = f"ERROR: LoraLoader unavailable: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        if self._delegate is None:
            self._delegate = LoraLoader()
        model_out, clip_out = self._delegate.load_lora(
            model, clip, chosen, float(strength_model), float(strength_clip)
        )
        headline = f"OK: applied {chosen} (m={strength_model}, c={strength_clip})"
        push_node_status(unique_id, headline, log)
        return (model_out, clip_out, chosen, idx, finalize_status(headline, log))


# ---------------------------------------------------------------------------
# Diffusion model (UNet)
# ---------------------------------------------------------------------------


class Soze_EnumSwitchDiffusionModelLoader:
    """Enum-switch loader: picks one of 10 diffusion (UNet) models by string match."""

    @classmethod
    def INPUT_TYPES(cls):
        models = folder_paths.get_filename_list("diffusion_models")
        opt = _common_optional(models, include_default=True)
        opt["weight_dtype"] = (
            ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
            {"default": "default", "tooltip": "Cast the model weights to this dtype on load."},
        )
        opt["allow_none"] = _allow_none_widget()
        return {
            "required": {
                "enum_value": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "String to match against each compare_text_N.",
                }),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "STRING", "INT", "STRING")
    RETURN_NAMES = ("MODEL", "model_name", "matched_index", "status")
    FUNCTION = "load"
    CATEGORY = "loaders"

    def load(self, enum_value, default_model=None, weight_dtype="default",
             case_sensitive=False, strip_whitespace=True, allow_none=False,
             unique_id=None, **kwargs):
        log = EventLog()
        if _is_none_request(enum_value, allow_none, strip=strip_whitespace):
            headline = "none requested; skipping diffusion model load (MODEL is None)."
            push_node_status(unique_id, headline, log)
            return (None, "", 0, finalize_status(headline, log))
        idx, label = _select_case(enum_value, kwargs, MAX_ENUM_SWITCH_CASES,
                                  case_sensitive=case_sensitive,
                                  strip=strip_whitespace)
        if idx:
            chosen = kwargs.get(f"model_{idx}")
            push_node_status(unique_id, f"matched case {idx} ({label!r}) -> {chosen}", log)
        else:
            chosen = default_model
            push_node_status(unique_id, f"no match for {enum_value!r}; using default {chosen!r}", log)

        if not chosen:
            headline = "ERROR: no diffusion model selected (matched slot is blank, or no match and no default_model)."
            push_node_status(unique_id, headline, log)
            raise ValueError(headline)

        try:
            from nodes import UNETLoader
        except ImportError as e:
            headline = f"ERROR: UNETLoader unavailable: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        (model_obj,) = UNETLoader().load_unet(chosen, weight_dtype)
        headline = f"OK: loaded {chosen} ({weight_dtype})"
        push_node_status(unique_id, headline, log)
        return (model_obj, chosen, idx, finalize_status(headline, log))
