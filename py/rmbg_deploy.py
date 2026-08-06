"""Adapter node: feed a model name into ComfyUI-RMBG's BiRefNet `model` selector.

BiRefNet's `model` input is a COMBO (dropdown). To drive it from a string source
— e.g. a ComfyDeploy "External Text" input exposed via the deployment API — you
convert BiRefNet's `model` widget to an input and wire this node's output into
it. This node outputs a wildcard ("*") type so it connects to that COMBO input
regardless of how the combo's type is declared.

The valid model names are hardcoded to match ComfyUI-RMBG's MODEL_CONFIG keys
(py/AILab_BiRefNet.py). If that pack adds models, update BIREFNET_MODELS below.
"""

from .status_utils import EventLog, push_node_status, finalize_status


# Hardcoded to mirror ComfyUI-RMBG MODEL_CONFIG keys, in source order.
BIREFNET_MODELS = [
    "BiRefNet-general",
    "BiRefNet_512x512",
    "BiRefNet-HR",
    "BiRefNet-portrait",
    "BiRefNet-matting",
    "BiRefNet-HR-matting",
    "BiRefNet_lite",
    "BiRefNet_lite-2K",
    "BiRefNet_dynamic",
    "BiRefNet_lite-matting",
    "BiRefNet_toonout",
]

# Lowercase lookup -> canonical name, for forgiving (case/space-insensitive) match.
_CANON = {m.casefold(): m for m in BIREFNET_MODELS}


class _AnyType(str):
    """Wildcard type that compares equal to any other type in ComfyUI."""
    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class Soze_ComfyDeployBiRefNetModelInput:
    """Resolve a BiRefNet model name and emit it as a wildcard for BiRefNet's
    `model` input. Wire a ComfyDeploy External Text node into `model_override`
    to drive the selection from the deployment API; otherwise the dropdown
    `model` value is used."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (BIREFNET_MODELS, {
                    "default": BIREFNET_MODELS[0],
                    "tooltip": "Default BiRefNet model. Used when model_override is empty/unconnected.",
                }),
            },
            "optional": {
                "model_override": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Optional model name (e.g. from a ComfyDeploy External Text input). "
                               "If non-empty, overrides the dropdown. Case/whitespace-insensitive.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # Wildcard output connects to BiRefNet's COMBO `model` input (converted to input).
    RETURN_TYPES = (_ANY, "STRING", "STRING")
    RETURN_NAMES = ("model", "model_name", "status")
    FUNCTION = "resolve"
    CATEGORY = "Soze Nodes"

    def resolve(self, model, model_override="", unique_id=None):
        log = EventLog()

        raw = (model_override or "").strip()
        source = "override" if raw else "dropdown"
        candidate = raw if raw else model

        canonical = _CANON.get((candidate or "").strip().casefold())
        if canonical is None:
            headline = (
                f"ERROR: unknown BiRefNet model {candidate!r}. "
                f"Valid options: {', '.join(BIREFNET_MODELS)}"
            )
            push_node_status(unique_id, headline, log)
            raise ValueError(headline)

        note = ""
        if raw and canonical != candidate:
            note = f" (normalized from {candidate!r})"
        headline = f"OK: {canonical} (from {source}){note}"
        push_node_status(unique_id, headline, log)
        return (canonical, canonical, finalize_status(headline, log))
