import os

import comfy.utils
import comfy.sd

from .status_utils import EventLog, push_node_status, finalize_status


class Soze_LoraFilePathLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_filepath": ("STRING", {"default": ""}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "status")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.", "Status text — LoRA name, strengths, and cache hit/miss.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_filepath, strength_model, strength_clip, unique_id=None):
        log = EventLog()
        lora_name = os.path.basename(lora_filepath) if lora_filepath else "(empty)"
        push_node_status(unique_id, f"LoRA: {lora_name} (model={strength_model}, clip={strength_clip})", log)

        if strength_model == 0 and strength_clip == 0:
            headline = "Skipped: both strengths are 0 (passthrough)."
            push_node_status(unique_id, headline, log)
            return (model, clip, finalize_status(headline, log))

        cache_hit = False
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_filepath:
                lora = self.loaded_lora[1]
                cache_hit = True
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            if not lora_filepath or not os.path.isfile(lora_filepath):
                headline = f"ERROR: LoRA not found at {lora_filepath}"
                push_node_status(unique_id, headline, log)
                raise FileNotFoundError(f"LoRA not found: {lora_filepath}")
            push_node_status(unique_id, "Cache miss — loading from disk.", log)
            lora = comfy.utils.load_torch_file(lora_filepath, safe_load=True)
            self.loaded_lora = (lora_filepath, lora)
        else:
            push_node_status(unique_id, "Cache hit — reusing previously loaded LoRA.", log)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        headline = f"OK: applied {lora_name} ({'cached' if cache_hit else 'loaded'}; m={strength_model}, c={strength_clip})"
        push_node_status(unique_id, headline, log)
        return (model_lora, clip_lora, finalize_status(headline, log))
