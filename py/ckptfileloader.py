import os

import comfy.sd
import folder_paths

from .status_utils import EventLog, push_node_status, finalize_status


class Soze_CheckpointFilePathLoader:
    """Load a checkpoint by absolute path. ComfyUI's model-management layer
    handles caching upstream, so this node is a thin wrapper around
    comfy.sd.load_checkpoint_guess_config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "undefined"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "status")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
        "Status text — current checkpoint name, size, and load result.",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint from an absolute filesystem path."

    def load_checkpoint(self, ckpt_path, unique_id=None):
        log = EventLog()
        ckpt_name = os.path.basename(ckpt_path) if ckpt_path else "(empty)"
        push_node_status(unique_id, f"Loading checkpoint: {ckpt_name}", log)

        if not ckpt_path or not os.path.isfile(ckpt_path):
            headline = f"ERROR: checkpoint not found at {ckpt_path}"
            push_node_status(unique_id, headline, log)
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        try:
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            push_node_status(unique_id, f"Size: {size_mb:.1f} MB", log)
        except OSError:
            pass

        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        # Comfy's loader returns (model, clip, vae, clip_vision); we only expose 3.
        headline = f"OK: loaded {ckpt_name}"
        push_node_status(unique_id, headline, log)
        return tuple(list(out[:3]) + [finalize_status(headline, log)])
