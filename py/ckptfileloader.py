import comfy.sd
import folder_paths


class Soze_CheckpointFilePathLoader:
    """Load a checkpoint by absolute path. ComfyUI's model-management layer
    handles caching upstream, so this node is a thin wrapper around
    comfy.sd.load_checkpoint_guess_config."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "undefined"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint from an absolute filesystem path."

    def load_checkpoint(self, ckpt_path):
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        # Comfy's loader returns (model, clip, vae, clip_vision); we only expose 3.
        return out[:3]
