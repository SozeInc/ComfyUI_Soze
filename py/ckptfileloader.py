import torch
import folder_paths
import comfy.utils




class Soze_CheckpointFilePathLoader:
    def __init__(self):
        pass

    previous_output = None
    previous_ckpt_path = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "undefined"}),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.",
                       "The CLIP model used for encoding text prompts.",
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_path):
        if ckpt_path != self.previous_ckpt_path or self.previous_ckpt_path is None:
            self.previous_ckpt_path = ckpt_path
            self.previous_output = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            return self.previous_output
        return self.previous_output