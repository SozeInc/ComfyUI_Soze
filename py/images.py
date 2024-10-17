import torch
import os
import re
import numpy as np
import hashlib

from PIL import Image, ImageOps, ImageSequence, ImageFile
from comfy.utils import ProgressBar, common_upscale

import node_helpers
import folder_paths

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)



class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_NAMES = ("Image", "Mask", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext")
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING")
    FUNCTION = "load_image"
    def load_image(self, image):
        input_filepath = folder_paths.get_annotated_filepath(image)
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]

        img = node_helpers.pillow(Image.open, input_filepath)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask, input_filepath, input_filename, input_filename_no_ext)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True


# Code from https://github.com/kijai/ComfyUI-KJNodes 
# Added filename outputs etc
class LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Input_Folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "Image_Load_Count": ("INT", {"default": 1, "min": 0, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Load_Count", "Input_Path",  "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext")
    FUNCTION = "load_images"

    CATEGORY = "image"

    def load_images(self, Input_Folder, Image_Load_Count, seed):
        if not os.path.isdir(Input_Folder):
            raise FileNotFoundError(f"Folder not found: {Input_Folder}")
        dir_files = os.listdir(Input_Folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {Input_Folder}")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(Input_Folder, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[seed:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if Image_Load_Count > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= Image_Load_Count:
                break
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
                has_non_empty_mask = True
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            images.append(image)
            masks.append(mask)
            image_path_list.append(image_path)
            image_count += 1

        if len(images) == 1:

            input_filenamepath = folder_paths.get_annotated_filepath(image_path_list[0])
            input_filename = os.path.basename(input_filenamepath)
            input_filename_no_ext = os.path.splitext(input_filename)[0]
            return (images[0], masks[0], 1, Input_Folder, input_filenamepath, input_filename, input_filename_no_ext)

        elif len(images) > 1:
            image1 = images[0]
            mask1 = None

            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                if has_non_empty_mask:
                    if image1.shape[1:3] != mask2.shape:
                        mask2 = torch.nn.functional.interpolate(mask2.unsqueeze(0).unsqueeze(0), size=(image1.shape[2], image1.shape[1]), mode='bilinear', align_corners=False)
                        mask2 = mask2.squeeze(0)
                    else:
                        mask2 = mask2.unsqueeze(0)
                else:
                    mask2 = mask2.unsqueeze(0)

                if mask1 is None:
                    mask1 = mask2
                else:
                    mask1 = torch.cat((mask1, mask2), dim=0)

            input_filenamepath = folder_paths.get_annotated_filepath(image_path_list[0])
            input_filename = os.path.basename(input_filenamepath)
            input_filename_no_ext = os.path.splitext(input_filename)[0]

            return (image1, mask1, len(images), Input_Folder, input_filenamepath, input_filename, input_filename_no_ext)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")


# Code from https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
#Added filename outputs etc
class BatchProcessSwitch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input": (["Image", "Image Batch"],),
            },
            "optional": {
                "Image": ("IMAGE", ),
                "Image_Batch": ("IMAGE", ),
                "Image_Filename_Path_Passthrough": ("STRING", {"default": "", "forceInput": True}),
                "Image_Batch_Filename_Path_Passthrough": ("STRING", {"default": "", "forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext")
    FUNCTION = "switch"
    CATEGORY = "Process"

    def switch(self, Input, Image=None, Image_Batch=None, Image_Filename_Path_Passthrough="", Image_Batch_Filename_Path_Passthrough=""):
        input_filenamepath = ""
        input_filename = ""
        input_filename_no_ext = ""

        if Input == "Image":
            if Image_Filename_Path_Passthrough != "":
                try: input_filenamepath = folder_paths.get_annotated_filepath(Image_Filename_Path_Passthrough) 
                except Exception: pass
                try: input_filename = os.path.basename(input_filenamepath)
                except Exception: pass
                try: input_filename_no_ext = os.path.splitext(input_filename)[0]
                except Exception: pass
            return (Image, input_filenamepath, input_filename, input_filename_no_ext)
        else:
            if Image_Batch_Filename_Path_Passthrough != "":
                try: input_filenamepath = folder_paths.get_annotated_filepath(Image_Batch_Filename_Path_Passthrough)
                except Exception: pass
                try: input_filename = os.path.basename(input_filenamepath)
                except Exception: pass
                try: input_filename_no_ext = os.path.splitext(input_filename)[0]
                except Exception: pass
            return (Image_Batch, input_filenamepath, input_filename, input_filename_no_ext)
    
import requests

#Code From https://github.com/melMass/comfy_mtb
class LoadImageFromUrl:
    """Load an image from the given URL"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": ""
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "Image_Filename", "Image_Filename_No_Ext")
    FUNCTION = "load"
    CATEGORY = "images"

    def load(self, url):
        # Get the image from the url
        response = requests.get(url, stream=True)
        image = Image.open(response.raw)
        image = ImageOps.exif_transpose(image)

        # Extract filename from URL
        filename = os.path.basename(url)
        filename_no_ext = os.path.splitext(filename)[0]

        return (
            pil2tensor(image),
            filename,
            filename_no_ext
        )
