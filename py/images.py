import torch
import os
import random
import re
import numpy as np
import hashlib
import re
import requests
import ast
import json
from numpy import ndarray
from comfy.cli_args import args
from PIL.PngImagePlugin import PngInfo

from typing import Tuple, List, Dict, Any, Optional

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, TYPE_CHECKING

from PIL import ImageFont, ImageDraw, Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.font_manager as fm
from torch import Tensor

from .status_utils import EventLog, push_node_status, finalize_status
from .utils import (
    zip_with_fill,
    tensor2pil,
    pil2tensor,
    read_from_file,
    write_to_file
)
if TYPE_CHECKING:
    from mypy.typeshed.stdlib._typeshed import SupportsDunderGT, SupportsDunderLT

from PIL import Image, ImageOps, ImageSequence, ImageFile
from comfy.utils import ProgressBar, common_upscale

import node_helpers
import folder_paths


def _folder_not_found_hint(path: str) -> str:
    """Explain *why* a folder path failed isdir(), to speed up diagnosis."""
    try:
        if os.path.isfile(path):
            return " (this is a file, not a folder)"
        parent = os.path.dirname(path.rstrip("/\\"))
        if parent and os.path.isdir(parent):
            return f" (parent {parent!r} exists, but this subfolder does not)"
        if parent and not os.path.exists(parent):
            return (f" (parent {parent!r} is also missing — is the drive/mount "
                    "accessible from where ComfyUI runs? A path from another "
                    "machine/OS won't resolve here)")
    except Exception:
        pass
    return ""


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



class Soze_LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {"image": (sorted(files), {"image_upload": True})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "image"

    RETURN_NAMES = ("Image", "Mask", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Image_Changed", "status")
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "BOOL", "STRING")
    FUNCTION = "load_image"
    def load_image(self, image, unique_id=None):
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

        previous_input_filename = read_from_file('sozeimagecache.txt')
        write_to_file('sozeimagecache.txt', input_filename)
        changed = previous_input_filename != input_filename
        try:
            h, w = output_image.shape[1], output_image.shape[2]
        except Exception:
            h, w = 0, 0
        status = f"OK: {input_filename} ({w}x{h}, frames={len(output_images)}, changed={changed})"
        push_node_status(unique_id, status)
        return (output_image, output_mask, input_filepath, input_filename, input_filename_no_ext, changed, status)

    @classmethod
    def IS_CHANGED(s, image, unique_id=None):
        previous_input_filepath = s.read_previous_image_filename()
        if previous_input_filepath != image:
            return True
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, unique_id=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)

        return True
    
    def read_previous_image_filename():
        try:
            return read_from_file('sozeimagecache.txt')
        except Exception:
            return ""
    
# Code from https://github.com/kijai/ComfyUI-KJNodes 
# Added filename outputs etc
class Soze_LoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Input_Folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "Image_Load_Count": ("INT", {"default": 1, "min": 0, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "STRING", "STRING", "BOOL", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Load_Count", "Input_Path",  "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Image_Changed", "status")
    FUNCTION = "load_images"

    CATEGORY = "image"

    def load_images(self, Input_Folder, Image_Load_Count, index, unique_id=None):
        log = EventLog()
        # Clean the path: trailing whitespace/newlines and surrounding quotes are
        # a very common copy-paste artifact that makes a valid path fail isdir().
        Input_Folder = (Input_Folder or "").strip().strip('"').strip("'").rstrip()
        push_node_status(unique_id, f"Scanning {Input_Folder}", log)
        if not os.path.isdir(Input_Folder):
            headline = f"ERROR: folder not found: {Input_Folder!r}{_folder_not_found_hint(Input_Folder)}"
            push_node_status(unique_id, headline, log)
            raise FileNotFoundError(headline)
        dir_files = os.listdir(Input_Folder)
        if len(dir_files) == 0:
            push_node_status(unique_id, f"ERROR: folder is empty: {Input_Folder}", log)
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {Input_Folder}")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(Input_Folder, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if Image_Load_Count > 0:
            limit_images = True
        image_count = 0

        excluded_formats = ['MPO']

        for image_path in dir_files:
            if os.path.isdir(image_path):
                continue
            if limit_images and image_count >= Image_Load_Count:
                break

            # image_path is already an absolute path (folder + filename); do NOT
            # route it through get_annotated_filepath, which is only for
            # input-dir-relative annotated names and rejects arbitrary paths.
            input_filepath = image_path
            input_filename = os.path.basename(input_filepath)
            input_filename_no_ext = os.path.splitext(input_filename)[0]

            img = node_helpers.pillow(Image.open, input_filepath)

            output_images = []
            output_masks = []
            w, h = None, None

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

            images.append(output_image)
            masks.append(output_mask)
            image_path_list.append(input_filepath)
            image_count += 1

        if len(images) == 1:
            input_filenamepath = image_path_list[0]
            input_filename = os.path.basename(input_filenamepath)
            input_filename_no_ext = os.path.splitext(input_filename)[0]
            headline = f"OK: 1 image at index {index} — {input_filename}"
            push_node_status(unique_id, headline, log)
            return (images[0], masks[0], 1, Input_Folder, input_filenamepath, input_filename, input_filename_no_ext, True, finalize_status(headline, log))

        elif len(images) > 1:
            image1 = images[0]
            mask1 = masks[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)

            for mask2 in masks[1:]:
                # Compare spatial dims only — mask1's batch dim grows each cat.
                if mask1.shape[1:] != mask2.shape[1:]:
                    # mask2 is [1, H, W]; interpolate needs 4D [N, C, H, W].
                    mask2 = torch.nn.functional.interpolate(
                        mask2.unsqueeze(0),
                        size=(mask1.shape[-2], mask1.shape[-1]),
                        mode='bilinear', align_corners=False,
                    ).squeeze(0)
                mask1 = torch.cat((mask1, mask2), dim=0)

            input_filenamepath = image_path_list[0]
            input_filename = os.path.basename(input_filenamepath)
            input_filename_no_ext = os.path.splitext(input_filename)[0]

            previous_input_filename = read_from_file('sozeimagebatchcache.txt')
            write_to_file('sozeimagebatchcache.txt', input_filename)
            changed = previous_input_filename != input_filename

            headline = f"OK: {len(images)} images from index {index}, first={input_filename}"
            push_node_status(unique_id, headline, log)
            return (image1, mask1, len(images), Input_Folder, input_filenamepath, input_filename, input_filename_no_ext, changed, finalize_status(headline, log))

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")

class Soze_LoadImageFromFilepath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image_Filepath": ("STRING", {"default": ""}),
                "Return_None_If_Not_Found": ("BOOLEAN", {"default": False, "tooltip": "If enabled, will return empty outputs instead of erroring if the file is not found."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "BOOL", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Input_Path",  "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Image_Changed", "status")
    FUNCTION = "load_image_from_filepath"

    CATEGORY = "image"

    def load_image_from_filepath(self, Image_Filepath, Return_None_If_Not_Found=False, unique_id=None):
        if not os.path.isfile(Image_Filepath) and not os.path.exists(Image_Filepath):
            if Return_None_If_Not_Found:
                msg = f"Skipped: file not found ({Image_Filepath}); Return_None_If_Not_Found=True"
                push_node_status(unique_id, msg)
                return (None, None, "", "", "", "", False, msg)
            push_node_status(unique_id, f"ERROR: file not found: {Image_Filepath}")
            raise FileNotFoundError(f"File not found: {Image_Filepath}")

        # Image_Filepath is an arbitrary absolute path (validated to exist
        # above); use it directly rather than get_annotated_filepath, which
        # only accepts input-dir-relative annotated names.
        input_filepath = Image_Filepath
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

        try:
            h, wd = output_image.shape[1], output_image.shape[2]
        except Exception:
            h, wd = 0, 0
        status = f"OK: {input_filename} ({wd}x{h}, frames={len(output_images)})"
        push_node_status(unique_id, status)
        return (output_image, output_mask, os.path.dirname(input_filepath), input_filepath, input_filename, input_filename_no_ext, False, status)




class Soze_LoadImagesFromFolderXLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Input_Folder": ("STRING", {"default": ""}),
                "start_lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the starting LoRA."}),
                "lora_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "The number of LoRAs to load."})
            },
            "optional": {
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Input_Path",  "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Lora_Full_Path", "Lora_Name_Only", "Lora_Index", "status")
    FUNCTION = "load_images"

    CATEGORY = "image"

    def load_images(self, Input_Folder, index, start_lora_name, lora_count, unique_id=None):
        log = EventLog()
        Input_Folder = (Input_Folder or "").strip().strip('"').strip("'").rstrip()
        push_node_status(unique_id, f"Folder: {Input_Folder}", log)
        if not os.path.isdir(Input_Folder):
            push_node_status(unique_id, f"ERROR: folder not found: {Input_Folder}", log)
            raise FileNotFoundError(f"Folder not found: {Input_Folder}")
        dir_files = os.listdir(Input_Folder)
        if len(dir_files) == 0:
            push_node_status(unique_id, f"ERROR: folder is empty: {Input_Folder}", log)
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {Input_Folder}")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(Input_Folder, x) for x in dir_files]

        images = []
        masks = []
        image_path_list = []

        excluded_formats = ['MPO']

        # Calculate which image and lora to use
        num_images = len(dir_files)
        if num_images == 0:
            push_node_status(unique_id, f"ERROR: no images in {Input_Folder}", log)
            raise FileNotFoundError(f"No valid images found in folder: {Input_Folder}")
        push_node_status(unique_id, f"Found {num_images} image(s).", log)

        image_idx = index % num_images
        lora_list = folder_paths.get_filename_list("loras")
        try:
            start_lora_index = lora_list.index(start_lora_name)
        except ValueError:
            push_node_status(unique_id, f"ERROR: lora '{start_lora_name}' not found.", log)
            raise ValueError(f"Lora '{start_lora_name}' not found in lora list.")

        lora_index = start_lora_index + (index // num_images)
        if lora_index >= len(lora_list):
            push_node_status(unique_id, f"ERROR: lora_index {lora_index} >= lora_list size {len(lora_list)}", log)
            raise ValueError(f"There are no more lora in the list ({len(lora_list)})")
        elif lora_index >= (start_lora_index + lora_count):
            push_node_status(unique_id, f"ERROR: iteration complete after {num_images} rows × {lora_count} loras.", log)
            raise ValueError(f"Index {index} has completed the iteration of rows {num_images} against each lora indicated {lora_count}.")

        image_path = dir_files[image_idx]
        if os.path.isdir(image_path):
            push_node_status(unique_id, f"ERROR: image path is a directory: {image_path}", log)
            raise ValueError(f"Image path is a directory: {image_path}")

        # image_path is already an absolute path (Input_Folder + filename); use
        # it directly — get_annotated_filepath rejects arbitrary absolute paths.
        input_filepath = image_path
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]

        img = node_helpers.pillow(Image.open, input_filepath)

        output_images = []
        output_masks = []
        w, h = None, None

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

        lora_full_path = folder_paths.get_full_path_or_raise("loras", lora_list[lora_index])
        lora_name_only = os.path.basename(lora_list[lora_index])

        previous_input_filename = read_from_file('sozeimagebatchcache.txt')
        write_to_file('sozeimagebatchcache.txt', input_filename)

        headline = (
            f"OK: image {image_idx+1}/{num_images} ({input_filename}), "
            f"lora {lora_index+1}/{len(lora_list)} ({lora_name_only})"
        )
        push_node_status(unique_id, headline, log)
        return (
            output_image,
            output_mask,
            Input_Folder,
            input_filepath,
            input_filename,
            input_filename_no_ext,
            lora_full_path,
            lora_name_only,
            lora_index,
            finalize_status(headline, log),
        )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")



# Code from https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
#Added filename outputs etc
class Soze_BatchProcessSwitch:
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
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "status")
    FUNCTION = "switch"
    CATEGORY = "batch"

    def switch(self, Input, Image=None, Image_Batch=None, Image_Filename_Path_Passthrough="", Image_Batch_Filename_Path_Passthrough="", unique_id=None):
        input_filenamepath = ""
        input_filename = ""
        input_filename_no_ext = ""

        def _resolve(p):
            # Passthrough paths often come from folder loaders as absolute paths.
            # get_annotated_filepath only accepts input-dir-relative annotated
            # names and rejects absolute paths, so prefer the raw path when it's
            # already absolute or exists on disk.
            if os.path.isabs(p) or os.path.exists(p):
                return p
            try:
                return folder_paths.get_annotated_filepath(p)
            except Exception:
                return p

        if Input == "Image":
            if Image_Filename_Path_Passthrough != "":
                input_filenamepath = _resolve(Image_Filename_Path_Passthrough)
                input_filename = os.path.basename(input_filenamepath) if input_filenamepath else ""
                input_filename_no_ext = os.path.splitext(input_filename)[0] if input_filename else ""
            connected = "Image" if Image is not None else "MISSING Image"
            status = f"Branch: Image ({connected}); filename={input_filename or '(none)'}"
            push_node_status(unique_id, status)
            return (Image, input_filenamepath, input_filename, input_filename_no_ext, status)
        else:
            if Image_Batch_Filename_Path_Passthrough != "":
                input_filenamepath = _resolve(Image_Batch_Filename_Path_Passthrough)
                input_filename = os.path.basename(input_filenamepath) if input_filenamepath else ""
                input_filename_no_ext = os.path.splitext(input_filename)[0] if input_filename else ""
            connected = "Image Batch" if Image_Batch is not None else "MISSING Image Batch"
            status = f"Branch: Image Batch ({connected}); filename={input_filename or '(none)'}"
            push_node_status(unique_id, status)
            return (Image_Batch, input_filenamepath, input_filename, input_filename_no_ext, status)
    
#Code From https://github.com/melMass/comfy_mtb
class Soze_LoadImageFromUrl:
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
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "Image_Filename", "Image_Filename_No_Ext", "status")
    FUNCTION = "load"
    CATEGORY = "images"

    def load(self, url, unique_id=None):
        log = EventLog()
        push_node_status(unique_id, f"Fetching: {url}", log)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            headline = f"ERROR: {e!r}"
            push_node_status(unique_id, headline, log)
            raise

        # Extract filename from URL
        filename = os.path.basename(url)
        filename_no_ext = os.path.splitext(filename)[0]

        try:
            wd, h = image.size
        except Exception:
            wd, h = 0, 0
        headline = f"OK: {filename or '(no filename)'} ({wd}x{h})"
        push_node_status(unique_id, headline, log)
        return (
            pil2tensor(image),
            filename,
            filename_no_ext,
            finalize_status(headline, log),
        )


# Node from abandoned repo https://github.com/M1kep/Comfy_KepListStuff 

class Soze_ImageLabelOverlay:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "float_labels": ("FLOAT", {"forceInput": True}),
                "int_labels": ("INT", {"forceInput": True}),
                "str_labels": ("STR", {"forceInput": True}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "process"

    CATEGORY = "image"

    def process(
            self,
            images: List[Tensor],
            float_labels: Optional[List[float]] = None,
            int_labels: Optional[List[int]] = None,
            str_labels: Optional[List[str]] = None,
    ) -> Tuple[List[Tensor]]:
        batches = images

        labels_to_check: Dict[str, Union[List[float], List[int], List[str], None]] = {
            "float": float_labels if float_labels is not None else None,
            "int": int_labels if int_labels is not None else None,
            "str": str_labels if str_labels is not None else None
        }

        for l_type, labels in labels_to_check.items():
            if labels is None:
                continue
            if len(batches) != len(labels) and len(labels) != 1:
                raise Exception(
                    f"Non-matching input sizes got {len(batches)} Image Batches, {len(labels)} Labels for label type {l_type}"
                )

        image_h, _, _ = batches[0][0].size()

        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), 60)

        ret_images: List[Tensor]= []
        loop_gen = zip_with_fill(batches, float_labels, int_labels, str_labels)
        for b_idx, (img_batch, float_lbl, int_lbl, str_lbl) in enumerate(loop_gen):
            batch: List[Tensor] = []
            for i_idx, img in enumerate(img_batch):
                pil_img = tensor2pil(img)
                # print(f"Batch: {b_idx} | img: {i_idx}")
                # print(img.size())
                draw = ImageDraw.Draw(pil_img)

                draw.text((0, image_h - 60), f"B: {b_idx} | I: {i_idx}", fill="red", font=font)

                y_offset = 0
                for _, lbl in zip(["float", "int", "str"], [float_lbl, int_lbl, str_lbl]):
                    if lbl is None:
                        continue
                    draw.rectangle((0, 0 + y_offset, 512, 60 + y_offset), fill="#ffff33")
                    draw.text((0, 0 + y_offset), str(lbl), fill="red", font=font)
                    y_offset += 60
                batch.append(pil2tensor(pil_img))

            ret_images.append(torch.cat(batch))

        return (ret_images,)


# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
ANY = AnyType("*")

# Node from abandoned repo https://github.com/M1kep/Comfy_KepListStuff 

class Soze_XYImage:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "images": ("IMAGE",),
                "splits": ("INT", {"forceInput": True, "min": 1}),
                "flip_axis": (["False", "True"], {"default": "False"}),
                "batch_stack_mode": (["horizontal", "vertical"], {"default": "horizontal"}),
                "z_enabled": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "x_main_label": ("STRING", {}),
                "y_main_label": ("STRING", {}),
                "z_main_label": ("STRING", {}),
                "x_labels": (ANY,{}),
                "y_labels": (ANY,{}),
                "z_labels": (ANY,{}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "xy_image"

    CATEGORY = "image"


    MAIN_LABEL_SIZE = 60
    LABEL_SIZE = 60
    Z_LABEL_SIZE = 60
    LABEL_COLOR = "#000"
    def xy_image(
            self,
            images: List[Tensor],
            splits: List[int],
            flip_axis: List[str],
            batch_stack_mode: List[str],
            z_enabled: List[str],
            x_main_label: Optional[List[str]] = None,
            y_main_label: Optional[List[str]] = None,
            z_main_label: Optional[List[str]] = None,
            x_labels: Optional[List[str]] = None,
            y_labels: Optional[List[str]] = None,
            z_labels: Optional[List[str]] = None,
    ) -> Tuple[List[Tensor]]:
        if len(flip_axis) != 1:
            raise Exception("Only single flip_axis value supported.")
        if len(batch_stack_mode) != 1:
            raise Exception("Only single batch stack mode supported.")
        if len(z_enabled) != 1:
            raise Exception("Only single z_enabled value supported.")
        if x_main_label is not None and len(x_main_label) != 1:
            raise Exception("Only single x_main_label value supported.")
        if y_main_label is not None and len(y_main_label) != 1:
            raise Exception("Only single y_main_label value supported.")
        if z_main_label is not None and len(z_main_label) != 1:
            raise Exception("Only single z_main_label value supported.")

        if x_main_label is not None and not isinstance(x_main_label[0], str):
            try:
                x_main_label[0] = str(x_main_label[0])
            except:
                raise Exception("x_main_label must be a string or convertible to a string.")
        if y_main_label is not None and not isinstance(y_main_label[0], str):
            try:
                y_main_label[0] = str(y_main_label[0])
            except:
                raise Exception("y_main_label must be a string or convertible to a string.")
        if z_main_label is not None and not isinstance(z_main_label[0], str):
            try:
                z_main_label[0] = str(z_main_label[0])
            except:
                raise Exception("z_main_label must be a string or convertible to a string.")

        if x_main_label is not None and x_main_label[0] == '':
            x_main_label = None
        if y_main_label is not None and y_main_label[0] == '':
            y_main_label = None
        if z_main_label is not None and z_main_label[0] == '':
            z_main_label = None

        stack_direction = "horizontal"
        if flip_axis[0] == "True":
            stack_direction = "vertical"
            x_labels, y_labels = y_labels, x_labels
            x_main_label, y_main_label = y_main_label, x_main_label

        batch_stack_direction = batch_stack_mode[0]

        if len(splits) == 1:
            splits = splits * (int(len(images) / splits[0]))
            if sum(splits) != len(images):
                splits.append(len(images) - sum(splits))
        else:
            if sum(splits) != len(images):
                raise Exception("Sum of splits must equal number of images.")

        batches = images
        batch_size = len(batches[0])

        # TODO: Some better way...
        # Currently chops splits to match x_labels/y_labels and then loops over the split set over and over
        num_z = 1
        splits_per_z = len(splits)
        images_per_z = len(images)
        if z_enabled[0] == "True":
            if y_labels is None or x_labels is None:
                raise Exception("Must provide x_labels and y_labels when z_enabled is True.")

            if stack_direction == "horizontal":
                splits_per_z = len(x_labels)
            else:
                splits_per_z = len(y_labels)

            num_z = int(len(splits) / splits_per_z)
            splits = splits[:splits_per_z]
            images_per_z = sum(splits)

        image_h, image_w, _ = batches[0][0].size()
        if batch_stack_direction == "horizontal":
            batch_h = image_h
            # stack horizontally
            batch_w = image_w * batch_size
        else:
            # stack vertically
            batch_h = image_h * batch_size
            batch_w = image_w

        if stack_direction == "horizontal":
            full_w = batch_w * len(splits)
            full_h = batch_h * max(splits)
        else:
            full_w = batch_w * max(splits)
            full_h = batch_h * len(splits)
        grid_w = full_w
        _ = full_h

        y_label_offset = 0
        has_horizontal_labels = False
        if x_labels is not None:
            x_labels = [str(lbl) for lbl in x_labels]
            if stack_direction == "horizontal":
                if len(x_labels) != len(splits):
                    raise Exception("Number of horizontal labels must match number of splits.")
            else:
                if len(x_labels) != max(splits):
                    raise Exception("Number of horizontal labels must match maximum split size.")
            full_h += self.LABEL_SIZE
            y_label_offset = self.LABEL_SIZE
            has_horizontal_labels = True

        x_label_offset = 0
        has_vertical_labels = False
        if y_labels is not None:
            y_labels = [str(lbl) for lbl in y_labels]
            if stack_direction == "horizontal":
                if len(y_labels) != max(splits):
                    raise Exception(f"Number of vertical labels must match maximum split size. Got {len(y_labels)} labels for {max(splits)} splits.")
            else:
                if len(y_labels) != len(splits):
                    raise Exception(f"Number of vertical labels must match number of splits. Got {len(y_labels)} labels for {len(splits)} splits.")
            full_w += self.LABEL_SIZE
            x_label_offset = self.LABEL_SIZE
            has_vertical_labels = True

        has_z_labels = False
        if z_labels is not None:
            has_z_labels = True
            z_labels = [str(lbl) for lbl in z_labels]
            if z_main_label is not None:
                z_labels = [f"{z_main_label[0]}: {lbl}" for lbl in z_labels]
            full_h += self.Z_LABEL_SIZE
            y_label_offset += self.Z_LABEL_SIZE
            if len(z_labels) != num_z:
                raise Exception(f"Number of z_labels must match number of z splits. Got {len(z_labels)} labels for {num_z} splits.")

        has_main_x_label = False
        if x_main_label is not None:
            full_h += self.MAIN_LABEL_SIZE
            y_label_offset += self.MAIN_LABEL_SIZE
            has_main_x_label = True

        has_main_y_label = False
        if y_main_label is not None:
            full_w += self.MAIN_LABEL_SIZE
            x_label_offset += self.MAIN_LABEL_SIZE
            has_main_y_label = True

        images = []
        for z_idx in range(num_z):
            full_image = Image.new("RGB", (full_w, full_h))
            full_draw = ImageDraw.Draw(full_image)

            full_draw.rectangle((0, 0, full_w, full_h), fill="#ffffff")

            batch_idx = 0
            active_y_offset = 0
            active_x_offset = 0
            if has_z_labels:
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.Z_LABEL_SIZE)
                full_draw.rectangle((0, 0, full_w, self.Z_LABEL_SIZE), fill="#ffffff")
                full_draw.text((grid_w//2 + x_label_offset, 0),  z_labels[z_idx], anchor='ma', fill=self.LABEL_COLOR, font=font)
                active_y_offset += self.Z_LABEL_SIZE

            if has_main_x_label:
                assert x_main_label is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.MAIN_LABEL_SIZE)
                full_draw.rectangle((0, active_y_offset, full_w, self.MAIN_LABEL_SIZE + active_y_offset), fill="#ffffff")
                full_draw.text((grid_w//2 + x_label_offset, 0 + active_y_offset), x_main_label[0], anchor='ma', fill=self.LABEL_COLOR, font=font)
                active_y_offset += self.MAIN_LABEL_SIZE

            if has_horizontal_labels:
                assert x_labels is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.LABEL_SIZE)
                for label_idx, label in enumerate(x_labels):
                    x_offset = (batch_w * label_idx) + x_label_offset
                    full_draw.rectangle((x_offset, 0 + active_y_offset, x_offset + batch_w, self.LABEL_SIZE + active_y_offset), fill="#ffffff")
                    full_draw.text((x_offset + (batch_w / 2), 0 + active_y_offset), label, anchor='ma', fill=self.LABEL_COLOR, font=font)

            if has_main_y_label:
                assert y_main_label is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.MAIN_LABEL_SIZE)

                img_txt = Image.new('RGB', (full_h - active_y_offset, self.MAIN_LABEL_SIZE))
                draw_txt = ImageDraw.Draw(img_txt)
                draw_txt.rectangle((0, 0, full_h - active_y_offset, self.MAIN_LABEL_SIZE), fill="#ffffff")
                draw_txt.text(((full_h - active_y_offset)//2, 0),  y_main_label[0], anchor='ma', fill=self.LABEL_COLOR, font=font)
                img_txt = img_txt.rotate(90, expand=True)
                full_image.paste(img_txt, (active_x_offset, active_y_offset))
                active_x_offset += self.MAIN_LABEL_SIZE

            if has_vertical_labels:
                assert y_labels is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.LABEL_SIZE)
                for label_idx, label in enumerate(y_labels):
                    y_offset = (batch_h * label_idx) + y_label_offset

                    img_txt = Image.new('RGB', (batch_h, self.LABEL_SIZE))
                    draw_txt = ImageDraw.Draw(img_txt)
                    draw_txt.rectangle((0, 0, batch_h, self.LABEL_SIZE), fill="#ffffff")
                    draw_txt.text((batch_h//2, 0),  label, anchor='ma', fill=self.LABEL_COLOR, font=font)
                    img_txt = img_txt.rotate(90, expand=True)
                    full_image.paste(img_txt, (active_x_offset, y_offset))

            for split_idx, split in enumerate(splits):
                for idx_in_split in range(split):
                    batch_img = Image.new("RGB", (batch_w, batch_h))
                    batch = batches[batch_idx + idx_in_split + images_per_z * z_idx]
                    if batch_stack_direction == "horizontal":
                        for img_idx, img in enumerate(batch):
                            x_offset = image_w * img_idx
                            batch_img.paste(tensor2pil(img), (x_offset, 0))
                    else:
                        for img_idx, img in enumerate(batch):
                            y_offset = image_h * img_idx
                            batch_img.paste(tensor2pil(img), (0, y_offset))

                    if stack_direction == "horizontal":
                        x_offset = batch_w * split_idx + x_label_offset
                        y_offset = batch_h * idx_in_split + y_label_offset
                    else:
                        x_offset = batch_w * idx_in_split + x_label_offset
                        y_offset = batch_h * split_idx + y_label_offset
                    full_image.paste(batch_img, (x_offset, y_offset))

                batch_idx += split
            images.append(pil2tensor(full_image))
        return (images,)

# Node from abandoned repo https://github.com/M1kep/Comfy_KepListStuff 

class Soze_VariableImageBuilder:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "r": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "g": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "b": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "a": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "width": ("INT", {"defaultInput": False, "default": 512}),
                "height": ("INT", {"defaultInput": False, "default": 512}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"

    CATEGORY = "image"

    def generate_images(
            self,
            r: int,
            g: int,
            b: int,
            a: int,
            width: int,
            height: int,
            batch_size: int,
    ) -> Tuple[Tensor]:
        batch_tensors: List[Tensor] = []
        for _ in range(batch_size):
            image = Image.new("RGB", (width, height), color=(r, g, b, a))
            batch_tensors.append(pil2tensor(image))
        return (torch.cat(batch_tensors),)

# Node from abandoned repo https://github.com/M1kep/Comfy_KepListStuff 

class Soze_EmptyImages:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {},
            "optional": {
                "num_images": ("INT", {"forceInput": True, "min": 1}),
                "splits": ("INT", {"forceInput": True, "min": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "generate_empty_images"

    CATEGORY = "image"

    def generate_empty_images(
            self,
            num_images: Optional[List[int]] = None,
            splits: Optional[List[int]] = None,
            batch_size: Optional[List[int]] = None,
    ) -> Tuple[List[Tensor]]:
        if batch_size is None:
            batch_size = [1]
        else:
            if len(batch_size) != 1:
                raise Exception("Only single batch size supported.")

        if num_images is None and splits is None:
            raise Exception("Must provide either num_images or splits.")

        if num_images is not None and len(num_images) != 1:
            raise Exception("Only single num_images supported.")

        if num_images is not None and splits is None:
            # If splits is None, then all images are in one split
            splits = [num_images[0]]

        if num_images is None and splits is not None:
            # If num_images is None, then it should be the sum of all splits
            num_images = [sum(splits)]

        if num_images is not None and splits is not None:
            if len(splits) == 1:
                # Fill splits with same value enough times to sum to num_images
                fills = int(num_images[0] / splits[0])
                splits = [splits[0]] * fills
                if sum(splits) != num_images[0]:
                    splits.append(num_images[0] - sum(splits))
            else:
                if sum(splits) != num_images[0]:
                    raise Exception("Sum of splits must match number of images.")

        if splits is None:
            raise ValueError("Unexpected error: Splits is None")

        ret_images: List[Tensor] = []
        for split_idx, split in enumerate(splits):
            # Rotate between fully dynamic range of colors
            base_color = (
                50 + (split_idx * 45) % 200,  # Cycle between 50 and 250
                30 + (split_idx * 75) % 200,
                10 + (split_idx * 105) % 200,
            )

            for _ in range(split):
                batch_tensor = torch.zeros(batch_size[0], 512, 512, 3)
                for batch_idx in range(batch_size[0]):
                    batch_color = (
                        (base_color[0] + int(((255 - base_color[0]) / batch_size[0]) * batch_idx)),
                        (base_color[1] + int(((255 - base_color[1]) / batch_size[0]) * batch_idx)),
                        (base_color[2] + int(((255 - base_color[2]) / batch_size[0]) * batch_idx)),
                    )
                    image = Image.new("RGB", (512, 512), color=batch_color)
                    batch_tensor[batch_idx] = pil2tensor(image)
                ret_images.append(batch_tensor)
        return (ret_images,)

# Node from abandoned repo https://github.com/M1kep/Comfy_KepListStuff 

class Soze_ImageListLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "folder_path": ("STRING", {}),
                "file_filter": ("STRING", {"default": "*.png"}),
                "sort_method": (["numerical", "alphabetical"], {"default": "numerical"}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_images"

    CATEGORY = "image"

    @staticmethod
    def numerical_sort(file_name: Path) -> int:
        subbed = re.sub(r"\D", "", str(file_name))
        if subbed == "":
            return 0
        return int(subbed)
    
    
    @staticmethod
    def alphabetical_sort(file_name: Path) -> str:
        return str(file_name)

    def load_images(
        self, folder_path: str, file_filter: str, sort_method: str
    ) -> Tuple[List[Tensor]]:
        folder = Path(folder_path)
    
        if not folder.is_dir():
            raise Exception(f"Folder path {folder_path} does not exist.")

        sort_method_impl: Callable[[str], Union[SupportsDunderGT, SupportsDunderLT]]
        if sort_method == "numerical":
            sort_method_impl = self.numerical_sort
        elif sort_method == "alphabetical":
            sort_method_impl = self.alphabetical_sort
        else:
            raise ValueError(f"Unknown sort method {sort_method}")

        files = sorted(folder.glob(file_filter), key=sort_method_impl)
        images = [pil2tensor(Image.open(file)) for file in files]
    
        return (images,)




class Soze_GetImageColors:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "num_colors": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Number of colors to detect",
                    },
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "#000000,#FFFFFF",
                        "tooltip": "Comma-separated list of colors to exclude from the output",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "rgb_colors",
        "hex_colors",
        "status",
    )
    FUNCTION = "main"
    CATEGORY = "image"

    def main(
        self,
        input_image: torch.Tensor,
        num_colors: int = 5,
        exclude_colors: str = "",
        unique_id=None,
    ) -> tuple[str, ...]:
        # Process exclude colors
        if exclude_colors.strip():
            self.exclude = [color.strip().lower() for color in exclude_colors.strip().split(",")]
        else:
            self.exclude = []

        # Convert image to pixels
        pixels = input_image.view(-1, input_image.shape[-1]).numpy()
        pixels = (pixels * 255).astype(int)  # Scale to 0-255 and convert to integers

        # Create color strings and count them
        color_counts = {}
        for pixel in pixels:
            if pixel.shape[0] == 3:  # RGB image
                r, g, b = pixel
            else:  # RGBA image
                r, g, b, _ = pixel  # Ignore alpha channel
            rgb_str = f"rgb({r}, {g}, {b})"
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            
            # Skip if this color should be excluded
            if hex_str.lower() in self.exclude or rgb_str.lower() in self.exclude:
                continue
                
            if (rgb_str, hex_str) in color_counts:
                color_counts[(rgb_str, hex_str)] += 1
            else:
                color_counts[(rgb_str, hex_str)] = 1

        # Sort by frequency and take top num_colors
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        top_colors = sorted_colors[:num_colors]

        # Separate RGB and hex colors
        rgb_colors = []
        hex_colors = []
        for (rgb, hex_color), _ in top_colors:
            rgb_colors.append(rgb)
            hex_colors.append(hex_color)

        status = f"OK: top {len(hex_colors)} of {num_colors} requested — {', '.join(hex_colors[:8])}"
        push_node_status(unique_id, status)
        return (
            ", ".join(rgb_colors),
            ", ".join(hex_colors),
            status,
        )
    

#Code from https://github.com/dzqdzq/ComfyUI-crop-alpha
# class Soze_AlphaCropAndPositionImage:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "maintain_aspect": (["True", "False"], {"default": "True"}),
#                 "left_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
#                 "top_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
#                 "right_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
#                 "bottom_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE", "INT", "INT")
#     RETURN_NAMES = ("image", "width", "height")

#     FUNCTION = "crop"
#     CATEGORY = "image/processing"

#     def crop(self, image, maintain_aspect, left_padding: int = 0, right_padding: int = 0, top_padding: int = 0, bottom_padding: int = 0):
#         cropped_images = []
#         cropped_masks = []

#         for img in image:
#             alpha = img[..., 3]

#             height = img.shape[0]
#             width = img.shape[1]
#             mask = (alpha > 0.01)

#             rows = torch.any(mask, dim=1)
#             cols = torch.any(mask, dim=0)

#             ymin, ymax = self._find_boundary(rows)
#             xmin, xmax = self._find_boundary(cols)

#             if ymin is None or xmin is None:
#                 cropped_images.append(img)
#                 cropped_masks.append(torch.zeros_like(alpha))
#                 continue

#             cropped = img[ymin:ymax, xmin:xmax, :4]
#             cropped_mask = alpha[ymin:ymax, xmin:xmax]

#             # Apply padding to the cropped image
#             padded_height = (ymax - ymin) + top_padding + bottom_padding
#             padded_width = (xmax - xmin) + left_padding + right_padding

#             if maintain_aspect == "True":
#                 if padded_height > padded_width:
#                     pad = (padded_height - padded_width) // 2
#                     left_padding += pad
#                     right_padding += pad
#                     padded_width = padded_height
#                 else:
#                     pad = (padded_width - padded_height) // 2
#                     top_padding += pad
#                     bottom_padding += pad
#                     padded_height = padded_width

#             padded_image = torch.zeros((padded_height, padded_width, 4), dtype=img.dtype)
#             padded_image[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin), :] = cropped

#             padded_mask = torch.zeros((padded_height, padded_width), dtype=alpha.dtype)
#             padded_mask[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin)] = cropped_mask

#             cropped_images.append(padded_image)
#             cropped_masks.append(padded_mask)

#         return cropped_images, padded_width, padded_height
    
#     def _find_boundary(self, arr):
#         nz = torch.nonzero(arr)
#         if nz.numel() == 0:
#             return (None, None)
#         return (nz[0].item(), nz[-1].item() + 1)


class Soze_ShrinkImage:
    __doc__ = """
    ShrinkImage(
        image: IMAGE,
        mode: ["scale", "pixels"] = "scale",
        resize_algorithm: ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"] = "LANCZOS",
        maintain_aspect: ["True", "False"] = "True",
        scale: FLOAT = 0.5,
        width: FLOAT = 100,
        height: FLOAT = 100
    ) -> IMAGE

    Shrinks the input image to the specified scale or pixel dimensions using the selected resize algorithm.
    Optionally maintains the aspect ratio of the image.

    Parameters:
    - image: The input image to be shrunk.
    - mode: The mode of shrinking, either by scale (relative to original size) or by absolute pixel dimensions.
    - resize_algorithm: The algorithm to use for resizing the image.
    - maintain_aspect: Whether to maintain the aspect ratio of the image when resizing.
    - scale: The scale factor to shrink the image. Ignored if mode is set to "pixels".
    - width: The target width in pixels if mode is set to "pixels".
    - height: The target height in pixels if mode is set to "pixels".

    Returns:
    - The shrunk image.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["scale", "pixels"], {"default": "scale"}),
                "resize_algorithm": (list(resize_algorithms.keys()), {"default": "LANCZOS"}),
                "maintain_aspect": (["True", "False"], {"default": "True"})
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1}),
                "height": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1})
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "status")
    FUNCTION = "shrink_image"
    CATEGORY = "image/processing"

    def calculate_scale(self, img, mode, maintain_aspect, scale=None, width=None, height=None):
        if mode == "scale":
            return scale
        else:
            img_width, img_height = img.size
            if maintain_aspect == "True":
                aspect_ratio = img_width / img_height
                if width / height > aspect_ratio:
                    width = height * aspect_ratio
                else:
                    height = width / aspect_ratio
            scale_x = width / img_width
            scale_y = height / img_height
            return min(scale_x, scale_y)

    def shrink_image_with_scale(self, img, scale, algorithm):
        width, height = img.size
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        return img.resize((new_width, new_height), algorithm)

    def shrink_image(self, image, mode, resize_algorithm, maintain_aspect, scale=None, width=None, height=None, unique_id=None):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        algorithm = resize_algorithms[resize_algorithm]

        output_images = []
        first_in = None
        last_out = None
        for img in image:
            img = to_pil_image(img.permute(2, 0, 1))
            if first_in is None:
                first_in = img.size
            scale_used = self.calculate_scale(img, mode, maintain_aspect, scale, width, height)
            resized_img = self.shrink_image_with_scale(img, scale_used, algorithm)
            last_out = resized_img.size
            resized_img_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_img_np = torch.from_numpy(resized_img_np)
            output_images.append(resized_img_np)

        if first_in and last_out:
            status = f"OK: {len(output_images)} image(s) — {first_in[0]}x{first_in[1]} → {last_out[0]}x{last_out[1]} ({resize_algorithm})"
        else:
            status = f"OK: {len(output_images)} image(s) ({resize_algorithm})"
        push_node_status(unique_id, status)
        return (output_images, status)
    

class Soze_PadMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "top_padding": ("INT", {"default": 0, "min": 0}),
                "bottom_padding": ("INT", {"default": 0, "min": 0}),
                "left_padding": ("INT", {"default": 0, "min": 0}),
                "right_padding": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "pad_mask"
    CATEGORY = "soze"

    def pad_mask(self, mask, top_padding, bottom_padding, left_padding, right_padding):
        # Get the original mask dimensions
        original_height, original_width = mask.shape[-2:]

        # Ensure the new dimensions do not exceed the original mask area
        max_top_bottom_padding = original_height // 2
        max_left_right_padding = original_width // 2

        top_padding = min(top_padding, max_top_bottom_padding)
        bottom_padding = min(bottom_padding, max_top_bottom_padding)
        left_padding = min(left_padding, max_left_right_padding)
        right_padding = min(right_padding, max_left_right_padding)

        # Calculate the new dimensions
        new_height = original_height + top_padding + bottom_padding
        new_width = original_width + left_padding + right_padding

        # Ensure the new dimensions are valid
        if new_height <= 0 or new_width <= 0:
            raise ValueError("Invalid padding values resulting in non-positive dimensions.")

        # Create a new mask filled with zeros (same dtype and device as the original mask)
        padded_mask = torch.zeros((new_height, new_width), dtype=mask.dtype, device=mask.device)

        # Calculate the placement indices for the original mask
        start_y = top_padding
        end_y = start_y + original_height
        start_x = left_padding
        end_x = start_x + original_width

        # Ensure indices are within bounds
        if end_y > new_height or end_x > new_width:
            raise ValueError("Padding values result in out-of-bounds placement of the original mask.")

        # Place the original mask in the center of the new mask
        padded_mask[start_y:end_y, start_x:end_x] = mask

        return padded_mask



class Soze_MultiImageBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            },
            "optional": {
                "images1": ("IMAGE",),
                "images2": ("IMAGE",),
                "images3": ("IMAGE",),
                "images4": ("IMAGE",),
                "images5": ("IMAGE",),
                "images6": ("IMAGE",),
                # Theoretically, an infinite number of image input parameters can be added.
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "multiImageBatch"
    CATEGORY = "soze"

    def _check_img_dims(self, tensors, names):
        reference_dimensions = tensors[0].shape[1:]  # Ignore batch dimension
        mismatched_images = [(names[i], tensors[i].shape[1:]) for i, tensor in enumerate(tensors) if tensor.shape[1:] != reference_dimensions]

        if mismatched_images:
            raise ValueError(f"Multi Image Batch Warning: Input image dimensions do not match. Reference dimensions: {reference_dimensions}. Mismatched images: {mismatched_images}")

    def multiImageBatch(self, **kwargs):
        batched_tensors = [kwargs[key] for key in kwargs if kwargs[key] is not None]
        image_names = [key for key in kwargs if kwargs[key] is not None]

        if not batched_tensors:
            # Return an empty tensor if no valid images are provided
            return (torch.empty(0, 3, 0, 0),)

        # Normalize channels to 3 (RGB) by dropping alpha if present
        for i in range(len(batched_tensors)):
            if batched_tensors[i].shape[-1] == 4:
                batched_tensors[i] = batched_tensors[i][..., :3]

        self._check_img_dims(batched_tensors, image_names)
        batched_tensors = torch.cat(batched_tensors, dim=0)
        return (batched_tensors,)



class Soze_ImageSizeWithMaximum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "max_long_edge": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 64}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "status")

    FUNCTION = "check_size"
    CATEGORY = "soze"

    def check_size(self, image, max_long_edge, unique_id=None):
        if image is None:
            push_node_status(unique_id, "Skipped: image is None")
            return (0, 0, "Skipped: image is None")

        height, width = image.shape[1], image.shape[2]
        is_within_size = (height <= max_long_edge) and (width <= max_long_edge)

        target_width = width
        target_height = height

        if not is_within_size:
            aspect_ratio = width / height
            if width > height:
                target_width = max_long_edge
                target_height = int(max_long_edge / aspect_ratio)
            else:
                target_height = max_long_edge
                target_width = int(max_long_edge * aspect_ratio)

        if is_within_size:
            status = f"OK (within max): {width}x{height}"
        else:
            status = f"OK (downscaled to max {max_long_edge}): {width}x{height} → {target_width}x{target_height}"
        push_node_status(unique_id, status)
        return (target_width, target_height, status)
    
    
class Soze_SaveImageWithAbsoluteFilename:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, unique_id=None):
        if images is None:
            push_node_status(unique_id, "Skipped: no images provided.")
            return ()
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        last_path = None
        total_bytes = 0
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            if len(images) == 1:
                 file = f"{filename_with_batch_num}.png"
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
            out_path = os.path.join(full_output_folder, file)
            img.save(out_path, pnginfo=metadata, compress_level=self.compress_level)
            try:
                total_bytes += os.path.getsize(out_path)
            except OSError:
                pass
            last_path = out_path
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        status = f"OK: saved {len(images)} image(s), {total_bytes:,} bytes total. Last: {last_path}"
        push_node_status(unique_id, status)
        return {"ui": {"images": results, "text": [status]}}


class Soze_LoadRandomImagesFromFolder:
    """Pick N random images from a folder and return them as a single batch.

    Different image dimensions are normalized by resizing each subsequent image
    to match the first one (matches ComfyUI's standard batch-loader convention).
    """

    VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "tooltip": "Absolute or relative folder path."}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1, "tooltip": "How many images to randomly select."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "0 = nondeterministic; any other value seeds the RNG for reproducible picks."}),
                "allow_repeats": ("BOOLEAN", {"default": False, "tooltip": "If True, the same image may be picked more than once when image_count exceeds the folder size."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING")
    RETURN_NAMES = ("Image_Batch", "Mask_Batch", "Loaded_Count", "Filenames", "status")
    FUNCTION = "load_random_images"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, input_folder, image_count, seed=0, allow_repeats=False, unique_id=None):
        # If seed > 0, IS_CHANGED is purely a function of inputs → cacheable.
        # If seed == 0, force re-execution every prompt.
        if seed and seed != 0:
            return f"{input_folder}|{image_count}|{seed}|{allow_repeats}"
        return float("NaN")

    def _load_one(self, image_path):
        """Open one image into a (image_tensor, mask_tensor) pair, both batched as [1, H, W, *]."""
        img = node_helpers.pillow(Image.open, image_path)
        img = node_helpers.pillow(ImageOps.exif_transpose, img)
        if img.mode == 'I':
            img = img.point(lambda x: x * (1 / 255))
        rgb = img.convert("RGB")
        arr = np.array(rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr)[None,]
        if 'A' in img.getbands():
            mask_arr = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = (1.0 - torch.from_numpy(mask_arr)).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
        return image_tensor, mask_tensor

    def load_random_images(self, input_folder, image_count, seed=0, allow_repeats=False, unique_id=None):
        log = EventLog()
        push_node_status(unique_id, f"Scanning {input_folder}", log)

        if not input_folder or not os.path.isdir(input_folder):
            headline = f"ERROR: folder not found: {input_folder!r}"
            push_node_status(unique_id, headline, log)
            raise FileNotFoundError(f"Folder not found: {input_folder}")

        try:
            entries = os.listdir(input_folder)
        except OSError as e:
            headline = f"ERROR listing folder: {e}"
            push_node_status(unique_id, headline, log)
            raise

        candidates = sorted(
            os.path.join(input_folder, f)
            for f in entries
            if f.lower().endswith(self.VALID_EXTENSIONS) and os.path.isfile(os.path.join(input_folder, f))
        )
        push_node_status(unique_id, f"Found {len(candidates)} image(s) matching {self.VALID_EXTENSIONS}", log)

        if not candidates:
            headline = f"ERROR: no images in {input_folder}"
            push_node_status(unique_id, headline, log)
            raise FileNotFoundError(f"No image files found in folder: {input_folder}")

        rng = random.Random(seed) if seed and seed != 0 else random.Random()

        if allow_repeats:
            picks = [rng.choice(candidates) for _ in range(image_count)]
        else:
            n = min(image_count, len(candidates))
            if image_count > len(candidates):
                push_node_status(unique_id, f"Note: requested {image_count} but only {len(candidates)} unique images; returning {n}.", log)
            picks = rng.sample(candidates, n)

        push_node_status(unique_id, f"Picked {len(picks)} image(s) (seed={seed if seed else 'random'}, repeats={'on' if allow_repeats else 'off'})", log)

        images = []
        masks = []
        loaded_filenames = []
        for path in picks:
            try:
                img_tensor, mask_tensor = self._load_one(path)
            except Exception as e:
                logger.error("Failed to load %s: %s", path, e)
                push_node_status(unique_id, f"Skipping {os.path.basename(path)}: {e!r}", log)
                continue
            images.append(img_tensor)
            masks.append(mask_tensor)
            loaded_filenames.append(os.path.basename(path))

        if not images:
            headline = "ERROR: every selected image failed to load."
            push_node_status(unique_id, headline, log)
            raise RuntimeError(headline)

        # Normalize sizes so torch.cat doesn't blow up.
        base = images[0]
        base_h, base_w = base.shape[1], base.shape[2]
        aligned_imgs = [base]
        aligned_masks = [masks[0]]
        resized = 0
        for img, msk in zip(images[1:], masks[1:]):
            if img.shape[1] != base_h or img.shape[2] != base_w:
                img = common_upscale(img.movedim(-1, 1), base_w, base_h, "bilinear", "center").movedim(1, -1)
                msk = torch.nn.functional.interpolate(
                    msk.unsqueeze(0), size=(base_h, base_w), mode="bilinear", align_corners=False,
                ).squeeze(0)
                resized += 1
            aligned_imgs.append(img)
            aligned_masks.append(msk)

        image_batch = torch.cat(aligned_imgs, dim=0)
        mask_batch = torch.cat(aligned_masks, dim=0)

        if resized:
            push_node_status(unique_id, f"Resized {resized} of {len(aligned_imgs)} image(s) to {base_w}x{base_h} for batching.", log)

        headline = f"OK: loaded {len(aligned_imgs)} image(s) at {base_w}x{base_h}"
        push_node_status(unique_id, headline, log)
        return (
            image_batch,
            mask_batch,
            len(aligned_imgs),
            "\n".join(loaded_filenames),
            finalize_status(headline, log),
        )


# ---------------------------------------------------------------------------
# Scribble XDoG — eXtended Difference-of-Gaussians sketch / scribble filter
# ---------------------------------------------------------------------------


def _xdog_gaussian_blur(gray_2d, sigma):
    """Gaussian-blur a [H,W] float32 tensor on CPU using torchvision.

    torchvision's gaussian_blur wants (..., C, H, W) and an odd kernel size; we
    derive kernel_size from sigma the same way scipy/opencv do (2*ceil(3σ)+1).
    """
    from torchvision.transforms.functional import gaussian_blur
    if sigma <= 0:
        return gray_2d
    k = max(3, 2 * int(3 * sigma + 0.5) + 1)
    t = gray_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    blurred = gaussian_blur(t, kernel_size=[k, k], sigma=[float(sigma), float(sigma)])
    return blurred.squeeze(0).squeeze(0)


def _xdog_one_frame(rgb_hwc, sigma, k, sharpness_p, epsilon, phi, gamma,
                    invert, binarize, binarize_threshold):
    """Run XDoG on a single [H, W, 3] float32 frame in [0, 1].

    Implements Winnemöller's eXtended DoG:
        D(x)  = G_σ(x) - τ · G_(kσ)(x)            where τ = p/(1+p) ... we use
                                                   the equivalent (1+p)·G_σ - p·G_(kσ)
        T(D) = 1                       if D >= ε
               1 + tanh(φ · (D - ε))   otherwise
    """
    # 1) Grayscale (BT.601 weights) — XDoG is defined on luminance.
    r, g, b = rgb_hwc[..., 0], rgb_hwc[..., 1], rgb_hwc[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 2) Two Gaussian blurs.
    g_small = _xdog_gaussian_blur(gray, sigma)
    g_large = _xdog_gaussian_blur(gray, sigma * k)

    # 3) eXtended difference of Gaussians.
    dog = (1.0 + sharpness_p) * g_small - sharpness_p * g_large

    # 4) Soft threshold: 1 above epsilon, 1 + tanh(...) below.
    above = dog >= epsilon
    soft = 1.0 + torch.tanh(phi * (dog - epsilon))
    out = torch.where(above, torch.ones_like(dog), soft)
    out = out.clamp(0.0, 1.0)

    # 5) Optional gamma adjustment.
    if gamma and abs(gamma - 1.0) > 1e-6:
        # avoid 0^x edge cases
        out = out.clamp(min=1e-6) ** (1.0 / float(gamma))
        out = out.clamp(0.0, 1.0)

    # 6) Optional hard binarization for crisp scribble.
    if binarize:
        out = (out >= float(binarize_threshold)).to(out.dtype)

    # 7) Optional invert (XDoG's natural orientation is black lines on white;
    # most ControlNet scribble preprocessors expect white lines on black).
    if invert:
        out = 1.0 - out

    return out  # [H, W] float32 in [0, 1]


class Soze_ScribbleXDoG:
    """Convert an image (or batch) into a black-and-white scribble / sketch
    using the eXtended Difference-of-Gaussians (XDoG) algorithm.

    Good defaults for general line-art: sigma=0.5, k=1.6, sharpness=10,
    epsilon=0.0, phi=10. Tweak `sharpness` and `phi` for edge contrast,
    `sigma` for line scale, and toggle `binarize` for crisp B&W output
    suited to ControlNet scribble preprocessors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.05, "tooltip": "Base Gaussian sigma. Higher = thicker/coarser lines."}),
                "k": ("FLOAT", {"default": 1.6, "min": 1.05, "max": 5.0, "step": 0.05, "tooltip": "Sigma ratio between the two Gaussians. 1.6 is the classic DoG sweet spot."}),
                "sharpness": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 200.0, "step": 0.5, "tooltip": "XDoG 'p' parameter — emphasizes edges. Higher = bolder lines."}),
                "epsilon": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "tooltip": "Threshold below which the soft-step kicks in. Negative widens lines."}),
                "phi": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 500.0, "step": 0.5, "tooltip": "Soft-threshold steepness. Higher = harder edges (more binary feel)."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Output gamma. <1 darkens, >1 brightens."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "On: white lines on black background (ControlNet scribble convention). Off: black lines on white."}),
                "binarize": ("BOOLEAN", {"default": False, "tooltip": "Hard threshold for crisp B&W. Use with binarize_threshold."}),
                "binarize_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Only used when binarize=True."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "status")
    FUNCTION = "process"
    CATEGORY = "image/preprocessors"

    def process(self, image, sigma, k, sharpness, epsilon, phi, gamma,
                invert, binarize, binarize_threshold, unique_id=None):
        if image is None or image.shape[0] == 0:
            status = "Skipped: image is None or empty."
            push_node_status(unique_id, status)
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (blank, blank_mask, status)

        # Run on CPU — XDoG kernels are tiny and CPU is plenty fast for typical
        # preview/preprocessor sizes; avoids hitting GPU when the user just
        # wants a quick scribble pass.
        src = image.detach().cpu().to(torch.float32)
        if src.shape[-1] == 4:
            # Drop alpha — XDoG operates on RGB luminance.
            src = src[..., :3]

        out_frames = []
        for i in range(src.shape[0]):
            scribble_2d = _xdog_one_frame(
                src[i], sigma, k, sharpness, epsilon, phi, gamma,
                invert, binarize, binarize_threshold,
            )
            # Re-replicate the single channel to RGB so it's a normal IMAGE tensor.
            out_frames.append(scribble_2d.unsqueeze(-1).expand(-1, -1, 3))

        image_out = torch.stack(out_frames, dim=0)  # [B, H, W, 3]
        # MASK is [B, H, W] single-channel (use the same scribble — caller can
        # invert downstream if they need the opposite polarity).
        mask_out = image_out[..., 0].clone()

        mode_bits = []
        if binarize:
            mode_bits.append(f"binarized@{binarize_threshold}")
        if invert:
            mode_bits.append("inverted")
        mode_tag = " | ".join(mode_bits) or "soft"
        status = (
            f"OK: {src.shape[0]} frame(s) {src.shape[2]}x{src.shape[1]}; "
            f"sigma={sigma}, k={k}, sharpness={sharpness}, epsilon={epsilon}, phi={phi}, gamma={gamma}; {mode_tag}"
        )
        push_node_status(unique_id, status)
        return (image_out, mask_out, status)


# ---------------------------------------------------------------------------
# Lineart — Sobel-gradient lineart with morphology + thresholding controls
# ---------------------------------------------------------------------------


def _sobel_magnitude(gray_2d):
    """3x3 Sobel gradient magnitude of a [H, W] float32 tensor.

    Pure-torch implementation (no opencv) so this works in any ComfyUI env.
    """
    import torch.nn.functional as F
    device = gray_2d.device
    dtype = gray_2d.dtype
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  2.0,  1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    g = gray_2d.unsqueeze(0).unsqueeze(0)        # [1, 1, H, W]
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy)
    return mag.squeeze(0).squeeze(0)


def _morph_dilate(gray_2d, iterations):
    """Morphological dilation via repeated 3x3 max-pooling. iterations=0 is a no-op."""
    import torch.nn.functional as F
    if iterations <= 0:
        return gray_2d
    x = gray_2d.unsqueeze(0).unsqueeze(0)        # [1, 1, H, W]
    for _ in range(int(iterations)):
        x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x.squeeze(0).squeeze(0)


def _morph_erode(gray_2d, iterations):
    """Morphological erosion = invert -> dilate -> invert. iterations=0 is a no-op."""
    if iterations <= 0:
        return gray_2d
    return 1.0 - _morph_dilate(1.0 - gray_2d, iterations)


def _normalize_lines(mag, mode):
    """Rescale the gradient magnitude to roughly [0, 1] using the chosen mode."""
    if mode == "max":
        denom = float(mag.max())
        if denom > 1e-6:
            return (mag / denom).clamp(0.0, 1.0)
        return mag
    if mode == "percentile_95":
        # Robust to bright outliers — anchor 95th percentile to 1.0.
        flat = mag.flatten()
        denom = float(torch.quantile(flat, 0.95))
        if denom > 1e-6:
            return (mag / denom).clamp(0.0, 1.0)
        return mag
    # "none" — pass through (caller will clamp later if needed).
    return mag


def _lineart_one_frame(rgb_hwc, pre_blur_sigma, line_strength, normalize_mode,
                       gamma, thickness, thinning, soft_threshold,
                       binarize, binarize_threshold, invert):
    """Run the Lineart pipeline on a single [H, W, 3] float32 frame in [0, 1]."""
    r, g, b = rgb_hwc[..., 0], rgb_hwc[..., 1], rgb_hwc[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    if pre_blur_sigma > 0:
        gray = _xdog_gaussian_blur(gray, pre_blur_sigma)

    mag = _sobel_magnitude(gray)
    if line_strength != 1.0:
        mag = mag * float(line_strength)

    mag = _normalize_lines(mag, normalize_mode)
    mag = mag.clamp(0.0, 1.0)

    # Optional soft floor — values below this drop to zero, above are kept.
    # Acts like a continuous noise gate without going fully binary.
    if soft_threshold > 0:
        mag = torch.where(mag >= float(soft_threshold), mag, torch.zeros_like(mag))

    if gamma and abs(gamma - 1.0) > 1e-6:
        mag = mag.clamp(min=1e-6) ** (1.0 / float(gamma))
        mag = mag.clamp(0.0, 1.0)

    # Morphology: thin first (one iteration), then thicken if requested. Thinning
    # before thickening keeps a base line then optionally fattens it.
    if thinning:
        mag = _morph_erode(mag, 1)
    if thickness and thickness > 0:
        mag = _morph_dilate(mag, int(thickness))

    if binarize:
        mag = (mag >= float(binarize_threshold)).to(mag.dtype)

    # Default output is white-lines-on-black (natural Sobel output, which is
    # also the ControlNet lineart preprocessor convention). `invert` flips to
    # black-on-white for a preview-friendly sketch look.
    if invert:
        mag = 1.0 - mag

    return mag.clamp(0.0, 1.0)


class Soze_Lineart:
    """Convert an image (or batch) into a clean line drawing using Sobel
    gradients + morphology + optional binarization. Complement to the Scribble
    XDoG node — same look-and-feel, different algorithm (gradient vs. DoG).

    Defaults give thin white lines on a black background (ControlNet lineart
    preprocessor convention). Toggle `invert` for black lines on white.
    """

    NORMALIZE_MODES = ["percentile_95", "max", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pre_blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.05, "tooltip": "Gaussian denoise before Sobel. 0 = none. Higher hides texture noise."}),
                "line_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Multiplier on the gradient magnitude before normalize / threshold."}),
                "normalize": (cls.NORMALIZE_MODES, {"default": "percentile_95", "tooltip": "How to map raw gradient magnitude to [0,1]. percentile_95 is robust to outliers."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05, "tooltip": "Output gamma. <1 darkens lines, >1 brightens / spreads them."}),
                "soft_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Continuous noise gate — anything below this drops to 0. 0 = off."}),
                "thickness": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Morphological dilation iterations. 0 = thin / native; each step adds ~1 pixel of line width."}),
                "thinning": ("BOOLEAN", {"default": False, "tooltip": "Apply one erosion pass before any thickening (skeleton-ish)."}),
                "binarize": ("BOOLEAN", {"default": False, "tooltip": "Hard black/white using binarize_threshold."}),
                "binarize_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Only used when binarize=True."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Off: white lines on black (ControlNet convention). On: black lines on white (sketch preview)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "status")
    FUNCTION = "process"
    CATEGORY = "image/preprocessors"

    def process(self, image, pre_blur_sigma, line_strength, normalize, gamma,
                soft_threshold, thickness, thinning, binarize, binarize_threshold,
                invert, unique_id=None):
        if image is None or image.shape[0] == 0:
            status = "Skipped: image is None or empty."
            push_node_status(unique_id, status)
            blank = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            blank_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (blank, blank_mask, status)

        src = image.detach().cpu().to(torch.float32)
        if src.shape[-1] == 4:
            src = src[..., :3]

        out_frames = []
        for i in range(src.shape[0]):
            lines_2d = _lineart_one_frame(
                src[i], pre_blur_sigma, line_strength, normalize, gamma,
                thickness, thinning, soft_threshold,
                binarize, binarize_threshold, invert,
            )
            out_frames.append(lines_2d.unsqueeze(-1).expand(-1, -1, 3))

        image_out = torch.stack(out_frames, dim=0)             # [B, H, W, 3]
        mask_out = image_out[..., 0].clone()                   # [B, H, W]

        bits = [f"blur={pre_blur_sigma}", f"strength={line_strength}", f"norm={normalize}", f"gamma={gamma}"]
        if soft_threshold > 0:
            bits.append(f"soft@{soft_threshold}")
        if thinning:
            bits.append("thinned")
        if thickness:
            bits.append(f"thick+{thickness}")
        if binarize:
            bits.append(f"binarized@{binarize_threshold}")
        if invert:
            bits.append("inverted(black-on-white)")
        else:
            bits.append("white-on-black")
        status = (
            f"OK: {src.shape[0]} frame(s) {src.shape[2]}x{src.shape[1]} | "
            + ", ".join(bits)
        )
        push_node_status(unique_id, status)
        return (image_out, mask_out, status)


# ---------------------------------------------------------------------------
# Save Image Batch with per-image filenames
# ---------------------------------------------------------------------------


def _parse_filename_array(s):
    """Parse a filename list from a flexible input.

    Accepts:
      * a Python `list` / `tuple` (e.g. when upstream emits one directly)
      * a JSON-encoded array string         `["a.png", "b.png"]`
      * a Python-style repr string          `['a.png', 'b.png']`  (single quotes)
      * newline-separated values            `a.png\nb.png`
      * a single bare filename              `a.png`
    Empty / None / whitespace entries are dropped. Recursive — if the input
    is a list containing nested lists or strings, each is parsed in turn.
    """
    if s is None:
        return []

    # Already a Python sequence (the bug we just hit — upstream handed us a list).
    if isinstance(s, (list, tuple)):
        out = []
        for item in s:
            out.extend(_parse_filename_array(item))
        return out

    s = str(s).strip()
    if not s:
        return []

    if s.startswith("["):
        # Try strict JSON first
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except json.JSONDecodeError:
            # Fall back to Python literal (handles single-quoted repr strings)
            try:
                arr = ast.literal_eval(s)
                if isinstance(arr, (list, tuple)):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass

    if "\n" in s:
        return [line.strip() for line in s.split("\n") if line.strip()]

    return [s]


def _next_suffix_path(path):
    """Return the path with the next free _NNNNN counter inserted before the
    extension. Matches ComfyUI's standard save-image suffix convention."""
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter:05d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _iter_image_frames(images):
    """Yield each frame as a [H, W, C] uint-tensor view, regardless of the
    input shape ComfyUI hands us.

    Accepts:
      * a single tensor `[B, H, W, C]` (the usual case) -> yields B frames
      * a single tensor `[H, W, C]`                     -> yields 1 frame
      * a Python list of any of the above              -> flattens across
        items (handles upstream nodes with OUTPUT_IS_LIST = (True,)).
    Empty inputs yield nothing.
    """
    if images is None:
        return
    if isinstance(images, list):
        for item in images:
            yield from _iter_image_frames(item)
        return
    # Assume a torch.Tensor at this point.
    if images.dim() == 4:                # [B, H, W, C]
        for b in range(images.shape[0]):
            yield images[b]
    elif images.dim() == 3:              # [H, W, C]
        yield images
    elif images.dim() == 0:
        return
    else:
        # Unknown shape — best-effort: treat as a single frame.
        yield images


class Soze_SaveImageBatchWithFilenames:
    """Save an IMAGE batch using a paired array of filenames.

    Each image in the batch is paired by index with the matching entry in the
    filename array. Both JSON arrays (`["a.png","b.png"]`) and newline-separated
    strings are accepted. Filenames without an extension get the chosen
    `default_format` appended automatically (e.g. `.png`).

    Filenames are placed under `output_path`. Subdirectories inside a filename
    are honored — e.g. `subdir/foo.png` creates `subdir/` inside the output.

    `overwrite`:
      * On  -> existing files at the target path are overwritten in-place.
      * Off -> on collision, append the standard ComfyUI `_00001` counter
               (and `_00002`, etc.) until the path is free.

    Filename array shorter than the image batch: extras are saved with
    auto-generated names like `image_<N>.png`. Filename array longer than
    the batch: the trailing names are ignored (logged in status).
    """

    DEFAULT_FORMAT_CHOICES = ["png", "jpeg", "webp"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of images to save."}),
                "filenames": ("STRING", {"default": "", "multiline": True, "tooltip": "JSON array or newline-separated filenames, one per image. Subdirectories inside a filename are honored."}),
                "output_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Directory to save into. Relative to ComfyUI output dir, or absolute. Empty = output dir root."}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "On: overwrite existing files. Off: append the standard ComfyUI _NNNNN counter on collision."}),
                "default_format": (cls.DEFAULT_FORMAT_CHOICES, {"default": "png", "tooltip": "Format used when a filename has no extension."}),
            },
            "optional": {
                "jpeg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1, "tooltip": "Only used for jpeg / webp."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("saved_paths", "count", "status")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Save an IMAGE batch using a paired array of filenames into a target directory."

    def save(self, images, filenames, output_path, overwrite, default_format,
             jpeg_quality=95, prompt=None, extra_pnginfo=None, unique_id=None):
        # Flatten whatever ComfyUI handed us (single tensor, single frame, or
        # a Python list of either) into one ordered sequence of per-frame
        # tensors. Upstream nodes with OUTPUT_IS_LIST=(True,) deliver a list,
        # which is why the old `images.shape[0]` check blew up.
        frames = list(_iter_image_frames(images))
        if not frames:
            status = "Skipped: empty IMAGE batch."
            push_node_status(unique_id, status)
            return {"ui": {"images": [], "text": [status]}, "result": ("", 0, status)}

        names = _parse_filename_array(filenames)
        batch_count = len(frames)

        # Resolve target directory: empty -> output dir root; relative -> under
        # output dir; absolute -> used as-is.
        if output_path and output_path.strip():
            cleaned = output_path.strip()
            if os.path.isabs(cleaned):
                out_dir = os.path.normpath(cleaned)
            else:
                out_dir = os.path.normpath(os.path.join(self.output_dir, cleaned))
        else:
            out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        default_ext = "jpg" if default_format == "jpeg" else default_format

        saved_paths = []
        ui_results = []
        autogen_count = 0
        overwritten_count = 0
        suffixed_count = 0

        for i in range(batch_count):
            # Pair by index, fall back to auto-generated name when names runs short.
            if i < len(names) and names[i]:
                fname = names[i]
            else:
                fname = f"image_{i}.{default_ext}"
                autogen_count += 1

            # Strip leading separators so the join below stays inside out_dir.
            fname = fname.lstrip("/").lstrip("\\")

            base_no_ext, ext = os.path.splitext(fname)
            if not ext:
                ext = "." + default_ext
                fname = base_no_ext + ext

            # Determine PIL save format from the resolved extension.
            ext_lower = ext.lower().lstrip(".")
            if ext_lower in ("jpg", "jpeg"):
                save_format = "JPEG"
            elif ext_lower == "webp":
                save_format = "WEBP"
            else:
                save_format = "PNG"

            target = os.path.normpath(os.path.join(out_dir, fname))

            # Make sure any subdirectory inside the filename exists.
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)

            if os.path.exists(target):
                if overwrite:
                    overwritten_count += 1
                else:
                    target = _next_suffix_path(target)
                    suffixed_count += 1

            # Convert [H, W, C] tensor (float32 in [0,1]) -> PIL.
            arr = frames[i].cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            save_kwargs = {}
            if save_format == "JPEG":
                if pil.mode == "RGBA":
                    pil = pil.convert("RGB")
                save_kwargs["quality"] = int(jpeg_quality)
            elif save_format == "WEBP":
                save_kwargs["quality"] = int(jpeg_quality)
            else:
                # PNG: embed metadata like the standard SaveImage node.
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    save_kwargs["pnginfo"] = metadata
                save_kwargs["compress_level"] = self.compress_level

            pil.save(target, save_format, **save_kwargs)
            saved_paths.append(target)

            # UI thumbnail entry — only for files under the output dir.
            try:
                rel_path = os.path.relpath(target, self.output_dir)
                if not rel_path.startswith(".."):
                    ui_results.append({
                        "filename": os.path.basename(target),
                        "subfolder": os.path.dirname(rel_path),
                        "type": self.type,
                    })
            except ValueError:
                # Different drive on Windows — skip UI preview but still save.
                pass

        notes = []
        if autogen_count:
            notes.append(f"{autogen_count} auto-named (filename array shorter than batch)")
        if len(names) > batch_count:
            notes.append(f"{len(names) - batch_count} extra filename(s) ignored")
        if overwritten_count:
            notes.append(f"{overwritten_count} overwritten")
        if suffixed_count:
            notes.append(f"{suffixed_count} suffixed with _NNNNN to avoid collision")

        status = f"OK: saved {len(saved_paths)} image(s) to {out_dir}"
        if notes:
            status += " | " + ", ".join(notes)
        push_node_status(unique_id, status)

        return {
            "ui": {"images": ui_results, "text": [status]},
            "result": ("\n".join(saved_paths), len(saved_paths), status),
        }


class Soze_LoadImagesFromUrlList:
    """Load up to 9 images from a multiline string of URLs (one per row)."""

    MAX_IMAGES = 9

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": f"One URL per row. Up to {cls.MAX_IMAGES} rows are used; empty/missing rows output None.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",) * MAX_IMAGES + ("INT", "STRING")
    RETURN_NAMES = tuple(f"image{i+1}" for i in range(MAX_IMAGES)) + ("loaded_count", "status")
    FUNCTION = "load"
    CATEGORY = "image"

    def _load_one(self, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return pil2tensor(image)

    def load(self, urls, unique_id=None):
        log = EventLog()
        rows = (urls or "").splitlines()

        outputs = [None] * self.MAX_IMAGES
        loaded = 0
        errors = 0

        for i in range(self.MAX_IMAGES):
            if i >= len(rows):
                break
            url = rows[i].strip()
            if not url:
                continue
            try:
                outputs[i] = self._load_one(url)
                loaded += 1
                push_node_status(unique_id, f"OK row {i+1}: {url}", log)
            except Exception as e:
                errors += 1
                push_node_status(unique_id, f"ERROR row {i+1} ({url}): {e!r}", log)

        headline = f"OK: loaded {loaded}/{self.MAX_IMAGES} image(s)"
        if errors:
            headline += f", {errors} error(s)"
        push_node_status(unique_id, headline, log)

        return tuple(outputs) + (loaded, finalize_status(headline, log))


class Soze_ImageCrop:
    """Crop pixels off the edges of an image or image batch.

    Accepts a single IMAGE or an IMAGE batch ([B,H,W,C]) and removes `top`,
    `bottom`, `left`, and `right` pixels from each frame. The same crop is
    applied to every frame, so the batch stays aligned. Crop amounts are
    clamped so at least a 1x1 image remains.
    """

    @classmethod
    def INPUT_TYPES(cls):
        edge = {"default": 0, "min": 0, "max": 8192, "step": 1}
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image or image batch to crop."}),
                "top": ("INT", {**edge, "tooltip": "Pixels to remove from the top edge."}),
                "bottom": ("INT", {**edge, "tooltip": "Pixels to remove from the bottom edge."}),
                "left": ("INT", {**edge, "tooltip": "Pixels to remove from the left edge."}),
                "right": ("INT", {**edge, "tooltip": "Pixels to remove from the right edge."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "status")
    FUNCTION = "crop"
    CATEGORY = "image"

    def crop(self, image, top, bottom, left, right, unique_id=None):
        if image is None:
            status = "Skipped: image is None."
            push_node_status(unique_id, status)
            return (None, 0, 0, status)

        # Normalize to [B,H,W,C].
        if image.dim() == 3:
            image = image.unsqueeze(0)

        b, h, w, c = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

        # Clamp negatives to 0.
        top = max(0, int(top))
        bottom = max(0, int(bottom))
        left = max(0, int(left))
        right = max(0, int(right))

        # Clamp so opposing crops never remove the whole axis (leave >= 1px).
        clamped = False
        if top + bottom >= h:
            top = min(top, h - 1)
            bottom = min(bottom, h - 1 - top)
            clamped = True
        if left + right >= w:
            left = min(left, w - 1)
            right = min(right, w - 1 - left)
            clamped = True

        y0, y1 = top, h - bottom
        x0, x1 = left, w - right

        cropped = image[:, y0:y1, x0:x1, :]
        new_h, new_w = cropped.shape[1], cropped.shape[2]

        status = (
            f"OK: {w}x{h} -> {new_w}x{new_h} "
            f"(T{top} B{bottom} L{left} R{right}, batch={b})"
        )
        if clamped:
            status += " [crop clamped to keep >=1px]"
        push_node_status(unique_id, status)
        return (cropped, new_w, new_h, status)



