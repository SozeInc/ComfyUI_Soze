import torch
import os
import re
import numpy as np
import hashlib
import re
import requests
from numpy import ndarray

from typing import Tuple, List, Dict, Any, Optional

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, TYPE_CHECKING

from PIL import ImageFont, ImageDraw, Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.font_manager as fm
from torch import Tensor

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
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_NAMES = ("Image", "Mask", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Image_Changed")
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "BOOL")
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

        previous_input_filename = read_from_file('sozeimagecache.txt')
        write_to_file('sozeimagecache.txt', input_filename)
        return (output_image, output_mask, input_filepath, input_filename, input_filename_no_ext, previous_input_filename != input_filename)

    @classmethod
    def IS_CHANGED(s, image):
        previous_input_filepath = s.read_previous_image_filename()
        if previous_input_filepath != image:
            return True
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
class Soze_LoadImagesFromFolder:
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

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "STRING", "STRING", "BOOL")
    RETURN_NAMES = ("Image", "Mask", "Load_Count", "Input_Path",  "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext", "Image_Changed")
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

            previous_input_filename = read_from_file('sozeimagebatchcache.txt')
            write_to_file('sozeimagebatchcache.txt', input_filename)

            return (image1, mask1, len(images), Input_Folder, input_filenamepath, input_filename, input_filename_no_ext, previous_input_filename != input_filename)

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
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "Image_Filename_Path", "Image_Filename", "Image_Filename_No_Ext")
    FUNCTION = "switch"
    CATEGORY = "batch"

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
    FUNCTION = "put_overlay"

    CATEGORY = "image"

    def put_overlay(
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
            print(f"Splits: {split} | Base Color: {base_color}")

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
        subbed = re.sub("\D", "", str(file_name))
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
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "rgb_colors",
        "hex_colors"
    )
    FUNCTION = "main"
    CATEGORY = "image"

    def main(
        self,
        input_image: torch.Tensor,
        num_colors: int = 5,
        exclude_colors: str = "",
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

        return (
            ", ".join(rgb_colors),
            ", ".join(hex_colors)
        )
    

#Code from https://github.com/dzqdzq/ComfyUI-crop-alpha
class Soze_AlphaCropAndPositionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "maintain_aspect": (["True", "False"], {"default": "True"}),
                "left_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "top_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "right_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "bottom_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")

    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, image, maintain_aspect, left_padding: int = 0, right_padding: int = 0, top_padding: int = 0, bottom_padding: int = 0):
        cropped_images = []
        cropped_masks = []
        alpha_images = []

        for img in image:
            alpha = img[..., 3]

            height = img.shape[0]
            width = img.shape[1]
            mask = (alpha > 0.01)

            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)

            ymin, ymax = self._find_boundary(rows)
            xmin, xmax = self._find_boundary(cols)

            if ymin is None or xmin is None:
                cropped_images.append(img)
                cropped_masks.append(torch.zeros_like(alpha))
                alpha_images.append(img)
                continue

            ymin = max(0, ymin - top_padding)
            ymax = min(height, ymax + bottom_padding)
            xmin = max(0, xmin - left_padding)
            xmax = min(width, xmax + right_padding)

            cropped = img[ymin:ymax, xmin:xmax, :4]
            cropped_mask = alpha[ymin:ymax, xmin:xmax]

            # Apply padding to the cropped image
            padded_height = ymax - ymin + top_padding + bottom_padding
            padded_width = xmax - xmin + left_padding + right_padding

            if maintain_aspect == "True":
                if padded_height > padded_width:
                    pad = (padded_height - padded_width) // 2
                    left_padding += pad
                    right_padding += pad
                    padded_width = padded_height
                else:
                    pad = (padded_width - padded_height) // 2
                    top_padding += pad
                    bottom_padding += pad
                    padded_height = padded_width

            padded_image = torch.zeros((padded_height, padded_width, 4), dtype=img.dtype)
            padded_image[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin), :] = cropped

            padded_mask = torch.zeros((padded_height, padded_width), dtype=alpha.dtype)
            padded_mask[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin)] = cropped_mask

            cropped_masks.append(padded_mask)
            alpha_images.append(padded_image)

        return alpha_images, cropped_masks, xmax - xmin + left_padding + right_padding, ymax - ymin + top_padding + bottom_padding
    
    def _find_boundary(self, arr):
        nz = torch.nonzero(arr)
        if nz.numel() == 0:
            return (None, None)
        return (nz[0].item(), nz[-1].item() + 1)


class Soze_ShrinkImage:
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
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

    def shrink_image(self, image, mode, resize_algorithm, maintain_aspect, scale=None, width=None, height=None):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        algorithm = resize_algorithms[resize_algorithm]

        output_images = []
        for img in image:
            img = to_pil_image(img.permute(2, 0, 1))
            scale = self.calculate_scale(img, mode, maintain_aspect, scale, width, height)
            resized_img = self.shrink_image_with_scale(img, scale, algorithm)
            resized_img_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_img_np = torch.from_numpy(resized_img_np)
            output_images.append(resized_img_np)

        return (output_images,)
