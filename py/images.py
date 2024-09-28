import torch
import os
import numpy as np
import hashlib

from PIL import Image, ImageOps, ImageSequence, ImageFile
from comfy.utils import ProgressBar, common_upscale

import node_helpers
import folder_paths


class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_NAMES = ("Image", "Mask", "ImageFilePath", "ImageFilename", "ImageFilenameNoExt")
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
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Image", "Mask", "LoadCount", "InputPath", "ImageFileNamePath", "ImageFilename", "ImageFilenameNoExt")
    FUNCTION = "load_images"

    CATEGORY = "image"

    def load_images(self, folder, image_load_cap, start_index):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder} cannot be found.'")
        dir_files = os.listdir(folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{folder}'.")

        # Filter files by extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(folder, x) for x in dir_files]

        # start at start_index
        dir_files = dir_files[start_index:]

        images = []
        masks = []
        image_path_list = []

        limit_images = False
        if image_load_cap > 0:
            limit_images = True
        image_count = 0

        has_non_empty_mask = False

        for image_path in dir_files:
            if os.path.isdir(image_path) and os.path.ex:
                continue
            if limit_images and image_count >= image_load_cap:
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
            return (images[0], masks[0], 1, folder, input_filenamepath, input_filename, input_filename_no_ext)

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

            return (image1, mask1, len(images), folder, input_filenamepath, input_filename, input_filename_no_ext)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")
