import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os

from nodes import MAX_RESOLUTION
from comfy.utils import common_upscale
import folder_paths
from comfy import model_management
try:
    from server import PromptServer
except:
    PromptServer = None

script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Soze_ImageResizeWithAspectCorrection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "aspect_correction": (["crop", "pad", "none"], { "default": "none", "tooltip": "How to handle aspect ratio differences between input and target size." }),
                "pad_color": ("STRING", { "default": "0, 0, 0", "tooltip": "Color to use for padding."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
            },
            "optional" : {
                "mask": ("MASK",),
            },
             "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK",)
    RETURN_NAMES = ("IMAGE", "width", "height", "mask",)
    FUNCTION = "resize"
    CATEGORY = "soze"
    DESCRIPTION = """
Resizes the image while keeping the aspect ratio, adjusting to fit the target dimensions based on aspect_correction.
- 'crop': Resize to cover the target area, then crop to exact size.
- 'pad': Resize to fit within the target area, then pad to exact size.
- 'none': Resize to fit within the target area without cropping or padding, output size may not match exactly.
"""

    def resize(self, image, width, height, aspect_correction, pad_color, crop_position, unique_id, device="cpu", mask=None, per_batch=0):
        if image is None:
            return (None, 0, 0, None)

        B, H, W, C = image.shape

        device = torch.device("cpu")

        if width == 0:
            width = W
        if height == 0:
            height = H

        target_w = width
        target_h = height

        # Initialize variables
        pad_left = pad_right = pad_top = pad_bottom = 0
        crop_x = crop_y = crop_w = crop_h = 0
        output_width = target_w
        output_height = target_h

        if aspect_correction == "none":
            scale = min(target_w / W, target_h / H) if W > 0 and H > 0 else 1.0
            new_width = round(W * scale)
            new_height = round(H * scale)
            output_width = new_width
            output_height = new_height
        elif aspect_correction == "crop":
            scale = max(target_w / W, target_h / H) if W > 0 and H > 0 else 1.0
            new_width = round(W * scale)
            new_height = round(H * scale)
            crop_x = (new_width - target_w) // 2
            crop_y = (new_height - target_h) // 2
            crop_w = target_w
            crop_h = target_h
        
        if aspect_correction == "pad":
            scale = min(target_w / W, target_h / H) if W > 0 and H > 0 else 1.0
            new_width = round(W * scale)
            new_height = round(H * scale)
            # Calculate padding
            pad_left = (target_w - new_width) // 2
            pad_right = target_w - new_width - pad_left
            pad_top = (target_h - new_height) // 2
            pad_bottom = target_h - new_height - pad_top
        
        width = new_width
        height = new_height

        # Preflight estimate (log-only when batching is active)
        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()  # typically 4 for float32
                est_total_bytes = B * height * width * C * bytes_per_elem
                est_mb = est_total_bytes / (1024 * 1024)
                msg = f"<tr><td>Resize v2</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>"
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(msg, unique_id)
                    except:
                        pass
                else:
                    print(f"[ImageResizeKJv2] estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}")
            except:
                pass

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            # Avoid unnecessary clones; only move if needed
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            out_image = common_upscale(out_image.movedim(-1,1), width, height, "bicubic", crop="disabled").movedim(1,-1)
            if out_mask is not None:
                out_mask = common_upscale(out_mask.unsqueeze(1), width, height, "bicubic", crop="disabled").squeeze(1)

            # Crop logic
            if aspect_correction == "crop":
                out_image = out_image.narrow(-2, crop_x, crop_w).narrow(-3, crop_y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, crop_x, crop_w).narrow(-2, crop_y, crop_h)

            # Pad logic
            if aspect_correction == "pad" and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                pad_mode = "color"
                out_image, out_mask = Soze_ImagePadKJ.pad(self, out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode, mask=out_mask)

            return out_image, out_mask

        # If batching disabled (per_batch==0) or batch fits, process whole batch
        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(image, mask, pad_left, pad_right, pad_top, pad_bottom)
        else:
            chunks = []
            mask_chunks = [] if mask is not None else None
            total_batches = (B + per_batch - 1) // per_batch
            current_batch = 0
            for start_idx in range(0, B, per_batch):
                current_batch += 1
                end_idx = min(start_idx + per_batch, B)
                sub_img = image[start_idx:end_idx]
                sub_mask = mask[start_idx:end_idx] if mask is not None else None
                sub_out_img, sub_out_mask = _process_subbatch(sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                chunks.append(sub_out_img.cpu())
                if mask is not None:
                    mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                # Per-batch progress update
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(
                            f"<tr><td>Resize v2</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}</td></tr>",
                            unique_id
                        )
                    except:
                        pass
                else:
                    try:
                        print(f"[ImageResizeKJv2] batch {current_batch}/{total_batches} · images {end_idx}/{B}")
                    except:
                        pass
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if out_mask is not None else torch.zeros(64,64, device=torch.device("cpu"), dtype=torch.float32))



class Soze_ImagePadKJ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "extra_padding": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                    "pad_mode": (["edge", "edge_pixel", "color", "pillarbox_blur"],),
                    "color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
                  },
                "optional": {
                    "mask": ("MASK", ),
                    "target_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                    "target_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, "forceInput": True}),
                }
                }
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("images", "masks",)
    FUNCTION = "pad"
    CATEGORY = "soze"
    DESCRIPTION = "Pad the input image and optionally mask with the specified padding."
        
    def pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        # Ensure bg_color has the same number of channels as the image
        while len(bg_color) < C:
            bg_color.append(1.0 if len(bg_color) == 3 else 0.0)  # Alpha=1.0 for opaque, or 0.0 for extra channels
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = common_upscale(src_b, bg_w, bg_h, "lanczos", crop="disabled")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
                if pad_mode == "edge":
                    # Pad with edge color (mean)
                    top_edge = image[b, 0, :, :]
                    bottom_edge = image[b, H-1, :, :]
                    left_edge = image[b, :, 0, :]
                    right_edge = image[b, :, W-1, :]
                    out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                    out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                    out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                    out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                elif pad_mode == "edge_pixel":
                    # Pad with exact edge pixel values
                    for y in range(pad_top):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                    for y in range(pad_top+H, padded_height):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                    for x in range(pad_left):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                    for x in range(pad_left+W, padded_width):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                    out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                    out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                    out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                    out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                else:
                    # Pad with specified background color
                    out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask, 
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)
