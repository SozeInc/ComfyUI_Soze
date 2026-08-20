"""Combine two (image, audio) clips into a single image batch + audio track.

Frames are concatenated in order (clip 1 then clip 2); audio is concatenated in
time (clip 1 then clip 2). Differing frame sizes are resized to the first clip;
differing sample rates are resampled to the first clip; differing channel counts
are matched (mono<->stereo).
"""

import logging

import torch
from comfy.utils import common_upscale

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


def _align_and_cat_images(images):
    """Concatenate [B,H,W,C] image tensors along the frame axis (dim 0)."""
    imgs = [i for i in images if i is not None]
    if not imgs:
        return None
    base = imgs[0]
    base_h, base_w = base.shape[1], base.shape[2]
    aligned = [base]
    for t in imgs[1:]:
        if t.shape[1] != base_h or t.shape[2] != base_w:
            t = common_upscale(t.movedim(-1, 1), base_w, base_h, "bilinear", "center").movedim(1, -1)
        aligned.append(t)
    # Match channel count (e.g. RGB vs RGBA) by trimming to the common minimum.
    min_c = min(a.shape[-1] for a in aligned)
    aligned = [a[..., :min_c] for a in aligned]
    return torch.cat(aligned, dim=0)


def _match_channels(wf, target_c):
    """Match an audio waveform [B,C,S] to target_c channels."""
    c = wf.shape[1]
    if c == target_c:
        return wf
    if c == 1 and target_c > 1:
        return wf.repeat(1, target_c, 1)
    if c > 1 and target_c == 1:
        return wf.mean(dim=1, keepdim=True)
    if c > target_c:
        return wf[:, :target_c, :]
    # c < target_c (and c > 1): pad by repeating the first channel.
    pad = wf[:, :1, :].repeat(1, target_c - c, 1)
    return torch.cat([wf, pad], dim=1)


def _concat_audio(audios):
    """Concatenate ComfyUI AUDIO dicts in time. Returns a new AUDIO dict."""
    auds = [a for a in audios if a is not None]
    if not auds:
        return None
    base = auds[0]
    base_wf = base["waveform"]
    base_sr = int(base["sample_rate"])
    parts = [base_wf]
    for a in auds[1:]:
        wf = a["waveform"]
        sr = int(a["sample_rate"])
        if sr != base_sr:
            import torchaudio
            wf = torchaudio.functional.resample(wf, sr, base_sr)
        if wf.shape[1] != base_wf.shape[1]:
            wf = _match_channels(wf, base_wf.shape[1])
        # Align batch dim (normally 1) by trimming everything to the minimum.
        if wf.shape[0] != parts[0].shape[0]:
            b = min(wf.shape[0], min(p.shape[0] for p in parts))
            parts = [p[:b] for p in parts]
            wf = wf[:b]
        parts.append(wf)
    return {"waveform": torch.cat(parts, dim=-1), "sample_rate": base_sr}


class Soze_CombineVideo:
    """Combine two (image, audio) clips into one image batch + one audio track.

    Frames are appended in order (clip 1 then clip 2); audio is joined end-to-end.
    All inputs are optional, so you can also use it to merge just images, just
    audio, or pass a single clip through.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Clip 1 frames."}),
                "audio_1": ("AUDIO", {"tooltip": "Clip 1 audio."}),
                "image_2": ("IMAGE", {"tooltip": "Clip 2 frames (appended after clip 1)."}),
                "audio_2": ("AUDIO", {"tooltip": "Clip 2 audio (appended after clip 1)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("image", "audio", "status")
    FUNCTION = "combine"
    CATEGORY = "video"

    def combine(self, image_1=None, audio_1=None, image_2=None, audio_2=None, unique_id=None):
        log = EventLog()

        out_image = _align_and_cat_images([image_1, image_2])
        out_audio = _concat_audio([audio_1, audio_2])

        # Frame/audio summaries for the status line.
        img_frames = out_image.shape[0] if out_image is not None else 0
        img_dims = (f"{out_image.shape[2]}x{out_image.shape[1]}" if out_image is not None else "-")
        if out_audio is not None:
            wf = out_audio["waveform"]
            sr = int(out_audio["sample_rate"])
            secs = wf.shape[-1] / sr if sr else 0.0
            audio_desc = f"{sr}Hz, {wf.shape[1]}ch, {secs:.2f}s"
        else:
            audio_desc = "-"

        if out_image is None and out_audio is None:
            headline = "Skipped: no image or audio inputs connected."
            push_node_status(unique_id, headline, log)
            return (None, None, finalize_status(headline, log))

        headline = f"OK: image[{img_frames} frames {img_dims}] + audio[{audio_desc}]"
        push_node_status(unique_id, headline, log)
        return (out_image, out_audio, finalize_status(headline, log))
