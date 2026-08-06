"""Audio loaders for the Soze pack.

Soze_LoadAudio behaves like ComfyUI's built-in Load Audio node (same input-dir
picker + upload button, same AUDIO output), but adds the filename / path outputs
that the Soze Load Image node provides.
"""

import os
import hashlib

import folder_paths

from .status_utils import push_node_status
from .utils import read_from_file, write_to_file


AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".wma", ".opus", ".aiff", ".aif")


def _list_input_audio():
    input_dir = folder_paths.get_input_directory()
    try:
        names = os.listdir(input_dir)
    except OSError:
        return []
    files = [
        f for f in names
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(AUDIO_EXTENSIONS)
    ]
    return sorted(files)


def _load_audio_dict(audio_path):
    """Load an audio file into a ComfyUI AUDIO dict ({waveform, sample_rate}).

    Prefers ComfyUI's own Load Audio implementation so behavior matches the
    default node exactly; falls back to torchaudio if that import path changes.
    """
    # Preferred: delegate to ComfyUI's built-in loader for identical behavior.
    # Its load() resolves via annotated filepath, so pass the basename.
    try:
        from comfy_extras.nodes_audio import LoadAudio as _ComfyLoadAudio
        result = _ComfyLoadAudio().load(os.path.basename(audio_path))
        if isinstance(result, tuple) and result and isinstance(result[0], dict):
            return result[0]
    except Exception:
        pass

    # Fallback: load directly with torchaudio.
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


class Soze_LoadAudio:
    """Load an audio file from the input directory (with upload button), and
    expose filename / path outputs like the Soze Load Image node."""

    @classmethod
    def INPUT_TYPES(cls):
        # NOTE: we intentionally do NOT set {"audio_upload": True}. On current
        # ComfyUI the legacy upload flag creates an AUDIOUPLOAD widget whose
        # update handler expects a companion audio-player UI widget that only
        # the new schema-based LoadAudio creates — so it throws during node
        # creation ("Cannot read properties of undefined (reading 'element')")
        # and the node silently fails to add. A plain combo of input-dir audio
        # files works reliably. Drop files into ComfyUI/input to populate it.
        #
        # An EMPTY option list also breaks instantiation, so guarantee >= 1.
        files = _list_input_audio() or [""]
        return {
            "required": {
                "audio": (files,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "BOOL", "STRING")
    RETURN_NAMES = ("Audio", "Audio_Filename_Path", "Audio_Filename", "Audio_Filename_No_Ext", "Audio_Changed", "status")
    FUNCTION = "load_audio"
    CATEGORY = "audio"

    def load_audio(self, audio, unique_id=None):
        input_filepath = folder_paths.get_annotated_filepath(audio)
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]

        if not input_filepath or not os.path.isfile(input_filepath):
            headline = f"ERROR: audio file not found: {input_filepath}"
            push_node_status(unique_id, headline)
            raise FileNotFoundError(f"Audio file not found: {input_filepath}")

        audio_dict = _load_audio_dict(input_filepath)

        previous_input_filename = read_from_file('sozeaudiocache.txt')
        write_to_file('sozeaudiocache.txt', input_filename)
        changed = previous_input_filename != input_filename

        try:
            sr = int(audio_dict.get("sample_rate", 0))
            wf = audio_dict.get("waveform")
            ch = wf.shape[-2] if wf is not None and wf.dim() >= 2 else 0
            samples = wf.shape[-1] if wf is not None else 0
            dur = (samples / sr) if sr else 0.0
            details = f"{sr} Hz, {ch} ch, {dur:.2f}s"
        except Exception:
            details = "loaded"
        status = f"OK: {input_filename} ({details}, changed={changed})"
        push_node_status(unique_id, status)
        return (audio_dict, input_filepath, input_filename, input_filename_no_ext, changed, status)

    @classmethod
    def IS_CHANGED(cls, audio, unique_id=None):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio, unique_id=None):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True
