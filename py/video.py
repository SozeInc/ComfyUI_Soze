import logging
import os
import shutil
import subprocess
import uuid

import folder_paths  # type: ignore
from comfy.comfy_types import IO, ComfyNodeABC  # noqa: F401
from comfy_api.input_impl import VideoFromFile

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.flv')


def _safe_path_arg(path: str, label: str) -> str:
    """Reject paths ffmpeg would interpret as flags. Returns an absolute path."""
    if not path:
        raise ValueError(f"{label} path is empty")
    abs_path = os.path.abspath(path)
    if os.path.basename(abs_path).startswith('-'):
        raise ValueError(f"{label} filename starts with '-' which ffmpeg would parse as a flag: {abs_path}")
    return abs_path


class Soze_AppendToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video": ("STRING", {"default": "", "multiline": False}),
                "append_video": ("STRING", {"default": "", "multiline": False}),
                "save_filename_path": ("STRING", {"default": "", "multiline": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_file", "status")
    FUNCTION = "append_video"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def append_video(self, base_video, append_video, save_filename_path, unique_id=None):
        log = EventLog()
        base_exists = base_video and os.path.exists(base_video)
        append_exists = append_video and os.path.exists(append_video)
        push_node_status(unique_id, f"Inputs: base={'YES' if base_exists else 'NO'}, append={'YES' if append_exists else 'NO'}", log)

        if not base_exists and not append_exists:
            headline = "Skipped: neither base_video nor append_video exists."
            push_node_status(unique_id, headline, log)
            return ("", finalize_status(headline, log))

        # Determine output filename
        if save_filename_path:
            filename = save_filename_path.split(os.sep)[-1]
        elif base_exists:
            filename = os.path.basename(base_video)
        else:
            filename = os.path.basename(append_video)

        output_file = os.path.join(
            folder_paths.get_output_directory(),
            save_filename_path if save_filename_path else filename,
        )

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = f"temp_{uuid.uuid4().hex}_{filename}"
        if not temp_filename.endswith(".mp4"):
            temp_filename += ".mp4"
        temp_output_file = os.path.join(temp_dir, temp_filename)

        try:
            if base_exists and append_exists:
                base_arg = _safe_path_arg(base_video, "base_video")
                append_arg = _safe_path_arg(append_video, "append_video")
                logger.info("Appending video. base=%s append=%s", base_arg, append_arg)
                push_node_status(unique_id, "Concatenating with ffmpeg...", log)

                command = [
                    "ffmpeg", "-y",
                    "-i", base_arg,
                    "-i", append_arg,
                    "-filter_complex",
                    "[1:v][0:v]scale2ref[v1][v0];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]",
                    "-map", "[v]",
                    "-map", "[a]",
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "slow",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    temp_output_file,
                ]
                try:
                    subprocess.run(command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr.decode(errors="replace") if e.stderr else str(e)
                    logger.error("FFmpeg failed: %s", error_msg)
                    headline = f"ERROR: ffmpeg failed: {error_msg[:200]}"
                    push_node_status(unique_id, headline, log)
                    raise RuntimeError(f"FFmpeg failed to append videos: {error_msg}")
            elif base_exists:
                push_node_status(unique_id, "Append missing — copying base only.", log)
                shutil.copy2(base_video, temp_output_file)
            else:
                push_node_status(unique_id, "Base missing — copying append only.", log)
                shutil.copy2(append_video, temp_output_file)

            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except OSError as e:
                    output_file_new = output_file.replace(".mp4", f"_new_{uuid.uuid4().hex}.mp4")
                    logger.warning("Could not remove existing file, saving to %s: %s", output_file_new, e)
                    push_node_status(unique_id, f"Existing output locked, redirecting to {output_file_new}", log)
                    output_file = output_file_new

            shutil.move(temp_output_file, output_file)
            logger.info("Video merge completed: %s", output_file)
            try:
                size = os.path.getsize(output_file)
                size_msg = f"{size:,} bytes"
            except OSError:
                size_msg = "unknown size"
            headline = f"OK: {size_msg} -> {output_file}"
            push_node_status(unique_id, headline, log)
            return (output_file, finalize_status(headline, log))
        finally:
            if os.path.exists(temp_output_file):
                try:
                    os.remove(temp_output_file)
                except OSError:
                    pass


class Soze_LoadVideosFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Input_Folder": ("STRING", {"default": ""}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (IO.VIDEO, "STRING", "STRING", "STRING", "STRING", "BOOL", "STRING")
    RETURN_NAMES = ("Video", "Input_Path", "Video_Filename_Path", "Video_Filename", "Video_Filename_No_Ext", "Video_Changed", "status")
    FUNCTION = "load_videos_from_folder"
    CATEGORY = "image"

    @classmethod
    def IS_CHANGED(cls, Input_Folder, index, unique_id=None):
        try:
            entries = sorted(
                f for f in os.listdir(Input_Folder)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            )
            if 0 <= index < len(entries):
                target = os.path.join(Input_Folder, entries[index])
                st = os.stat(target)
                return f"{target}:{st.st_mtime_ns}:{st.st_size}"
            return f"oob:{Input_Folder}:{index}:{len(entries)}"
        except OSError as e:
            return f"err:{Input_Folder}:{index}:{e}"

    def load_videos_from_folder(self, Input_Folder, index, unique_id=None):
        log = EventLog()
        push_node_status(unique_id, f"Scanning {Input_Folder} for videos...", log)
        if not os.path.isdir(Input_Folder):
            push_node_status(unique_id, f"ERROR: folder not found: {Input_Folder}", log)
            raise FileNotFoundError(f"Folder not found: {Input_Folder}")
        dir_files = os.listdir(Input_Folder)
        if not dir_files:
            push_node_status(unique_id, f"ERROR: folder is empty: {Input_Folder}", log)
            raise FileNotFoundError(f"Folder is empty: {Input_Folder}")

        dir_files = sorted(f for f in dir_files if f.lower().endswith(VIDEO_EXTENSIONS))
        dir_files = [os.path.join(Input_Folder, x) for x in dir_files]
        push_node_status(unique_id, f"Found {len(dir_files)} video(s).", log)

        if index >= len(dir_files):
            push_node_status(unique_id, f"ERROR: index {index} out of range (count={len(dir_files)}).", log)
            raise IndexError(f"Index {index} is out of range. Only {len(dir_files)} videos found.")

        video_path = dir_files[index]
        if os.path.isdir(video_path):
            push_node_status(unique_id, f"ERROR: path at index {index} is a directory.", log)
            raise ValueError(f"Path at index {index} is a directory, not a video file.")

        input_filepath = folder_paths.get_annotated_filepath(video_path)
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]
        video = VideoFromFile(input_filepath)

        headline = f"OK: {index+1}/{len(dir_files)} — {input_filename}"
        push_node_status(unique_id, headline, log)
        return (video, Input_Folder, video_path, input_filename, input_filename_no_ext, True, finalize_status(headline, log))
