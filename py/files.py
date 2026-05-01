import logging
import os
import re

import folder_paths
import requests

from .utils import read_from_file, write_to_file

DOWNLOAD_TIMEOUT = (10, 600)  # (connect, read) seconds
DOWNLOAD_CHUNK_SIZE = 64 * 1024

logger = logging.getLogger(__name__)


def _folder_signature(input_folder: str, predicate=None) -> str:
    """Cheap fingerprint of a folder's contents for IS_CHANGED."""
    try:
        entries = sorted(os.listdir(input_folder))
        if predicate is not None:
            entries = [e for e in entries if predicate(e)]
        parts = []
        for name in entries:
            path = os.path.join(input_folder, name)
            try:
                st = os.stat(path)
                parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{name}:?")
        return f"{input_folder}|" + "|".join(parts)
    except OSError as e:
        return f"err:{input_folder}:{e}"


def _file_signature(filepath: str) -> str:
    try:
        st = os.stat(filepath)
        return f"{filepath}:{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return f"missing:{filepath}"


class Soze_LoadFilesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "input_file_extensions": ("STRING", {"default": ".psb", "description": "Comma-separated list of extensions (e.g., .txt,.json)"}),
            },
            "optional": {
                "file_load_count": ("INT", {"default": 1, "min": 0, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("File_Path", "Load_Count", "Input_Folder", "Filename", "Filename_No_Ext")
    FUNCTION = "load_files"
    CATEGORY = "file"

    @classmethod
    def IS_CHANGED(cls, input_folder, input_file_extensions, file_load_count=1, index=0):
        exts = tuple(e.strip().lower() for e in input_file_extensions.split(','))
        sig = _folder_signature(input_folder, predicate=lambda f: f.lower().endswith(exts))
        return f"{sig}|{file_load_count}|{index}"

    def load_files(self, input_folder, input_file_extensions, file_load_count=1, index=0):
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Folder not found: {input_folder}")

        dir_files = os.listdir(input_folder)
        if not dir_files:
            raise FileNotFoundError(f"Folder is empty: {input_folder}")

        valid_extensions = tuple(ext.strip().lower() for ext in input_file_extensions.split(','))
        dir_files = [f for f in dir_files if f.lower().endswith(valid_extensions)]
        if not dir_files:
            raise FileNotFoundError(f"No files found with extensions: {input_file_extensions}")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(input_folder, x) for x in dir_files][index:]

        file_paths = []
        limit_files = file_load_count > 0
        for file_path in dir_files:
            if limit_files and len(file_paths) >= file_load_count:
                break
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        if not file_paths:
            raise FileNotFoundError(f"No valid files found in: {input_folder}")

        current_file = file_paths[0]
        read_from_file('sozefilebatchcache.txt')
        write_to_file('sozefilebatchcache.txt', current_file)

        return (
            current_file,
            len(file_paths),
            input_folder,
            os.path.basename(current_file),
            os.path.splitext(os.path.basename(current_file))[0],
        )


class Soze_FileLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filepath": ("STRING", )}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"
    CATEGORY = "Soze Nodes"

    @classmethod
    def IS_CHANGED(cls, filepath):
        return _file_signature(filepath)

    def read(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return (f.read(),)
        except OSError as e:
            logger.error("Error reading file %s: %s", filepath, e)
            return ("",)


class Soze_DoesFileExist:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"filepath": ("STRING", )}}

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "does_exist"
    CATEGORY = "Soze Nodes"

    @classmethod
    def IS_CHANGED(cls, filepath):
        return _file_signature(filepath)

    def does_exist(self, filepath):
        return (os.path.isfile(filepath),)


class Soze_LoadFilesWithPattern:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "filename_pattern": ("STRING", {"default": ".*", "description": "Regex pattern to match filenames"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("File_Paths", "Load_Count")
    FUNCTION = "load_file_with_pattern"
    CATEGORY = "Soze Nodes"

    @classmethod
    def IS_CHANGED(cls, input_folder, filename_pattern):
        try:
            compiled = re.compile(filename_pattern)
        except re.error as e:
            return f"badpattern:{filename_pattern}:{e}"
        sig = _folder_signature(input_folder, predicate=lambda f: bool(compiled.search(f)))
        return f"{sig}|{filename_pattern}"

    def load_file_with_pattern(self, input_folder, filename_pattern):
        if not os.path.isdir(input_folder):
            return ("", 0)

        try:
            compiled = re.compile(filename_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {filename_pattern!r}: {e}")

        try:
            dir_files = os.listdir(input_folder)
        except OSError as e:
            logger.error("Error listing folder %s: %s", input_folder, e)
            return ("", 0)

        matched_files = [os.path.join(input_folder, f) for f in dir_files if compiled.search(f)]
        return ("\n".join(matched_files), len(matched_files))


class Soze_DownloadURL:
    """Download a URL to disk and return the local filepath.

    `filepath` is interpreted relative to ComfyUI's output directory unless it is
    absolute. When `append_suffix` is True, an incrementing `_00001` suffix is
    appended (matching ComfyUI's save-image convention).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": False}),
                "filepath": ("STRING", {"default": "", "multiline": False, "tooltip": "Relative to ComfyUI output dir, or an absolute path."}),
            },
            "optional": {
                "append_suffix": ("BOOLEAN", {"default": True, "tooltip": "Append an incrementing _00001 counter to the filename (ComfyUI save-image convention)."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "download"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def download(self, url, filepath, append_suffix=True):
        # None-tolerance: if either input is missing, no-op rather than raise.
        if url is None or filepath is None or not str(url).strip() or not str(filepath).strip():
            logger.info("Download URL skipped: missing url or filepath.")
            return ("",)

        url = url.strip()
        filepath = filepath.strip()

        if os.path.isabs(filepath):
            target = os.path.normpath(filepath)
        else:
            target = os.path.normpath(os.path.join(folder_paths.get_output_directory(), filepath))

        parent_dir = os.path.dirname(target)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if append_suffix:
            base, ext = os.path.splitext(target)
            counter = 1
            while True:
                candidate = f"{base}_{counter:05d}{ext}"
                if not os.path.exists(candidate):
                    target = candidate
                    break
                counter += 1

        tmp_path = target + ".part"
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(target):
                os.remove(target)
            os.replace(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

        logger.info("Downloaded %s -> %s", url, target)
        return (target,)
