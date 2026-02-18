import os
import time
import folder_paths

from .utils import (
    read_from_file,
    write_to_file
)





class Soze_LoadFilesFromFolder:
    @classmethod
    def INPUT_TYPES(s):
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

    def load_files(self, input_folder, input_file_extensions, file_load_count=1, index=0):
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Folder not found: {input_folder}")
            
        dir_files = os.listdir(input_folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"Folder is empty: {input_folder}")

        # Filter files by provided extensions
        valid_extensions = [ext.strip().lower() for ext in input_file_extensions.split(',')]
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
        if not dir_files:
            raise FileNotFoundError(f"No files found with extensions: {input_file_extensions}")

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(input_folder, x) for x in dir_files]

        # Start at index
        dir_files = dir_files[index:]

        file_paths = []
        limit_files = file_load_count > 0
        file_count = 0

        for file_path in dir_files:
            if limit_files and file_count >= file_load_count:
                break
            if os.path.isfile(file_path):
                file_paths.append(file_path)
                file_count += 1

        if len(file_paths) == 0:
            raise FileNotFoundError(f"No valid files found in: {input_folder}")

        current_file = file_paths[0]
        previous_file = read_from_file('sozefilebatchcache.txt')
        write_to_file('sozefilebatchcache.txt', current_file)

        return (current_file, len(file_paths), input_folder, os.path.basename(current_file), os.path.splitext(os.path.basename(current_file))[0])

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")


class Soze_FileLoader:
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", ),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = False

    def read(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            return (content,)
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return ("",)



class Soze_DoesFileExist:
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", ),
            }
        }
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "does_exist"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = False

    def does_exist(self, filepath):
        exists = os.path.isfile(filepath) and os.path.exists(filepath)
        return (exists,)
    
    
class Soze_LoadFilesWithPattern:
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "filename_pattern": ("STRING", {"default": ".*", "description": "Regex pattern to match filenames"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("File_Path", "Load_Count")
    FUNCTION = "load_file_with_pattern"
    CATEGORY = "Soze Nodes"

    def load_file_with_pattern(self, input_folder, filename_pattern):
        import re

        if not os.path.isdir(input_folder):
            return ([], 0)

        try:
            dir_files = os.listdir(input_folder)
            matched_files = [os.path.join(input_folder, f) for f in dir_files if re.search(filename_pattern, f)]
        except Exception:
            return ([], 0)

        if not matched_files:
            return ([], 0)

        return (matched_files, len(matched_files), )