import os
import numpy as np
import torch
import folder_paths
import time    


class Soze_PromptFileFromFolderXLora:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_folder": ("STRING", {"default": ""}),
                "start_lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the starting LoRA."}),
                "lora_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "The number of LoRAs to load."})
            },
            "optional": {
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("Prompt", "Input_Path",  "Prompt_Filename_Path", "Prompt_Filename", "Prompt_Filename_No_Ext", "Lora_Full_Path", "Lora_Name_Only", "Lora_Index")
    FUNCTION = "load_prompt_from_folder"

    CATEGORY = "Soze Nodes"

    def load_prompt_from_folder(self, input_folder, index, start_lora_name, lora_count):
        if not os.path.isdir(input_folder):
            raise FileNotFoundError(f"Folder not found: {input_folder}")
        dir_files = os.listdir(input_folder)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {input_folder}")

        # Filter files by extension
        valid_extensions = ['.txt']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(input_folder , x) for x in dir_files]

        # Calculate which prompt and lora to use
        num_prompts = len(dir_files)
        if num_prompts == 0:
            raise FileNotFoundError(f"No valid prompts found in folder: {input_folder}")

        prompt_idx = index % num_prompts
        lora_list = folder_paths.get_filename_list("loras")
        try:
            start_lora_index = lora_list.index(start_lora_name)
        except ValueError:
            raise ValueError(f"Lora '{start_lora_name}' not found in lora list.")

        lora_index = start_lora_index + (index // num_prompts)
        if lora_index >= len(lora_list):
            raise ValueError(f"There are no more lora in the list ({len(lora_list)})")
        elif lora_index >= (start_lora_index + lora_count):
            raise ValueError(f"Index {index} has completed the iteration of rows {num_prompts} against each lora indicated {lora_count}.")

        prompt_path = dir_files[prompt_idx]
        if os.path.isdir(prompt_path):
            raise ValueError(f"Prompt path is a directory: {prompt_path}")

        input_filepath = folder_paths.get_annotated_filepath(prompt_path)
        input_filename = os.path.basename(input_filepath)
        input_filename_no_ext = os.path.splitext(input_filename)[0]

        retries = 3
        for attempt in range(retries):
            try:
                with open(input_filepath, 'r', encoding='utf-8') as file:
                    output_prompt = file.read()
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5)
                else:
                    raise e

        lora_full_path = folder_paths.get_full_path_or_raise("loras", lora_list[lora_index])
        lora_name_only = os.path.basename(lora_list[lora_index])

        return (
            output_prompt,
            input_folder,
            input_filepath,
            input_filename,
            input_filename_no_ext,
            lora_full_path,
            lora_name_only,
            lora_index
        )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Return a value that changes each time to force re-execution
        return float("NaN")
    

