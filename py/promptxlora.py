import os
import time

import folder_paths

from .status_utils import EventLog, push_node_status, finalize_status


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
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("Prompt", "Input_Path",  "Prompt_Filename_Path", "Prompt_Filename", "Prompt_Filename_No_Ext", "Lora_Full_Path", "Lora_Name_Only", "Lora_Index", "status")
    FUNCTION = "load_prompt_from_folder"

    CATEGORY = "Soze Nodes"

    def load_prompt_from_folder(self, input_folder, index, start_lora_name, lora_count, unique_id=None):
        log = EventLog()
        push_node_status(unique_id, f"Folder: {input_folder}", log)
        if not os.path.isdir(input_folder):
            push_node_status(unique_id, f"ERROR: folder not found: {input_folder}", log)
            raise FileNotFoundError(f"Folder not found: {input_folder}")
        dir_files = os.listdir(input_folder)
        if len(dir_files) == 0:
            push_node_status(unique_id, f"ERROR: folder is empty: {input_folder}", log)
            raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {input_folder}")

        # Filter files by extension
        valid_extensions = ['.txt']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)
        dir_files = [os.path.join(input_folder , x) for x in dir_files]

        # Calculate which prompt and lora to use
        num_prompts = len(dir_files)
        if num_prompts == 0:
            push_node_status(unique_id, "ERROR: no .txt prompts in folder.", log)
            raise FileNotFoundError(f"No valid prompts found in folder: {input_folder}")
        push_node_status(unique_id, f"Found {num_prompts} prompt file(s).", log)

        prompt_idx = index % num_prompts
        lora_list = folder_paths.get_filename_list("loras")
        try:
            start_lora_index = lora_list.index(start_lora_name)
        except ValueError:
            push_node_status(unique_id, f"ERROR: lora '{start_lora_name}' not found.", log)
            raise ValueError(f"Lora '{start_lora_name}' not found in lora list.")

        lora_index = start_lora_index + (index // num_prompts)
        if lora_index >= len(lora_list):
            push_node_status(unique_id, f"ERROR: lora_index {lora_index} >= lora_list size {len(lora_list)}", log)
            raise ValueError(f"There are no more lora in the list ({len(lora_list)})")
        elif lora_index >= (start_lora_index + lora_count):
            push_node_status(unique_id, f"ERROR: iteration complete after {num_prompts} rows × {lora_count} loras.", log)
            raise ValueError(f"Index {index} has completed the iteration of rows {num_prompts} against each lora indicated {lora_count}.")

        prompt_path = dir_files[prompt_idx]
        if os.path.isdir(prompt_path):
            push_node_status(unique_id, f"ERROR: prompt path is a directory: {prompt_path}", log)
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
                    push_node_status(unique_id, f"ERROR reading prompt: {e!r}", log)
                    raise e

        lora_full_path = folder_paths.get_full_path_or_raise("loras", lora_list[lora_index])
        lora_name_only = os.path.basename(lora_list[lora_index])

        headline = (
            f"OK: prompt {prompt_idx+1}/{num_prompts} ({input_filename}), "
            f"lora {lora_index+1}/{len(lora_list)} ({lora_name_only})"
        )
        push_node_status(unique_id, headline, log)
        return (
            output_prompt,
            input_folder,
            input_filepath,
            input_filename,
            input_filename_no_ext,
            lora_full_path,
            lora_name_only,
            lora_index,
            finalize_status(headline, log),
        )

    @classmethod
    def IS_CHANGED(cls, input_folder, start_lora_name, lora_count, index=0, unique_id=None):
        try:
            entries = sorted(
                f for f in os.listdir(input_folder) if f.lower().endswith('.txt')
            )
        except OSError as e:
            return f"err:{input_folder}:{e}"
        parts = []
        for name in entries:
            try:
                st = os.stat(os.path.join(input_folder, name))
                parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{name}:?")
        return f"{input_folder}|{'|'.join(parts)}|{start_lora_name}|{lora_count}|{index}"


