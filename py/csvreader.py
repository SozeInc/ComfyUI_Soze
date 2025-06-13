import time
import csv
import os
import random
import folder_paths
import comfy.utils

from comfy import model_management

class Soze_CSVReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "","multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),

            },
            "optional": {
                "csv_text": ("STRING", {"default": "","multiline": True})
            }
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Entire_Line', 'Row_Count')
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    FUNCTION = "read_csv"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self):
        return time.time()    
    
    def strip_quotes(self, value):
        return value.strip('"')

    def read_csv(self, csv_filename_path, csv_text, index):
        if csv_text.strip() != '':
            csv_data = csv_text.splitlines()
        else:
            if csv_filename_path.strip() != "":
                csv_path = os.path.join(os.path.dirname(__file__), "csv_files", csv_filename_path)
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")
                try:
                    with open(csv_path, 'r', newline='') as csvfile:
                        csv_data = csvfile.readlines()
                except FileNotFoundError:
                    raise  # Re-raise the FileNotFoundError
                except Exception as e:
                    print(f"Error reading CSV: {str(e)}")
                    return tuple([""] * 10 + [0])
            else:
                return tuple([""] * 10 + [0])

        csv_reader = csv.reader(csv_data)
        rows = list(csv_reader)
        row_count = len(rows)
        
        if index >= row_count:
            raise ValueError(f"There are no more rows in the CSV file ({row_count})")
        
        row = rows[index]
        # Strip quotes from each value in the row
        stripped_row = [self.strip_quotes(value) for value in row]
        output = stripped_row[:10] + [""] * (10 - len(stripped_row))
        entire_line = ",".join(stripped_row)
        return tuple(output + [entire_line, row_count])
    



class Soze_CSVReaderXCheckpoint:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "","multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1, "tooltip": "The row number to read from the CSV file."}),
                "start_ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the starting LoRA."}),
                "ckpt_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "The number of LoRAs to load."})
            }
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Entire_Line', 'Row_Count', "Ckpt_Full_Path", "Ckpt_Name_Only", "Cktp_Index")
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT", "STRING", "STRING", "INT")
    FUNCTION = "process"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, csv_filename_path, index, start_ckpt_name, ckpt_count):
        return time.time()    
    
    def strip_quotes(self, value):
        return value.strip('"')

    def process(self, csv_filename_path, index, start_ckpt_name, ckpt_count):
        ckpt_list = folder_paths.get_filename_list("checkpoints")
        
        if csv_filename_path.strip() == "":
            raise ValueError("CSV filename path cannot be empty.")

        if csv_filename_path.strip() != "":
            csv_path = os.path.join(os.path.dirname(__file__), "csv_files", csv_filename_path.strip())

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            try:
                with open(csv_path, 'r', newline='') as csvfile:
                    csv_data = csvfile.readlines()
            except FileNotFoundError:
                raise  # Re-raise the FileNotFoundError
            except Exception as e:
                print(f"Error reading CSV: {str(e)}")
                raise ValueError(f"Error reading CSV file: {str(e)}")
                

        csv_reader = csv.reader(csv_data)
        rows = list(csv_reader)
        row_count = len(rows)
        
        try:
            start_ckpt_index = ckpt_list.index(start_ckpt_name)
        except ValueError:
            raise ValueError(f"Checkpoint '{start_ckpt_name}' not found in checkpoint list.")
        
        csv_index = index % row_count
        ckpt_index = start_ckpt_index + (index // row_count)

        if ckpt_index >= len(ckpt_list):
            raise ValueError(f"There are no more checkpoints in the list ({len(ckpt_list)})")
        elif ckpt_index >= (start_ckpt_index + ckpt_count):
            raise ValueError(f"Index {index} has completed the iteration of rows {row_count} against each checkpoint indicated {ckpt_count}.")
        
        row = rows[csv_index]
        # Strip quotes from each value in the row
        stripped_row = [self.strip_quotes(value) for value in row]
        output = stripped_row[:10] + [""] * (10 - len(stripped_row))
        entire_line = ",".join(stripped_row)

        ckpt_full_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_list[ckpt_index])
        
        ckpt_name_only = os.path.basename(ckpt_list[ckpt_index])

        return tuple(output + [entire_line, row_count] + [ckpt_full_path, ckpt_name_only, ckpt_index + 1])

