import time
import csv
import os
import random

class Soze_CSVReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "","multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1})
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
    def IS_CHANGED(self, csv_filename_path, csv_text, seed):
        return time.time()    
    
    def strip_quotes(self, value):
        return value.strip('"')

    def read_csv(self, csv_filename_path, csv_text, seed):
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
        
        if seed >= row_count:
            raise ValueError(f"There are no more rows in the CSV file ({row_count})")
        
        row = rows[seed]
        # Strip quotes from each value in the row
        stripped_row = [self.strip_quotes(value) for value in row]
        output = stripped_row[:10] + [""] * (10 - len(stripped_row))
        entire_line = ",".join(stripped_row)
        return tuple(output + [entire_line, row_count])