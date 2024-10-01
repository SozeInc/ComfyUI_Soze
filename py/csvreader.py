import csv
import os
import random

class CSVReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_filename": ("STRING", {"default": "data.csv"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1})
            }
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Row_Count')
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "INT")
    FUNCTION = "read_csv"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(s, csv_filename, seed):
        return float("nan")
    
    def strip_quotes(self, value):
        return value.strip('"')

    def read_csv(self, csv_filename, seed):
        csv_path = os.path.join(os.path.dirname(__file__), "csv_files", csv_filename)
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                rows = list(csv_reader)
                row_count = len(rows)
                
                if seed >= row_count:
                    return tuple([""] * 10 + [row_count])
                
                row = rows[seed]
                # Strip quotes from each value in the row
                stripped_row = [self.strip_quotes(value) for value in row]
                output = stripped_row[:10] + [""] * (10 - len(stripped_row))
                return tuple(output + [row_count])
        except FileNotFoundError:
            raise  # Re-raise the FileNotFoundError
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return tuple([""] * 10 + [0])