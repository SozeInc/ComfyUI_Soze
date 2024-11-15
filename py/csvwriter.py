import time
import csv
import os

class Soze_CSVWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "","multiline": True}),
                "value1": ("STRING", {"default": "", 'isInput': True }),
                "value2": ("STRING", {"default": "", 'isInput': True }),
            },
            "optional": {
                "value3": ("STRING", {"default": ""}),
                "value4": ("STRING", {"default": ""}),
                "value5": ("STRING", {"default": ""}),
                "value6": ("STRING", {"default": ""}),
                "value7": ("STRING", {"default": ""}),
                "value8": ("STRING", {"default": ""}),
                "value9": ("STRING", {"default": ""}),
                "value10": ("STRING", {"default": ""}),
                
            }
        }

    RETURN_NAMES = ()
    RETURN_TYPES = ()
    FUNCTION = "write_csv"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(self, csv_filename_path, value1="", value2="", value3="", value4="", value5="", value6="", value7="", value8="", value9="", value10=""):
        return time.time()
    
    def wrap_quotes(self, value):
        return f'"{value}"'

    def write_csv(self, csv_filename_path, value1="", value2="", value3="", value4="", value5="", value6="", value7="", value8="", value9="", value10=""):
        values = [value1, value2, value3, value4, value5, value6, value7, value8, value9, value10]

        # Filter out empty values
        non_empty_values = [value for value in values if value]

        # Wrap each non-empty value in double quotes
        wrapped_values = [self.wrap_quotes(value) for value in non_empty_values]
        
        # Create a comma-separated line
        csv_line = ','.join(wrapped_values) + '\n'
        
        if csv_filename_path.strip() != "":
            # Ensure the directory exists
            os.makedirs(os.path.dirname(csv_filename_path), exist_ok=True)
            
            # Append the line to the CSV file, creating it if it doesn't exist
            try:
                with open(csv_filename_path, 'a', newline='') as csvfile:
                    csvfile.write(csv_line)
                return csv_line
            except Exception as e:
                raise FileNotFoundError(f"CSV file not found: {csv_filename_path}")
                return ""
                
        else:
            return ""