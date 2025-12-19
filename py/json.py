import json


class Soze_ParseValueFromJSONString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = False

    def run(self, json_string, key):
        try:
            # Strip whitespace and handle literal \n
            json_string = json_string.strip()
            json_string = json_string.replace('\\n', '\n')
            
            # Remove any single quotes wrapping the string
            if json_string.startswith("'") and json_string.endswith("'"):
                json_string = json_string[1:-1]
            
            # Check if JSON is properly wrapped in braces
            if not json_string.startswith('{'):
                json_string = '{' + json_string
            if not json_string.endswith('}'):
                json_string = json_string + '}'
            
            # Clean up any malformed URLs at the end
            if '"https:}' in json_string:
                json_string = json_string.replace('"https:}', '"https://"}')
            
            print(f"Attempting to parse JSON: {json_string[:100]}...")  # Debug first 100 chars
            
            # Try to parse the JSON
            data = json.loads(json_string)
            value = data.get(key)
            
            if value is None:
                return ("", None, None, None)
            
            # Return based on type
            if isinstance(value, bool):
                return ("", None, None, value)
            elif isinstance(value, int):
                return ("", value, None, None)
            elif isinstance(value, float):
                return ("", None, value, None)
            else:
                return (str(value), None, None, None)
                
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {str(e)}")
            print(f"Attempted to parse: {json_string[:200]}...") # Print first 200 chars
            return ("", None, None, None)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return ("", None, None, None)

import json
import ast
from typing import Union, Tuple
import time


class Soze_JSONArrayIteratorNode:
    stored_index = 0
    last_update_time = 0
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1, "tooltip": "The row number to read from the CSV file."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("item", "current_index", "total_items")
    FUNCTION = "iterate_json_array"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        mode = kwargs.get("mode", "fixed")
        if mode != "fixed":
            return time.time()
        return float("NaN")

    def iterate_json_array(self, json_input: str, index: int) -> Tuple[str, int, int]:
        try:
            # Try JSON first (standard). If it fails, try ast.literal_eval
            try:
                data = json.loads(json_input)
            except json.JSONDecodeError:
                # Allow Python-style single-quoted lists/dicts by using ast.literal_eval
                try:
                    data = ast.literal_eval(json_input)
                except Exception:
                    raise json.JSONDecodeError("Invalid JSON or Python literal", json_input, 0)
            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array")

            total_items = len(data)
            if total_items == 0:
                return ("", 0, 0)

            # Ensure index is within bounds (do not loop)
            if index < 0 or index >= total_items:
                raise IndexError(f"Index {index} out of range for array of length {total_items}")

            item = data[index]
            
            if isinstance(item, (dict, list)):
                item = json.dumps(item)
            else:
                item = str(item)
                
            return (item, index, total_items)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON input or Python literal (expected an array)") 
        

class Soze_SimpleJSONParserNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True}),
                "path1": ("STRING", {"default": ""}),
                "path2": ("STRING", {"default": ""}),
                "path3": ("STRING", {"default": ""}),
                "path4": ("STRING", {"default": ""}),
                "path5": ("STRING", {"default": ""}),
                "path6": ("STRING", {"default": ""}),
                "path7": ("STRING", {"default": ""}),
                "path8": ("STRING", {"default": ""}),
                "path9": ("STRING", {"default": ""}),
                "path10": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("parsed_data1", "parsed_data2", "parsed_data3", "parsed_data4", "parsed_data5", "parsed_data6", "parsed_data7", "parsed_data8", "parsed_data9", "parsed_data10")
    FUNCTION = "parse_json_values"
    CATEGORY = "utils"

    def parse_json_values(self, json_string: str, path1: str, path2: str, path3: str, path4: str, path5: str, path6: str, path7: str, path8: str, path9: str, path10: str) -> tuple:
        try:
            data = json.loads(json_string)

            paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10]
            results = []

            for path in paths:
                if path.strip() == "":
                    results.append("")
                    continue
                try:
                    value = data
                    for key in path.split('.'):
                        if key.isdigit():
                            value = value[int(key)]
                        elif '[' in key and ']' in key:
                            list_key, idx = key[:-1].split('[')
                            value = value[list_key][int(idx)]
                        else:
                            value = value[key]
                    if isinstance(value, (dict, list)):
                        results.append(json.dumps(value, indent=2))
                    else:
                        results.append(str(value))
                except Exception:
                    results.append("")
            return tuple(results)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")
        except Exception:
            raise ValueError("Invalid path or key not found")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


class Soze_JSONFileLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_filepath": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_contents",)
    FUNCTION = "load_json_file"
    CATEGORY = "utils"

    def load_json_file(self, json_filepath: str) -> str:
        try:
            with open(json_filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            return (content,)
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_filepath}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")



class Soze_CreateImageBatchFromJSONArray:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("STRING", {"multiline": False}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "create_image_batch"
    CATEGORY = "utils"
    
    def create_image_batch(self, json_array: str, prefix: str, suffix: str):
        try:
            # Accept either JSON or Python literal lists
            try:
                data = json.loads(json_array)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_array)
                except Exception:
                    raise ValueError("Invalid JSON input or Python literal (expected an array of filenames)")

            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array or Python list")

            # Build full paths and load images
            from PIL import Image, ImageOps
            import torch
            import numpy as np
            import os

            images = []
            base_size = None

            for item in data:
                if not isinstance(item, str):
                    raise ValueError("All items in the array must be string filenames or paths")

                image_path = f"{prefix}{item}{suffix}"
                if not os.path.isfile(image_path):
                    raise ValueError(f"Image file not found: {image_path}")

                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img).convert("RGB")

                if base_size is None:
                    base_size = img.size
                else:
                    # Resize other images to base size to allow stacking
                    if img.size != base_size:
                        img = img.resize(base_size, resample=Image.LANCZOS)

                arr = np.array(img).astype(np.float32) / 255.0
                # Keep HWC layout to match LoadImage (H, W, C)
                tensor = torch.from_numpy(arr)
                images.append(tensor.unsqueeze(0))  # (1, H, W, C)

            if len(images) == 0:
                raise ValueError("No images to create a batch from")

            batch = torch.cat(images, dim=0)  # (N, H, W, C)
            return (batch,)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error creating image batch: {str(e)}")



class Soze_LoadImagesFromJSONArray:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_array": ("STRING", {"multiline": False}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2", "image3", "image4", "remaining_images")
    FUNCTION = "load_images_from_json_array"
    CATEGORY = "utils"
    
    def load_images_from_json_array(self, json_array: str, prefix: str, suffix: str):
        try:
            # Accept either JSON or Python literal lists
            try:
                data = json.loads(json_array)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_array)
                except Exception:
                    raise ValueError("Invalid JSON input or Python literal (expected an array of filenames)")

            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array or Python list")

            # Build full paths and load images
            from PIL import Image, ImageOps
            import torch
            import numpy as np
            import os

            images = []
            base_size = None

            for item in data:
                if not isinstance(item, str):
                    raise ValueError("All items in the array must be string filenames or paths")

                image_path = f"{prefix}{item}{suffix}"
                if not os.path.isfile(image_path):
                    raise ValueError(f"Image file not found: {image_path}")

                img = Image.open(image_path)
                img = ImageOps.exif_transpose(img).convert("RGB")

                if base_size is None:
                    base_size = img.size
                else:
                    # Resize other images to base size to allow stacking
                    if img.size != base_size:
                        img = img.resize(base_size, resample=Image.LANCZOS)

                arr = np.array(img).astype(np.float32) / 255.0
                # Keep HWC layout to match LoadImage (H, W, C)
                tensor = torch.from_numpy(arr)
                images.append(tensor.unsqueeze(0))  # (1, H, W, C)

            if len(images) == 0:
                raise ValueError("No images to create a batch from")

            # All images are tensors shaped (1, H, W, C)
            # Use None for missing images to match expected behavior
            img1 = images[0] if len(images) >= 1 else None
            img2 = images[1] if len(images) >= 2 else None
            img3 = images[2] if len(images) >= 3 else None
            img4 = images[3] if len(images) >= 4 else None

            if len(images) > 4:
                remaining = torch.cat(images[4:], dim=0)
            else:
                remaining = None

            return (img1, img2, img3, img4, remaining)

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error creating image batch: {str(e)}")
