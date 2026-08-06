import json
import folder_paths
import os
import re
from pathlib import Path
import random
import ast
from typing import Union, Tuple
import time



from server import PromptServer
from aiohttp import web
# Build full paths and load images
from PIL import Image, ImageOps

from .status_utils import push_node_status
import torch
import numpy as np


import html

JSON_OUT_PATH = os.path.join(folder_paths.output_directory, "json")
Path(JSON_OUT_PATH).mkdir(parents=True, exist_ok=True)


def _parse_json_loose(text):
    """Parse JSON, tolerating a few common real-world malformations:

      - ```json ... ``` markdown fences
      - bare comma-separated top-level objects (a Postgres/SQL row dump):
            {..},{..}            ->  [{..},{..}]
      - JSON Lines (one object per line, no separating commas):
            {..}\\n{..}           ->  [{..},{..}]
      - Python-style dicts (single quotes / None / True) via ast.literal_eval

    Returns the parsed object. Raises json.JSONDecodeError if nothing works
    (so existing `except json.JSONDecodeError` handlers keep working — it is a
    subclass of ValueError).
    """
    if text is None:
        raise json.JSONDecodeError("Empty input", "", 0)

    s = text.strip()

    # Strip a leading/trailing markdown code fence.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*", "", s).strip()
        if s.endswith("```"):
            s = s[:-3].strip()

    if s == "":
        raise json.JSONDecodeError("Empty input", "", 0)

    # 1) Straight parse.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 2) Wrap bare comma-separated top-level values in an array:  {..},{..}
    if not s.startswith("["):
        try:
            return json.loads("[" + s + "]")
        except json.JSONDecodeError:
            pass

    # 3) JSON Lines — join non-empty lines with commas, then wrap.
    lines = [ln.strip().rstrip(",") for ln in s.splitlines() if ln.strip()]
    if len(lines) > 1:
        try:
            return json.loads("[" + ",".join(lines) + "]")
        except json.JSONDecodeError:
            pass

    # 4) Python literal fallback (single quotes / None / True / tuples).
    try:
        return ast.literal_eval(s)
    except Exception:
        raise json.JSONDecodeError("Invalid JSON or Python literal", s, 0)

@PromptServer.instance.routes.get("/toolbox/json/{filename}")
async def toolbox_json(request):
    filename = request.match_info["filename"]
    file_path = None
    if "temp" in filename:
        file_path = folder_paths.get_temp_directory()
    else:
        file_path = JSON_OUT_PATH

    content = ""
    with open(Path(file_path) / f"{filename}", "r") as f:
        content = f.read()
    content = html.escape(content)
    content = "<br />".join(content.split("\n"))
    return web.json_response({"content": content})




class Soze_FormatJson:
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("formatted_json", "minified_json",)
    FUNCTION = "format_json"
    CATEGORY = "utils"
    def format_json(self, json_string: str) -> str:
        try:
            # Parse, tolerating fences / row-dumps / JSONL / Python literals.
            data = _parse_json_loose(json_string)

            formatted_json = json.dumps(data, indent=2)
            minified_json = json.dumps(data, separators=(',', ':'))
            return (formatted_json, minified_json,)
        except Exception as e:
            raise ValueError(f"Invalid JSON string: {str(e)}")
    





class Soze_ParseValueFromJSONString(Soze_FormatJson):
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"default": ""}),
            },
            "optional": {
                "string_value_word_limit": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Limit the number of words in the returned string value. 0 means no limit."}),
                "default_value": ("STRING", {"default": "", "tooltip": "Default value to return if key is not found."}),
                "use_default_on_error": ("BOOLEAN", {"default": False, "tooltip": "If true, return the default value when the key is not found instead of an error message."}),
            },
        }
    RETURN_NAMES = ("string_value", "int_value", "float_value", "bool_value", "any_value" , "formatted_json_text",)
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN", "STRING", "STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = False

    def is_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def run(self, key, string_value_word_limit, json_string: str, default_value="", use_default_on_error=False) -> tuple:
        def get_default_tuple():
            # Parse default_value into different types
            string_value = default_value
            try:
                int_value = int(default_value)
            except ValueError:
                int_value = 0
            try:
                float_value = float(default_value)
            except ValueError:
                float_value = 0.0
            bool_value = default_value.lower() in ('true', '1', 'yes')
            any_value = default_value
            formatted_json_text = default_value
            return (string_value, int_value, float_value, bool_value, any_value, formatted_json_text)

        try:
            # Prepare a truncated preview of the input JSON for use in error returns
            input_preview = ''
            if string_value_word_limit > 0:
                words = json_string.split()
                if len(words) > string_value_word_limit:
                    input_preview = ' '.join(words[:string_value_word_limit])
                else:
                    input_preview = json_string

            try:
                data = _parse_json_loose(json_string)
            except json.JSONDecodeError:
                if use_default_on_error:
                    return get_default_tuple()
                else:
                    raise ValueError("Invalid JSON string\n\n" + json_string)

            minified_json = json.dumps(data, separators=(',', ':'))
            value = data
            # Unwrap arrays by taking the first item
            if isinstance(value, list) and len(value) > 0:
                value = value[0]

            # Normalize bracket notation into dotted segments so callers can
            # use any of: foo.0.bar  /  foo[0].bar  /  foo["bar"].baz
            normalized_key = key or ""
            # ["key"] or ['key'] -> .key
            normalized_key = re.sub(r"""\[\s*["']([^"']+)["']\s*\]""", r".\1", normalized_key)
            # [123] -> .123
            normalized_key = re.sub(r"\[\s*(\d+)\s*\]", r".\1", normalized_key)
            # Strip leading dot if the key began with a bracket (e.g. [0].foo)
            normalized_key = normalized_key.lstrip(".")

            for k in normalized_key.split('.'):
                k = k.strip()
                if k == "":
                    continue
                # If the current value is a string that looks like embedded
                # JSON (common when APIs/Postgres store JSON in a text column),
                # try to parse it so the next segment can descend into it.
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith(("[", "{")):
                        try:
                            value = json.loads(stripped)
                        except json.JSONDecodeError:
                            pass  # leave as string; lookup will fail naturally
                if isinstance(value, dict) and k in value:
                    value = value[k]
                elif isinstance(value, list) and k.isdigit():
                    idx = int(k)
                    if 0 <= idx < len(value):
                        value = value[idx]
                    else:
                        if use_default_on_error:
                            return get_default_tuple()
                        else:
                            raise ValueError(f"Index '{k}' out of range in JSON\n\n" + json_string)
                else:
                    if use_default_on_error:
                        return get_default_tuple()
                    else:
                        raise ValueError(f"Key '{k}' not found in JSON\n\n" + json_string)

            # Truncate the extracted value (not the whole JSON) when requested
            value_str = json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else str(value)
            if string_value_word_limit > 0:
                v_words = value_str.split()
                if len(v_words) > string_value_word_limit:
                    string_value = ' '.join(v_words[:string_value_word_limit])
                else:
                    string_value = value_str
            else:
                string_value = value_str

            # Integer detection: prefer exact ints, allow negative integers in strings
            int_value = 0
            if isinstance(value, int):
                int_value = value
            elif isinstance(value, float):
                int_value = int(value)
            elif isinstance(value, str) and value.lstrip('-').isdigit():
                try:
                    int_value = int(value)
                except Exception:
                    int_value = 0

            # Float detection
            float_value = 0.0
            if isinstance(value, (int, float)):
                float_value = float(value)
            elif isinstance(value, str) and self.is_float(value):
                try:
                    float_value = float(value)
                except Exception:
                    float_value = 0.0

            bool_value = value if isinstance(value, bool) else False
            formatted_json_text = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)

            return {"ui": {"Value: ": formatted_json_text}, "result": (string_value, int_value, float_value, bool_value, str(value), formatted_json_text,)}
        except json.JSONDecodeError:
            if use_default_on_error:
                return get_default_tuple()
            else:
                raise ValueError("Invalid JSON string\n\n" + json_string)
        except KeyError as e:
            if use_default_on_error:
                return get_default_tuple()
            else:
                raise ValueError(str(e) + "\n\n" + json_string)
        except Exception as e:
            if use_default_on_error:
                return get_default_tuple()
            else:
                raise ValueError(f"Error parsing value: {str(e)}\n\n" + json_string)

class Soze_JSONGetArrayCount:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"multiline": True}),
                "json_path": ("STRING", {"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "STRING", )
    RETURN_NAMES = ("array_count", "json_item",)
    FUNCTION = "calculate_array_count"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float:
        mode = kwargs.get("mode", "fixed")
        if mode != "fixed":
            return time.time()
        return float("NaN")

    def calculate_array_count(self, json_input: str, json_path: str, unique_id=None) -> Tuple[str, str]:
        try:
            # Parse, tolerating fences / row-dumps / JSONL / Python literals.
            data = _parse_json_loose(json_input)
            # Navigate to the specified path if provided
            if json_path.strip() != "":
                for key in json_path.split('.'):
                    if key.isdigit():
                        if isinstance(data, list):
                            data = data[int(key)]
                        else:
                            data = data[key]
                    elif '[' in key and ']' in key:
                        list_key, idx = key[:-1].split('[')
                        data = data[list_key][int(idx)]
                    else:
                        data = data[key]

            if not isinstance(data, list):
                push_node_status(unique_id, f"ERROR: path '{json_path}' did not lead to an array.")
                raise ValueError("The specified path does not lead to a JSON array")

            array_count = len(data)
            push_node_status(unique_id, f"OK: array at '{json_path or '(root)'}' has {array_count} item(s).")
            return (array_count, json.dumps(data, indent=2) if isinstance(data, (dict, list)) else str(data),)
        except json.JSONDecodeError as e:
            push_node_status(unique_id, f"ERROR: invalid JSON: {e}")
            raise ValueError("Invalid JSON input or Python literal (expected an array) Error: " + str(e))


        
                
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
            "hidden": {"unique_id": "UNIQUE_ID"},
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

    def iterate_json_array(self, json_input: str, index: int, unique_id=None) -> Tuple[str, int, int]:
        try:
            # Parse, tolerating fences / row-dumps / JSONL / Python literals.
            data = _parse_json_loose(json_input)
            if not isinstance(data, list):
                push_node_status(unique_id, "ERROR: input is not a JSON array.")
                raise ValueError("Input must be a JSON array")

            total_items = len(data)
            if total_items == 0:
                push_node_status(unique_id, "Empty array (0 items).")
                return ("", 0, 0)

            # Ensure index is within bounds (do not loop)
            if index < 0 or index >= total_items:
                push_node_status(unique_id, f"ERROR: index {index} out of range (total={total_items}).")
                raise IndexError(f"Index {index} out of range for array of length {total_items}")

            item = data[index]

            if isinstance(item, (dict, list)):
                item = json.dumps(item)
            else:
                item = str(item)

            preview = item[:80] + ("…" if len(item) > 80 else "")
            push_node_status(unique_id, f"OK: item {index+1}/{total_items} — {preview}")
            return (item, index, total_items)
        except json.JSONDecodeError as e:
            push_node_status(unique_id, f"ERROR: invalid JSON: {e}")
            raise ValueError("Invalid JSON input or Python literal (expected an array) Error: " + str(e))
        

class Soze_JSONPathExtractorNode:
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
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("parsed_data1", "parsed_data2", "parsed_data3", "parsed_data4", "parsed_data5", "parsed_data6", "parsed_data7", "parsed_data8", "parsed_data9", "parsed_data10")
    FUNCTION = "parse_json_values"
    CATEGORY = "utils"

    def parse_json_values(self, json_string: str, path1: str, path2: str, path3: str, path4: str, path5: str, path6: str, path7: str, path8: str, path9: str, path10: str, unique_id=None) -> tuple:
        try:
            if (json_string.strip().startswith("```json")):
                json_string = json_string.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_string)

            paths = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10]
            results = []
            hits = 0
            misses = 0

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
                    hits += 1
                except Exception:
                    results.append("")
                    misses += 1
            push_node_status(unique_id, f"OK: {hits} hit(s), {misses} miss(es)")
            return {"ui": {"Value: ": results}, "result": (results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9])}
        except json.JSONDecodeError:
            push_node_status(unique_id, "ERROR: invalid JSON string.")
            raise ValueError("Invalid JSON string")
        except Exception:
            push_node_status(unique_id, "ERROR: invalid path or key not found.")
            raise ValueError("Invalid path or key not found")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


def _navigate_json_key_path(data, path):
    """Walk a dotted/bracketed path into JSON-like data.

    Examples:
      "Story"                  -> data["Story"]
      "Story.Name"             -> data["Story"]["Name"]
      "items[0].id"            -> data["items"][0]["id"]
      "0.title"                -> data[0]["title"]   (bare digit at root)
      'data["users"][2].name'  -> data["users"][2]["name"]

    Tolerances (matching the JSON Value Parser node):
      - bracket notation is normalized to dotted segments
      - if the ROOT is a list and the first segment is a non-index key, the
        list is auto-unwrapped to its first element — so keys resolve against
        a single-/multi-row array dump (row 0)
      - a string value that itself contains embedded JSON (e.g. a stringified
        column like reference_image_urls) is decoded before descending
    """
    # Normalize bracket notation -> dotted segments.
    p = path or ""
    p = re.sub(r"""\[\s*["']([^"']+)["']\s*\]""", r".\1", p)  # ["k"] / ['k'] -> .k
    p = re.sub(r"\[\s*(\d+)\s*\]", r".\1", p)                   # [0] -> .0
    segments = [seg.strip() for seg in p.lstrip(".").split('.') if seg.strip() != ""]

    value = data

    # Auto-unwrap a top-level list when the first segment is a non-index key.
    if segments and isinstance(value, list) and not segments[0].isdigit():
        if not value:
            raise KeyError("empty array at root")
        value = value[0]

    for segment in segments:
        # Decode embedded JSON strings on the way down.
        if isinstance(value, str):
            st = value.strip()
            if st[:1] in ("[", "{"):
                try:
                    value = json.loads(st)
                except json.JSONDecodeError:
                    pass
        if segment.isdigit() and isinstance(value, list):
            value = value[int(segment)]
        else:
            value = value[segment]
    return value


class Soze_JSONStringParser10X:
    """Read up to 10 string values out of a JSON document by key/path.

    Sibling of `JSON Path Extractor` but with the simpler conventions you asked
    for: 10 named key widgets, 10 named string outputs, no word limits, no
    per-slot defaults, no "error on default" — a missing or invalid key just
    yields an empty string for that slot and the node keeps going.

    Each `key_N` widget accepts either:
      * a bare top-level name      e.g. `Story`
      * a dotted path              e.g. `Story.Name`
      * indexed access             e.g. `items[0].id`  or  `0.title`

    Object / array values are JSON-stringified into the output slot;
    primitives are stringified as-is; `null` becomes an empty string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "json_string": ("STRING", {"multiline": True}),
        }
        for i in range(1, 11):
            required[f"key_{i}"] = ("STRING", {"default": ""})
        return {
            "required": required,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",) * 10
    RETURN_NAMES = tuple(f"value_{i}" for i in range(1, 11))
    FUNCTION = "parse"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def parse(self, json_string, unique_id=None, **keys):
        s = (json_string or "").strip()
        if not s:
            push_node_status(unique_id, "ERROR: empty input; all outputs empty.")
            return tuple([""] * 10)

        try:
            data = _parse_json_loose(s)
        except Exception:
            push_node_status(unique_id, "ERROR: invalid JSON; all outputs empty.")
            return tuple([""] * 10)

        results = []
        hits = 0
        for i in range(1, 11):
            key = str(keys.get(f"key_{i}", "")).strip()
            if not key or data is None:
                results.append("")
                continue
            try:
                value = _navigate_json_key_path(data, key)
            except Exception:
                results.append("")
                continue
            if value is None:
                results.append("")
            elif isinstance(value, (dict, list)):
                results.append(json.dumps(value, indent=2))
            else:
                results.append(str(value))
            if results[-1]:
                hits += 1

        push_node_status(unique_id, f"OK: extracted {hits}/10 key(s).")
        return tuple(results)


class Soze_JSONFileLoader:
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_filepath": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Json_Contents", "Formatted_Json_Contents", "Minified_Json_Contents", "Filename_Path", "Filename", "Filename_No_Ext")
    FUNCTION = "load_json_file"
    CATEGORY = "utils"

    def load_json_file(self, json_filepath: str) -> str:
        try:
            with open(json_filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data = _parse_json_loose(content)
                formatted_json = json.dumps(data, indent=2)
                minified_json = json.dumps(data, separators=(',', ':'))
            return (content, formatted_json, minified_json, json_filepath, os.path.basename(json_filepath), os.path.splitext(os.path.basename(json_filepath))[0])
        except FileNotFoundError:
            raise ValueError(f"File not found: {json_filepath}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")


class Soze_LoadJSONFileFromFolder():
    @classmethod
    def IS_CHANGED(self):
        return time.time()      

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_folderpath": ("STRING", {"multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Json_Contents","Formatted_Json_Contents", "Minified_Json_Contents", "Filename_Path", "Filename", "Filename_No_Ext",)
    FUNCTION = "load_json_file"
    CATEGORY = "utils"

    def load_json_file(self, json_folderpath: str, index: int) -> str:
        try:
            dir_files = os.listdir(json_folderpath)
            if len(dir_files) == 0:
                raise FileNotFoundError(f"Folder only has {len(dir_files)} files in it: {json_folderpath}")

            # Filter files by extension
            valid_extensions = ['.json']
            dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

            dir_files = sorted(dir_files)
            dir_files = [os.path.join(json_folderpath, x) for x in dir_files]

            # Check if index is valid
            if index >= len(dir_files):
                raise IndexError(f"Index {index} is out of range. Only {len(dir_files)} JSON files found.")

            # Load only the single JSON file at the specified index
            json_filepath = dir_files[index]
            if os.path.isdir(json_filepath):
                raise ValueError(f"Path at index {index} is a directory, not a JSON file.") 
            with open(json_filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                data = _parse_json_loose(content)
                formatted_json = json.dumps(data, indent=2)
                minified_json = json.dumps(data, separators=(',', ':'))
            return content, formatted_json, minified_json, json_filepath, os.path.basename(json_filepath), os.path.splitext(os.path.basename(json_filepath))[0]
        except FileNotFoundError:
            raise ValueError(f"Folder not found: {json_folderpath}")    
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
        
        



class Soze_CreateImageBatchFromJSONArray:
    @classmethod
    def IS_CHANGED(self):
        return time.time()    

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
                if (json_array.strip().startswith("```json")):
                    json_array = json_array.replace("```json", "").replace("```", "").strip()
                data = json.loads(json_array)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_array)
                except Exception:
                    raise ValueError("Invalid JSON input or Python literal (expected an array of filenames)")

            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array or Python list")

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
    def IS_CHANGED(self):
        return time.time()    

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
                if (json_array.strip().startswith("```json")):
                    json_array = json_array.replace("```json", "").replace("```", "").strip()
                data = json.loads(json_array)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_array)
                except Exception:
                    raise ValueError("Invalid JSON input or Python literal (expected an array of filenames)")

            if not isinstance(data, list):
                raise ValueError("Input must be a JSON array or Python list")

            # Build full paths and load images
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



