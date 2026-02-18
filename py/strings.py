import datetime
import uuid
import numpy as np
import os
import re
import time
import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy_api.latest import _io

from .utils import (
    read_from_file,
    write_to_file,
    replace_date_placeholders,
    _remove_non_ascii,
    _sanitize_tuple,
    _sanitize_filename
)


# class Soze_IsInputInList:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "input_value": ("STRING", {"default": "", "forceInput": True}),
#                 "unmatched_value": ("STRING", {"default": ""}),
#                 "csv_list": ("STRING", {"default": "", "multiline": True}),
#             }
#         }

#     RETURN_NAMES = ("result",)
#     RETURN_TYPES = ("STRING",)
#     FUNCTION = "choose_from_list"
#     CATEGORY = "strings"

#     def choose_from_list(self, input_value, csv_list, unmatched_value):
#         # Convert the list to an array
#         array = csv_list.split(',')
        
#         # Check for a case-insensitive match
#         for item in array:
#             if input_value.lower() == item.lower():
#                 return str(item.strip())
        
#         # Return unmatched_value if no match is found
#         return unmatched_value
    
class Soze_SpecialCharacterReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True,"multiline": True}),
                "replace_chars": ("STRING", {"default": '"-;:/\|<>?}{[]()=+*^&%#!~`._,\' @$', "multiline": True}),
                "replace_with": ("STRING", {"default": ""}),
                "strip_newlines": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "replace_characters"
    CATEGORY = "strings"

    def replace_characters(self, input_string, replace_with, replace_chars, strip_newlines, max_length):
        if input_string is None:
            return _sanitize_tuple(("",))
        result = input_string
        for char in replace_chars:
            result = result.replace(char, replace_with)
        if strip_newlines:
            result = result.replace('\n', '').replace('\r', '')
        if max_length > 0:
            return _sanitize_tuple((result[:max_length],))
        else:
            return _sanitize_tuple((result,))        

class Soze_MultilineConcatenateStrings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "string1": ("STRING", {"multiline": True}),
            },
            "optional": {
                "string2": ("STRING", {"multiline": True}),
                "string3": ("STRING", {"multiline": True}),
                "string4": ("STRING", {"multiline": True}),
                "string5": ("STRING", {"multiline": True}),
                "string6": ("STRING", {"multiline": True}),
                "string7": ("STRING", {"multiline": True}),
                "string8": ("STRING", {"multiline": True}),
                "string9": ("STRING", {"multiline": True}),
                "string10": ("STRING", {"multiline": True}),
                "separator": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concatenate"
    CATEGORY = "strings"

    def concatenate(self, string1, string2="", string3="", string4="", string5="", string6="", string7="", string8="", string9="", string10="", separator=""):
        strings = [string1, string2, string3, string4, string5, string6, string7, string8, string9, string10]
        # Filter out empty strings
        non_empty_strings = [s for s in strings if s]
        # Join the non-empty strings with the separator
        result = separator.join(non_empty_strings)
        return _sanitize_tuple((result,))


class Soze_PromptCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_new_prompt": ("BOOL", {"default": True}),
                "new_prompt": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_NAMES = ("output_prompt", "is_new_prompt")
    RETURN_TYPES = ("STRING", "BOOL")
    FUNCTION = "prompt_with_cache"
    CATEGORY = "strings"

    def prompt_with_cache(self, new_prompt, use_new_prompt):
        if use_new_prompt:
            write_to_file('sozepromptcache.txt', new_prompt)
            return _sanitize_tuple((new_prompt, True))
        else:
            old_prompt = read_from_file('sozepromptcache.txt')
            write_to_file('sozepromptcache.txt', new_prompt)
            if old_prompt:
                return _sanitize_tuple((old_prompt, False))
            else:
                return _sanitize_tuple((new_prompt, True))
            


class Soze_TextContainsReturnString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "find_text": ("STRING", {"default": '', "multiline": False}),
                "piped_text_list": ("STRING", {"default": '', "multiline": False}),
                "found_return_value": ("STRING", {"default": '', "multiline": False}),
                "not_found_return_value": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "case_sensitive": ("BOOLEAN", {"default": False}),
                "partial_match": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_contains"

    CATEGORY = "strings"

    def text_contains(self, find_text, piped_text_list, case_sensitive,partial_match,found_return_value,not_found_return_value):
        # Split the sub_texts by the pipe character
        text_list = piped_text_list.split('|')
        found = False
        original_find_text = find_text

        if case_sensitive == False:
            find_text = find_text.lower()
            text_list = [entry.lower() for entry in text_list]

        for entry in text_list:
            if (partial_match):
                if find_text in entry:
                    found = True
            else:
                if find_text == entry:
                    found = True
        if found:
            if (found_return_value != ''):
                return _sanitize_tuple((found_return_value,))
            else:
                return _sanitize_tuple((original_find_text,))
        
        return _sanitize_tuple((not_found_return_value,))

class Soze_TextContains:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "find_text": ("STRING", {"default": '', "multiline": False}),
                "search_text": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "case_sensitive": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "text_contains"

    CATEGORY = "strings"

    def text_contains(self, find_text, search_text, case_sensitive):
        if case_sensitive == False:
            search_text = search_text.lower()
            find_text = find_text.lower()

        return (find_text in search_text,)



class Soze_IsStringEmpty:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {"default": '', "multiline": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "text_empty"

    CATEGORY = "strings"

    def text_empty(self, input_text):
        if input_text is None:
            return (True,)
        return (input_text.strip() == "",)
    


class Soze_MultiFindAndReplace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": False}),
            },
            "optional": {
                "find_text1": ("STRING", {"multiline": False}),
                "replace_text1": ("STRING", {"multiline": False}),
                "find_text2": ("STRING", {"multiline": False}),
                "replace_text2": ("STRING", {"multiline": False}),
                "find_text3": ("STRING", {"multiline": False}),
                "replace_text3": ("STRING", {"multiline": False}),
                "find_text4": ("STRING", {"multiline": False}),
                "replace_text4": ("STRING", {"multiline": False}),
                "find_text5": ("STRING", {"multiline": False}),
                "replace_text5": ("STRING", {"multiline": False}),
                "find_text6": ("STRING", {"multiline": False}),
                "replace_text6": ("STRING", {"multiline": False}),
                "find_text7": ("STRING", {"multiline": False}),
                "replace_text7": ("STRING", {"multiline": False}),
                "find_text8": ("STRING", {"multiline": False}),
                "replace_text8": ("STRING", {"multiline": False}),
                "find_text9": ("STRING", {"multiline": False}),
                "replace_text9": ("STRING", {"multiline": False}),
                "find_text10": ("STRING", {"multiline": False}),
                "replace_text10": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "findandreplace"
    CATEGORY = "strings"

    def findandreplace(self, input_string, find_text1="", replace_text1="", find_text2="", replace_text2="", find_text3="", replace_text3="", find_text4="", replace_text4="", find_text5="", replace_text5="", find_text6="", replace_text6="", find_text7="", replace_text7="", find_text8="", replace_text8="", find_text9="", replace_text9="", find_text10="", replace_text10=""):
        result = input_string
        find_replace_pairs = [
            (find_text1, replace_text1),
            (find_text2, replace_text2),
            (find_text3, replace_text3),
            (find_text4, replace_text4),
            (find_text5, replace_text5),
            (find_text6, replace_text6),
            (find_text7, replace_text7),
            (find_text8, replace_text8),
            (find_text9, replace_text9),
            (find_text10, replace_text10),
        ]

        for find_text, replace_text in find_replace_pairs:
            if find_text:
                result = result.replace(find_text, replace_text)

        return _sanitize_tuple((result,))


#Node that performs various string functions
class Soze_StringFunctions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": False, "forceInput": True}),
            },
            "optional": {
                "string_function_type1": ([
                        "NONE", 
                        "UPPERCASE",
                        "LOWERCASE",
                        "TITLECASE",
                        "REVERSE",
                        "TRIM_SPACES",
                        "REMOVE_DIGITS",
                        "REMOVE_PUNCTUATION",
                        "REPLACE_SPACES_WITH_UNDERSCORE",
                        "REPLACE_UNDERSCORES_WITH_SPACES",
                        "REPLACE_HYPHENS_WITH_SPACES",
                        "REPLACE_SPACES_WITH_HYPHENS",
                        "REMOVE_NUMBERS_IN_ROUND_BRACKETS",
                        "SPLIT_LINES",
                    ], {                    
                        "default": "NONE"
                }),
                "string_function_type2": ([
                        "NONE", 
                        "UPPERCASE",
                        "LOWERCASE",
                        "TITLECASE",
                        "REVERSE",
                        "TRIM_SPACES",
                        "REMOVE_DIGITS",
                        "REMOVE_PUNCTUATION",
                        "REPLACE_SPACES_WITH_UNDERSCORE",
                        "REPLACE_UNDERSCORES_WITH_SPACES",
                         "REPLACE_HYPHENS_WITH_SPACES",
                        "REPLACE_SPACES_WITH_HYPHENS",
                       "REMOVE_NUMBERS_IN_ROUND_BRACKETS",
                        "SPLIT_LINES",
                    ], {                    
                        "default": "NONE"
                }),
                "string_function_type3": ([
                        "NONE", 
                        "UPPERCASE",
                        "LOWERCASE",
                        "TITLECASE",
                        "REVERSE",
                        "TRIM_SPACES",
                        "REMOVE_DIGITS",
                        "REMOVE_PUNCTUATION",
                        "REPLACE_SPACES_WITH_UNDERSCORE",
                        "REPLACE_UNDERSCORES_WITH_SPACES",
                        "REPLACE_HYPHENS_WITH_SPACES",
                        "REPLACE_SPACES_WITH_HYPHENS",
                        "REMOVE_NUMBERS_IN_ROUND_BRACKETS",
                        "SPLIT_LINES",
                    ], {                    
                        "default": "NONE"
                }),
                "string_function_type4": ([
                        "NONE", 
                        "UPPERCASE",
                        "LOWERCASE",
                        "TITLECASE",
                        "REVERSE",
                        "TRIM_SPACES",
                        "REMOVE_DIGITS",
                        "REMOVE_PUNCTUATION",
                        "REPLACE_SPACES_WITH_UNDERSCORE",
                        "REPLACE_UNDERSCORES_WITH_SPACES",
                        "REPLACE_HYPHENS_WITH_SPACES",
                        "REPLACE_SPACES_WITH_HYPHENS",
                        "REMOVE_NUMBERS_IN_ROUND_BRACKETS",
                        "SPLIT_LINES",
                    ], {                    
                        "default": "NONE"
                }),
                "string_function_type5": ([
                        "NONE", 
                        "UPPERCASE",
                        "LOWERCASE",
                        "TITLECASE",
                        "REVERSE",
                        "TRIM_SPACES",
                        "REMOVE_DIGITS",
                        "REMOVE_PUNCTUATION",
                        "REPLACE_SPACES_WITH_UNDERSCORE",
                        "REPLACE_UNDERSCORES_WITH_SPACES",
                        "REPLACE_HYPHENS_WITH_SPACES",
                        "REPLACE_SPACES_WITH_HYPHENS",
                        "REMOVE_NUMBERS_IN_ROUND_BRACKETS",
                        "SPLIT_LINES",
                    ], {                    
                        "default": "NONE"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "executefunction"
    CATEGORY = "strings"

    def process_string_function(self, input_string, function_type):
        if function_type == "UPPERCASE":
            result = input_string.upper()
        elif function_type == "LOWERCASE":
            result = input_string.lower()
        elif function_type == "TITLECASE":
            result = input_string.title()
        elif function_type == "REVERSE":
            result = input_string[::-1]
        elif function_type == "TRIM_SPACES":
            result = input_string.strip()
        elif function_type == "REMOVE_DIGITS":
            result = re.sub(r'\d+', '', input_string)
        elif function_type == "REMOVE_PUNCTUATION":
            result = re.sub(r'[^\w\s]', '', input_string)
        elif function_type == "REPLACE_SPACES_WITH_UNDERSCORE":
            result = input_string.replace(' ', '_')
        elif function_type == "REPLACE_UNDERSCORES_WITH_SPACES":
            result = input_string.replace('_', ' ')
        elif function_type == "REPLACE_HYPHENS_WITH_SPACES":
            result = input_string.replace('-', ' ')
        elif function_type == "REPLACE_SPACES_WITH_HYPHENS":
            result = input_string.replace(' ', '-')
        elif function_type == "REMOVE_NUMBERS_IN_ROUND_BRACKETS":
            result = re.sub(r'\(\d+\)', '', input_string)
        elif function_type == "SPLIT_LINES":
            result = '|'.join(input_string.splitlines())
        else:
            result = input_string
        return result

    def executefunction(self, input_string, string_function_type1="NONE", string_function_type2="NONE", string_function_type3="NONE", string_function_type4="NONE", string_function_type5="NONE"):
        result = input_string
        function_types = [string_function_type1, string_function_type2, string_function_type3, string_function_type4, string_function_type5]
        for function_type in function_types:
            result = self.process_string_function(result, function_type)
        return _sanitize_tuple((result,))




class Soze_AppendToTextFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_path": ("STRING", {"default": "","multiline": False}),
                "append_text": ("STRING", {"default": "", 'forceInput': True }),
            }
        }
    RETURN_NAMES = ()
    RETURN_TYPES = ()
    FUNCTION = "append_to_file"
    CATEGORY = "utils"
    OUTPUT_NODE = True
    @classmethod
    def IS_CHANGED(self, filename_path, append_text):
        return time.time()  
    
    def append_to_file(self, filename_path, append_text):
        if filename_path.strip() != "":
            # Get the ComfyUI output directory
            output_dir = folder_paths.get_output_directory()
            
            # Normalize path separators for OS compatibility
            filename_path = os.path.normpath(filename_path)
            
            # Combine output directory with the provided filename path
            full_path = os.path.join(output_dir, filename_path)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # Append the text to the file, creating it if it doesn't exist
            retries = 3
            for attempt in range(retries):
                try:
                    with open(full_path, 'a', encoding='utf-8') as file:
                        file.write(append_text + '\n')
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                    else:
                        raise e
        return ()


class Soze_LoadTextFromFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_path": ("STRING", {"default": "","multiline": False}),
                "prepend_text_line": ("STRING", {"default": "","multiline": True}),
                "append_text_line": ("STRING", {"default": "","multiline": True}),
            }
        }
    RETURN_NAMES = ("text_output",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_text_from_file"
    CATEGORY = "soze"
    OUTPUT_NODE = True
    @classmethod
    def IS_CHANGED(self, filename_path, prepend_text_line, append_text_line):
        return time.time()
    
    def load_text_from_file(self, filename_path, prepend_text_line, append_text_line):
        loaded_text = ""
        if filename_path.strip() != "":
            # Normalize path separators for OS compatibility
            filename_path = os.path.normpath(filename_path)
            # Read the text from the file
            retries = 3
            for attempt in range(retries):
                try:
                    with open(filename_path, 'r', encoding='utf-8') as file:
                        loaded_text = file.read()
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                    else:
                        raise e
        # Prepend and append text if provided
        if prepend_text_line:
            loaded_text = prepend_text_line + '\n' + loaded_text
        if append_text_line:
            loaded_text = loaded_text + '\n' + append_text_line
        return _sanitize_tuple((loaded_text,))
    


class Soze_LoadRandomLineFromTextFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename_path": ("STRING", {"default": "","multiline": False}),
                "prepend_text_line": ("STRING", {"default": "","multiline": True}),
                "append_text_line": ("STRING", {"default": "","multiline": True}),
            }
        }
    RETURN_NAMES = ("text_output",)
    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_random_line_from_text_file"
    CATEGORY = "soze"
    OUTPUT_NODE = False
    @classmethod
    def IS_CHANGED(self, filename_path, prepend_text_line, append_text_line):
        return time.time()
    
    def load_random_line_from_text_file(self, filename_path, prepend_text_line, append_text_line):
        import random
        loaded_text = ""
        if filename_path.strip() != "":
            # Normalize path separators for OS compatibility
            filename_path = os.path.normpath(filename_path)
            # Read the text from the file
            retries = 3
            for attempt in range(retries):
                try:
                    with open(filename_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                    if lines:
                        loaded_text = random.choice(lines).strip()
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                    else:
                        raise e
        # Prepend and append text if provided
        if prepend_text_line:
            loaded_text = prepend_text_line + '\n' + loaded_text
        if append_text_line:
            loaded_text = loaded_text + '\n' + append_text_line
        return _sanitize_tuple((loaded_text,))
    

class Soze_StringSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": "","forceInput": True }),
                "split_string": ("STRING", {"default": "","multiline": False}),
            }
        }
    RETURN_NAMES = ("string1", "string2",)
    RETURN_TYPES = ("STRING", "STRING", )
    FUNCTION = "split_string"
    CATEGORY = "soze"
    OUTPUT_NODE = False
    @classmethod
    def IS_CHANGED(self, input_string, split_string):
        return time.time()  
    def split_string(self, input_string, split_string):
        parts = input_string.split(split_string, 1)
        string1 = parts[0]
        string2 = parts[1] if len(parts) > 1 else ""
        return _sanitize_tuple((string1, string2,))


class Soze_EmptyStringReplacement:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": "","forceInput": True }),
                "empty_string_replacement": ("STRING", {"default": "","multiline": False}),
            }
        }
    RETURN_NAMES = ("output_string",)
    RETURN_TYPES = ("STRING", )
    FUNCTION = "empty_string_replacement"
    CATEGORY = "soze"
    OUTPUT_NODE = False
    @classmethod
    def IS_CHANGED(self, input_string, empty_string_replacement):
        return time.time()  
    def empty_string_replacement(self, input_string, empty_string_replacement):
        if input_string.strip() == "":
            empty_string_replacement = replace_date_placeholders(empty_string_replacement)
            return _sanitize_tuple((empty_string_replacement,))
        else:
            return _sanitize_tuple((input_string,))
            


class Soze_SaveTextFileToOutput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"default": "","forceInput": True }),
                "filename_path": ("STRING", {"default": "","multiline": False}),
                "file_extension": ("STRING", {"default": "txt","multiline": False}),
                "overwrite_file": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_NAMES = ()
    RETURN_TYPES = ()
    FUNCTION = "load_text_from_file"
    CATEGORY = "soze"
    OUTPUT_NODE = True
    @classmethod
    def IS_CHANGED(self, input_string, file_extension, filename_path, overwrite_file):
        return time.time()
    
    def load_text_from_file(self, input_string, file_extension, filename_path, overwrite_file):
        if input_string is None:
            return ()
        if filename_path.strip() != "":
            # Get the ComfyUI output directory
            output_dir = folder_paths.get_output_directory()
            # Normalize path separators for OS compatibility
            filename_path = os.path.normpath(filename_path) 
            
            # Add extension if missing
            if file_extension and not filename_path.lower().endswith(f".{file_extension.lower()}"):
                filename_path = f"{filename_path}.{file_extension}"

            # Combine output directory with the provided filename path
            full_path = os.path.join(output_dir, filename_path)
            # Ensure the directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            if not overwrite_file and os.path.exists(full_path):
                base, ext = os.path.splitext(full_path)
                counter = 1
                while True:
                    new_path = f"{base}_{counter:05d}{ext}"
                    if not os.path.exists(new_path):
                        full_path = new_path
                        break
                    counter += 1

            # Save the text to the file
            retries = 3 
            for attempt in range(retries):
                try:
                    with open(full_path, 'w', encoding='utf-8') as file:
                        file.write(input_string)    
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(0.1)
                    else:
                        raise e 
        return ()
            





class Soze_OutputFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'Path_Delimiter': ('STRING', {'default': '/'}),
                             'File_Delimiter': ('STRING', {'default': '-'}),
                             },
                'optional': {'Path_Input_1': ('STRING', {'default': '.'}),
                             'Path_Input_2': ('STRING', {'default': 'ProjectName'}),
                             'Path_Input_3': ('STRING', {'default': '%date:yyyy-MM-dd%'}),
                             'Path_Input_4': ('STRING', {'default': 'GenerationType'}),
                             'Path_Input_5': ('STRING', {'default': ''}),
                             'File_Input_1': ('STRING', {'default': ''}),
                             'File_Input_2': ('STRING', {'default': ''}),
                             'File_Input_3': ('STRING', {'default': ''}),
                             'File_Input_4': ('STRING', {'default': ''}),
                             'File_Input_5': ('STRING', {'default': ''})
                             }}

    RETURN_NAMES = ('Filename_&_Path','Path','Filename')
    RETURN_TYPES = ('STRING','STRING','STRING')
    FUNCTION = 'OutputFilename'
    CATEGORY = 'strings'

    def OutputFilename(self, Path_Delimiter, Path_Input_1, Path_Input_2, Path_Input_3, Path_Input_4, Path_Input_5, File_Delimiter, File_Input_1, File_Input_2, File_Input_3, File_Input_4, File_Input_5):
        # Collect and process path inputs in a loop for cleaner code
        path_inputs = [Path_Input_1, Path_Input_2, Path_Input_3, Path_Input_4, Path_Input_5]
        paths = [replace_date_placeholders(p) for p in path_inputs if p]
        path = Path_Delimiter.join(paths)

        file_inputs = [File_Input_1, File_Input_2, File_Input_3, File_Input_4, File_Input_5]
        if not any(f.strip() for f in file_inputs):
            files = replace_date_placeholders(str(uuid.uuid4()))
        else:
            files_list = [replace_date_placeholders(f) for f in file_inputs if f]
            files = File_Delimiter.join(files_list)
        
        # Sanitize the filename part
        files = _sanitize_filename(files)

        filepath = Path_Delimiter.join([path, files])
        return _sanitize_tuple((filepath, path, files,))




