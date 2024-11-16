import numpy as np
import re

from .utils import (
    read_from_file,
    write_to_file
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
    
class Soze_StringReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True,"multiline": True}),
                "replace_chars": ("STRING", {"default": '",-,;,:,/,\,|,<,>,?,{,},[,],(,),=,+,*,^,&,%,#,!,~,`,.', "multiline": True}),
                "replace_with": ("STRING", {"default": ""}),
                "strip_newlines": ("BOOLEAN", {"default": True}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "replace_characters"
    CATEGORY = "strings"

    def replace_characters(self, input_string, replace_with, replace_chars, strip_newlines, max_length):
        if replace_chars.strip() != '':
            # Split the replace_chars string into a list
            chars_to_replace = replace_chars.split(',')
            
            # Create a regular expression pattern
            pattern = '[' + re.escape(''.join(chars_to_replace)) + ']'
            
            # Perform the replacement
            result = re.sub(pattern, replace_with, input_string)
        else:
            result = input_string

        # Strip newlines if requested
        if strip_newlines:
            result = result.replace('\n', '').replace('\r', '')

        if max_length > 0:
            return (result[:max_length],)
        else:
            return (result,)        

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
                "separator": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "concatenate"
    CATEGORY = "strings"

    def concatenate(self, string1, string2="", string3="", string4="", string5="", separator=""):
        strings = [string1, string2, string3, string4, string5]
        # Filter out empty strings
        non_empty_strings = [s for s in strings if s]
        # Join the non-empty strings with the separator
        result = separator.join(non_empty_strings)
        return (result,)


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
            return (new_prompt, True)
        else:
            old_prompt = read_from_file('sozepromptcache.txt')
            write_to_file('sozepromptcache.txt', new_prompt)
            if old_prompt:
                return (old_prompt, False)
            else:
                return (new_prompt, True)
            


class Soze_TextContainsReturnString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False}),
                "piped_sub_text": ("STRING", {"default": '', "multiline": False}),
                "founnd_return_value": ("STRING", {"default": '', "multiline": False}),
                "not_found_return_value": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "case_insensitive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "text_contains"

    CATEGORY = "strings"

    def text_contains(self, text, piped_sub_text, case_insensitive,founnd_return_value,not_found_return_value):
        # Split the sub_texts by the pipe character
        sub_text_list = piped_sub_text.split('|')
        
        if case_insensitive:
            text = text.lower()
            sub_text_list = [sub_text.lower() for sub_text in sub_text_list]

        # Check if any subtext is in the text
        for sub_text in sub_text_list:
            if sub_text in text:
                return (founnd_return_value,)
        
        return (not_found_return_value,)

class Soze_TextContains:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": '', "multiline": False}),
                "sub_text": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "case_insensitive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "text_contains"

    CATEGORY = "strings"

    def text_contains(self, text, sub_text, case_insensitive):
        if case_insensitive:
            sub_text = sub_text.lower()
            text = text.lower()

        return (sub_text in text,)
