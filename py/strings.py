import numpy as np
import re

from ..utils import (
    read_from_file,
    write_to_file
)


class Soze_StringReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True,"multiline": True}),
                "replace_chars": ("STRING", {"default": '",-,;,:,/,\,|,<,>,?,{,},[,],(,),=,+,*,^,&,%,#,!,~,`,.', "multiline": True}),
                "replace_with": ("STRING", {"default": ""}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1})
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "replace_characters"
    CATEGORY = "strings"

    def replace_characters(self, input_string, replace_with, replace_chars, max_length):
        # Split the replace_chars string into a list
        chars_to_replace = replace_chars.split(',')
        
        # Create a regular expression pattern
        pattern = '[' + re.escape(''.join(chars_to_replace)) + ']'
        
        # Perform the replacement
        result = re.sub(pattern, replace_with, input_string)

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
            


class Soze_ChooseFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"default": "", "forceInput": True}),
                "list": ("STRING", {"default": "", "multiline": True}),
                "unmatched_value": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_NAMES = ("output")
    RETURN_TYPES = ("STRING")
    FUNCTION = "choose_from_list"
    CATEGORY = "strings"

    def choose_from_list(self, input, list, unmatched_value):
        # Convert the list to an array
        array = list.splitlines()
        
        # Check for a case-insensitive match
        for item in array:
            if input.lower() == item.lower():
                return item
        
        # Return unmatched_value if no match is found
        return unmatched_value