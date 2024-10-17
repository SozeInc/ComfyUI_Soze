import numpy as np
import re

class StringReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True,"multiline": True}),
                "replace_chars": ("STRING", {"default": '",-,;,:,/,\,|,<,>,?,{,},[,],(,),=,+,*,^,&,%,#,!,~,`,.,/n'}),
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

class MultilineConcatenateStrings:
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
