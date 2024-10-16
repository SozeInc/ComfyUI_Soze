import numpy as np
import re

class StringReplacer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True}),
                "replace_chars": ("STRING", {"default": '",-,;,:,/,\,|,<,>,?,{,},[,],(,),=,+,*,^,&,%,#,!,~,`'}),
                "replace_with": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)

    FUNCTION = "replace_characters"
    CATEGORY = "strings"

    def replace_characters(self, input_string, replace_with, replace_chars):
        # Split the replace_chars string into a list
        chars_to_replace = replace_chars.split(',')
        
        # Create a regular expression pattern
        pattern = '[' + re.escape(''.join(chars_to_replace)) + ']'
        
        # Perform the replacement
        result = re.sub(pattern, replace_with, input_string)
        
        return (result,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "StringReplacer": StringReplacer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringReplacer": "String Replacer"
}