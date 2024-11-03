# ComfyUI Mobile Nodes - A collection of ComfyUI Mobile related custom nodes.
# by Soze Inc - 2024-05 
# https://github.com/SozeInc/ComfyUI-Mobile
import os



# Get the absolute path of various directories
mobilenodes_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
custom_nodes_dir = os.path.abspath(os.path.join(mobilenodes_dir, '..'))
comfy_dir = os.path.abspath(os.path.join(mobilenodes_dir, '..', '..'))









########################################################################################################################
# Ultimate_Concat
class Soze_Ultimate_Concat:
    @classmethod
    def INPUT_TYPES(cls):
        concat_max = 25
        inputs = {
            "required": {
                "concat_count": ("INT", {"default": 1, "min": 1, "max": concat_max, "step": 1, "hideInput": True}),
                "delimiter": ("STRING", {"default": ""},),
            }
        }

        inputs["optional"] = {}

        for i in range(1, concat_max):
            inputs["required"][f"prefix_{i}"] = ("STRING", {"default": ""}, {"hideInput": True})
            inputs["optional"][f"value_{i}"] = ("*", {"default": ""}, {"forceInput": True}, {"hideInput": True})

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "concatenate_all"
    CATEGORY = "strings"

    def concatenate_all(self, concat_count, delimiter, **kwargs):
        # Handle special case where delimiter is "\n" (literal newline).
        if delimiter == "\\n":
            delimiter = "\n"

        merged: list[str] = []
        for i in range(1, concat_count + 1):
            if str(kwargs.get(f"prefix_{i}")).strip() != "":
                merged.append(str(kwargs.get(f"prefix_{i}")) + str(kwargs.get(f"value_{i}")) + str(delimiter))

        return (merged,)




