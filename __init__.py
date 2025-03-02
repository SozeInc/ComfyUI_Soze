# ComfyUI Mobile Nodes - A collection of ComfyUI quality of life related custom nodes.
# by Soze Inc - 2024-09
# https://github.com/SozeInc/ComfyUI-Soze


import os
import subprocess
import importlib.util
import shutil
import sys
import traceback

from .py.outputfilename import Soze_OutputFilename
from .py.images import Soze_LoadImage
from .py.images import Soze_LoadImagesFromFolder
from .py.images import Soze_BatchProcessSwitch
from .py.images import Soze_LoadImageFromUrl
from .py.images import Soze_GetImageColors
from .py.csvreader import Soze_CSVReader
from .py.csvwriter import Soze_CSVWriter
from .py.xy import Soze_UnzippedProductAny
from .py.promptxlora import Soze_PromptXLora
from .py.lorafileloader import Soze_LoraFileLoader

from .py.strings import (
    Soze_StringReplacer,
    Soze_MultilineConcatenateStrings,
    Soze_PromptCache,
    Soze_TextContains,
    Soze_TextContainsReturnString
    # Soze_IsInputInList
    )

from .py.range_nodes import (
    Soze_IntRangeNode,
    Soze_FloatRangeNode,
    Soze_IntNumStepsRangeNode,
    Soze_FloatNumStepsRangeNode)


from .py.images import (
    Soze_ImageLabelOverlay,
    Soze_EmptyImages,
    Soze_XYImage,
    Soze_ImageListLoader,
    Soze_VariableImageBuilder,
    Soze_AlphaCropAndPositionImage,
    Soze_ShrinkImage
)

NODE_CLASS_MAPPINGS = { "Output Filename": Soze_OutputFilename,
                        "Load Image": Soze_LoadImage,
                        "Load Images From Folder": Soze_LoadImagesFromFolder,
                        "Image Batch Process Switch": Soze_BatchProcessSwitch,
                        "Load Image From URL": Soze_LoadImageFromUrl,
                        "CSV Reader": Soze_CSVReader,
                        "CSV Writer": Soze_CSVWriter,
                        "String Replacer": Soze_StringReplacer,
                        "Multiline Concatenate Strings": Soze_MultilineConcatenateStrings,
                        "Range(Step) - Int": Soze_IntRangeNode,
                        "Range(Num Steps) - Int": Soze_IntNumStepsRangeNode,
                        "Range(Step) - Float": Soze_FloatRangeNode,
                        "Range(Num Steps) - Float": Soze_FloatNumStepsRangeNode,
                        "XY Any": Soze_UnzippedProductAny,
                        "XY Image": Soze_XYImage,
                        "Image Overlay": Soze_ImageLabelOverlay,
                        "Empty Images": Soze_EmptyImages,
                        "Image List Loader": Soze_ImageListLoader,
                        "Variable Image Builder": Soze_VariableImageBuilder,
                        "Prompt Cache": Soze_PromptCache,
                        "Text Contains (Return Bool)": Soze_TextContains,
                        "Text Contains (Return String)": Soze_TextContainsReturnString,
                        "Get Most Common Image Colors": Soze_GetImageColors,
                        "Prompt X Lora": Soze_PromptXLora,
                        "Lora File Loader": Soze_LoraFileLoader,
                        "Alpha Crop and Position Image": Soze_AlphaCropAndPositionImage,
                        "Shrink Image": Soze_ShrinkImage,
                        # "Is Input In List": Soze_IsInputInList,
                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)",
                                "Load Image": "Load Image (Soze)",
                                "Load Images From Folder": "Load Images From Folder (Soze)",
                                "Image Batch Process Switch": "Image Batch Process Switch (Soze)",
                                "Load Image From URL": "Load Image From URL (Soze)",
                                "CSV Reader": "CSV Reader (Soze)",
                                "CSV Writer": "CSV Writer (Soze)",
                                "String Replacer": "String Replacer (Soze)",
                                "Multiline Concatenate Strings": "Multiline Concatenate (Soze)",
                                "Range(Step) - Int": "Int Step Range (Soze)",
                                "Range(Num Steps) - Int": "Int Step Count Range (Soze)",
                                "Range(Step) - Float": "Float Step Range (Soze)",
                                "Range(Num Steps) - Float": "Float Step Count Range (Soze)",
                                "XY Any": "XY Any (Soze)",
                                "XY Image": "XY Image (Soze)",
                                "Image Overlay": "Image Overlay (Soze)",
                                "Empty Images": "Empty Images (Soze)",
                                "Image List Loader": "Image List Loader (Soze)",
                                "Variable Image Builder": "Variable Image Builder (Soze)",
                                "Prompt Cache": "Prompt Cache (Soze)",
                                "Text Contains (Return Bool)": "Is Text In Text (Soze)",
                                "Text Contains (Return String)": "List Contains Text Return String (Soze)",
                                "Get Most Common Image Colors": "Get Most Common Image Colors (Soze)",
                                "Prompt X Lora": "Prompt X Lora (Soze)",
                                "Lora File Loader": "Lora File Loader (Soze)",
                                "Alpha Crop and Position Image": "Alpha Crop and Position Image (Soze)",
                                "Shrink Image": "Shrink Image (Soze)",
                                # "Is Input In List": "Is Input In List (Soze)"
                                
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']