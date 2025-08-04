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
from .py.images import Soze_LoadImage, Soze_LoadImagesFromFolder, Soze_BatchProcessSwitch, Soze_LoadImageFromUrl, Soze_GetImageColors,  Soze_PadMask
from .py.csvreader import Soze_CSVReader, Soze_CSVReaderXCheckpoint, Soze_CSVReaderXLora
from .py.csvwriter import Soze_CSVWriter
from .py.xy import Soze_UnzippedProductAny
#from .py.promptxlora import Soze_PromptXLora
from .py.lorafileloader import Soze_LoraFilePathLoader
from .py.ckptfileloader import Soze_CheckpointFilePathLoader
from .py.comfydeploy import Soze_ComfyDeployAPINode
from .py.elevenlabs import Soze_ElevenLabsVoiceRetrieverNode

from .py.strings import (
    Soze_SpecialCharacterReplacer,
    Soze_MultilineConcatenateStrings,
    Soze_PromptCache,
    Soze_TextContains,
    Soze_TextContainsReturnString,
    Soze_IsStringEmpty
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
                        "CSV Reader X Checkpoint": Soze_CSVReaderXCheckpoint,
                        "CSV Writer": Soze_CSVWriter,
                        "Special Character Replacer": Soze_SpecialCharacterReplacer,
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
                        "Lora File Loader": Soze_LoraFilePathLoader,
                        "Alpha Crop and Position Image": Soze_AlphaCropAndPositionImage,
                        "Shrink Image": Soze_ShrinkImage,
                        "Pad Mask": Soze_PadMask,
                        "Checkpoint File Loader": Soze_CheckpointFilePathLoader,
                        "CSV Reader X Lora": Soze_CSVReaderXLora,
                        "ComfyDeploy API Node": Soze_ComfyDeployAPINode,
                        "Is String Empty": Soze_IsStringEmpty,
                        "ElevenLabs Voice Retriever": Soze_ElevenLabsVoiceRetrieverNode,
                        # "Is Input In List": Soze_IsInputInList,
                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)",
                                "Load Image": "Load Image (Soze)",
                                "Load Images From Folder": "Load Images From Folder (Soze)",
                                "Image Batch Process Switch": "Image Batch Process Switch (Soze)",
                                "Load Image From URL": "Load Image From URL (Soze)",
                                "CSV Reader": "CSV Reader (Soze)",
                                "CSV Writer": "CSV Writer (Soze)",
                                "Special Character Replacer": "Special Character Replacer (Soze)",                               
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
                                "Pad Mask": "Pad Mask (Soze)",
                                "CSV Reader X Checkpoint": "CSV Reader X Checkpoint (Soze)",
                                "Checkpoint File Loader": "Checkpoint File Loader (Soze)",
                                "CSV Reader X Lora": "CSV Reader X Lora (Soze)",
                                "ComfyDeploy API Node": "ComfyDeploy API Node (Soze)",
                                "Is String Empty": "Is String Empty (Soze)",
                                "ElevenLabs Voice Retriever": "ElevenLabs Voice Retriever (Soze)",
                                # "Is Input In List": "Is Input In List (Soze)"
                                
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']