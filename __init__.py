# ComfyUI Mobile Nodes - A collection of ComfyUI quality of life related custom nodes.
# by Soze Inc - 2024-09
# https://github.com/SozeInc/ComfyUI-Soze


import os
import subprocess
import importlib.util
import shutil
import sys
import traceback

from .py.outputfilename import OutputFilename
from .py.images import LoadImage
from .py.images import LoadImagesFromFolder
from .py.images import BatchProcessSwitch

NODE_CLASS_MAPPINGS = { "Output Filename": OutputFilename,
                       "Load Image": LoadImage,
                          "Load Images From Folder": LoadImagesFromFolder,
                            "Image Batch Process Switch": BatchProcessSwitch,
                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)",
                                "Load Image": "Load Image (Soze)",
                                "Load Images From Folder": "Load Images From Folder (Soze)",
                                "Image Batch Process Switch": "Image Batch Process Switch (Soze)",
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']