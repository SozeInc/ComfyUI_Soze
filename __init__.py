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
from .py.images import LoadImageFromUrl
from .py.csvreader import CSVReader
from .py.csvwriter import CSVWriter
from .py.strings import StringReplacer


NODE_CLASS_MAPPINGS = { "Output Filename": OutputFilename,
                       "Load Image": LoadImage,
                          "Load Images From Folder": LoadImagesFromFolder,
                            "Image Batch Process Switch": BatchProcessSwitch,
                            "Load Image From URL": LoadImageFromUrl,
                            "CSV Reader": CSVReader,
                            "CSV Writer": CSVWriter,
                          "String Replacer": StringReplacer,

                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)",
                                "Load Image": "Load Image (Soze)",
                                "Load Images From Folder": "Load Images From Folder (Soze)",
                                "Image Batch Process Switch": "Image Batch Process Switch (Soze)",
                                "Load Image From URL": "Load Image From URL (Soze)",
                                "CSV Reader": "CSV Reader (Soze)",
                                "CSV Writer": "CSV Writer (Soze)",
                                "String Replacer": "String Replacer (Soze)",
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']