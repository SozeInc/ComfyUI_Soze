# ComfyUI Mobile Nodes - A collection of ComfyUI quality of life related custom nodes.
# by Soze Inc - 2024-09
# https://github.com/SozeInc/ComfyUI-Soze


import os
import subprocess
import importlib.util
import shutil
import sys
import traceback

NODE_CLASS_MAPPINGS = { "Output Filename": Output_Filename
                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)"
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']