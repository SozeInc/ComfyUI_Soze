# ComfyUI Mobile Nodes - A collection of ComfyUI quality of life related custom nodes.
# by Soze Inc - 2024-09
# https://github.com/SozeInc/ComfyUI-Soze


import os
import subprocess
import importlib.util
import shutil
import sys
import traceback

from .py.csvreader import Soze_CSVReader, Soze_CSVReaderXCheckpoint, Soze_CSVReaderXLora
from .py.csvwriter import Soze_CSVWriter
from .py.xy import Soze_UnzippedProductAny
#from .py.promptxlora import Soze_PromptXLora
from .py.lorafileloader import Soze_LoraFilePathLoader
from .py.ckptfileloader import Soze_CheckpointFilePathLoader
from .py.comfydeploy import (
    Soze_ComfyDeployAPINode, 
    Soze_ComfyDeployAPIIntParameters, 
    Soze_ComfyDeployAPIBooleanParameters, 
    Soze_ComfyDeployAPIImageParameters, 
    Soze_ComfyDeployAPIMixedParameters, 
    Soze_ComfyDeployAPIFloatParameters, 
    Soze_ComfyDeployAPIStringParameters, 
    Soze_ComfyDeployAPIMixedParametersV2, 
    Soze_ComfyDeployDownloadAPIFiles,
    Soze_ComfyDeployCacheAPIRunIDs,
    Soze_ComfyDeployRetrieveCachedAPIRunIDs,
    Soze_ComfyDeployCachedAPIRunInfo,
    Soze_ComfyDeployClearCachedAPIRunIDs
    )
from .py.elevenlabs import Soze_ElevenLabsVoiceRetrieverNode

from .py.strings import (
    Soze_SpecialCharacterReplacer,
    Soze_MultilineConcatenateStrings,
    Soze_PromptCache,
    Soze_TextContains,
    Soze_TextContainsReturnString,
    Soze_IsStringEmpty,
    Soze_MultiFindAndReplace,
    Soze_StringFunctions,
    Soze_AppendToTextFile,
    Soze_LoadTextFromFile,
    Soze_LoadRandomLineFromTextFile,
    Soze_StringSplitter,
    Soze_EmptyStringReplacement,
    Soze_SaveTextFileToOutput,
    Soze_OutputFilename,
    )

from .py.range_nodes import (
    Soze_IntRangeNode,
    Soze_FloatRangeNode,
    Soze_IntNumStepsRangeNode,
    Soze_FloatNumStepsRangeNode)

# from .py.imageresize import Soze_ImageResizeWithAspectCorrection

from .py.files import (
    Soze_LoadFilesFromFolder, 
    Soze_FileLoader, 
    Soze_DoesFileExist,
    Soze_LoadFilesWithPattern,
    )

from .py.json import (
    Soze_ParseValueFromJSONString, 
    Soze_JSONArrayIteratorNode, 
    Soze_JSONPathExtractorNode, 
    Soze_JSONFileLoader, 
    Soze_CreateImageBatchFromJSONArray,
    Soze_LoadImagesFromJSONArray,
    Soze_FormatJson, 
    Soze_JSONGetArrayCount,
    Soze_LoadJSONFileFromFolder,
    )

from .py.images import (
    Soze_ImageLabelOverlay,
    Soze_EmptyImages,
    Soze_XYImage,
    Soze_ImageListLoader,
    Soze_VariableImageBuilder,
    #Soze_AlphaCropAndPositionImage,
    Soze_ShrinkImage,
    Soze_LoadImage, 
    Soze_LoadImagesFromFolder, 
    Soze_BatchProcessSwitch, 
    Soze_LoadImageFromUrl, 
    Soze_GetImageColors, 
    Soze_PadMask,
    Soze_LoadImagesFromFolderXLora,
    Soze_LoadImageFromFilepath,
    Soze_MultiImageBatch,
    Soze_ImageSizeWithMaximum,
    Soze_SaveImageWithAbsoluteFilename
)
# from .py.samaudio import Soze_SAMAudioTextPrompt

from .py.fal import Veo31RefImgVideoNode

from .py.converters import (
    Soze_IntToString,
    Soze_StringToFloat,
    Soze_BoolToString,
    Soze_StringToBool,
    Soze_FloatToString,
    Soze_StringToInt,
    Soze_FloatToInt,
    Soze_IntToFloat,
    Soze_BooleanInverter
)

from .py.video import Soze_AppendToVideo, Soze_LoadVideosFromFolder

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
                        "Shrink Image": Soze_ShrinkImage,
                        "Pad Mask": Soze_PadMask,
                        "Checkpoint File Loader": Soze_CheckpointFilePathLoader,
                        "CSV Reader X Lora": Soze_CSVReaderXLora,
                        "Is String Empty": Soze_IsStringEmpty,
                        "ElevenLabs Voice Retriever": Soze_ElevenLabsVoiceRetrieverNode,
                        
                       
                        #ComfyDeploy
                        "ComfyDeploy API Mixed Parameters V2": Soze_ComfyDeployAPIMixedParametersV2,
                        "ComfyDeploy API Download Files": Soze_ComfyDeployDownloadAPIFiles,
                        "ComfyDeploy API Cache Run IDs": Soze_ComfyDeployCacheAPIRunIDs,
                        "ComfyDeploy API Retrieve Cached Run IDs": Soze_ComfyDeployRetrieveCachedAPIRunIDs,
                        "ComfyDeploy API Node": Soze_ComfyDeployAPINode,
                        "ComfyDeploy API String Parameters": Soze_ComfyDeployAPIStringParameters,
                        "ComfyDeploy API Int Parameters": Soze_ComfyDeployAPIIntParameters,
                        "ComfyDeploy API Float Parameters": Soze_ComfyDeployAPIFloatParameters,
                        "ComfyDeploy API Image Parameters": Soze_ComfyDeployAPIImageParameters,
                        "ComfyDeploy API Mixed Parameters": Soze_ComfyDeployAPIMixedParameters,
                        "ComfyDeploy API Boolean Parameters": Soze_ComfyDeployAPIBooleanParameters,
                        "ComfyDeploy API Cached Run Info": Soze_ComfyDeployCachedAPIRunInfo,
                        "ComfyDeploy API Clear Cached Run IDs": Soze_ComfyDeployClearCachedAPIRunIDs,
                        
                        #JSON
                        "JSON Value Parser": Soze_ParseValueFromJSONString,
                        "JSON Array Iterator": Soze_JSONArrayIteratorNode,
                        "JSON Path Extractor": Soze_JSONPathExtractorNode,
                        "JSON File Loader": Soze_JSONFileLoader,
                        "JSON Formatter": Soze_FormatJson,
                        "JSON Get Array Count": Soze_JSONGetArrayCount,
                        "JSON Load File From Folder": Soze_LoadJSONFileFromFolder,
                        
                        "Load Files From Folder": Soze_LoadFilesFromFolder,
                        "Load Images From Folder X Lora": Soze_LoadImagesFromFolderXLora,
                        "File Loader": Soze_FileLoader,
                        "String Functions": Soze_StringFunctions,
                        "Multi Find And Replace": Soze_MultiFindAndReplace,
                        "Append To Text File": Soze_AppendToTextFile,
                        "Create Image Batch From JSON Array": Soze_CreateImageBatchFromJSONArray,
                        "Load Image From Filepath": Soze_LoadImageFromFilepath,
                        "Load Images From JSONArray": Soze_LoadImagesFromJSONArray,
                        # "Image Resize With Aspect Correction": Soze_ImageResizeWithAspectCorrection,
                        "Multi Image Batch": Soze_MultiImageBatch,
                        "Load Text From File": Soze_LoadTextFromFile,
                        "Load Random Line From Text File": Soze_LoadRandomLineFromTextFile,
                        "String Splitter": Soze_StringSplitter,
                        "Empty String Replacement": Soze_EmptyStringReplacement,
                        "Save Text File To Output": Soze_SaveTextFileToOutput,
                        "Veo31 RefImg Video Node": Veo31RefImgVideoNode,
                        
                        #Converters
                        "Int To String": Soze_IntToString,
                        "String To Int": Soze_StringToInt,
                        "String To Float": Soze_StringToFloat,
                        "Float To String": Soze_FloatToString,
                        "Bool To String": Soze_BoolToString,
                        "String To Bool": Soze_StringToBool,
                        "Float To Int": Soze_FloatToInt,
                        "Int To Float": Soze_IntToFloat,
                        "Boolean Inverter": Soze_BooleanInverter,
                        
                        #Images
                        "Soze Image Size With Maximum": Soze_ImageSizeWithMaximum,
                        "Save Image With Absolute Filename": Soze_SaveImageWithAbsoluteFilename,
                        
                        #Video
                        "Append To Video": Soze_AppendToVideo,
                        "Load Videos From Folder": Soze_LoadVideosFromFolder,
                        
                        #Files
                        "Does File Exist": Soze_DoesFileExist,
                        "Load Files With Pattern": Soze_LoadFilesWithPattern,
                        
                        
                        
                        # "SAM Audio Text Prompt": Soze_SAMAudioTextPrompt,
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
                                "Shrink Image": "Shrink Image (Soze)",
                                "Pad Mask": "Pad Mask (Soze)",
                                "CSV Reader X Checkpoint": "CSV Reader X Checkpoint (Soze)",
                                "Checkpoint File Loader": "Checkpoint File Loader (Soze)",
                                "CSV Reader X Lora": "CSV Reader X Lora (Soze)",
                                "Is String Empty": "Is String Empty (Soze)",
                                "ElevenLabs Voice Retriever": "ElevenLabs Voice Retriever (Soze)",
                                                               
                                #ComfyDeploy 
                                "ComfyDeploy API Mixed Parameters V2": "ComfyDeploy API Mixed Parameters V2 (Soze)",
                                "ComfyDeploy API Download Files": "ComfyDeploy API Download Files (Soze)",
                                "ComfyDeploy API Cache Run IDs": "ComfyDeploy API Cache Run IDs (Soze)",
                                "ComfyDeploy API Retrieve Cached Run IDs": "ComfyDeploy API Retrieve Cached Run IDs (Soze)",
                                "ComfyDeploy API Node": "ComfyDeploy API Node (Soze)",
                                "ComfyDeploy API String Parameters": "ComfyDeploy API String Parameters (Soze)",
                                "ComfyDeploy API Int Parameters": "ComfyDeploy API Int Parameters (Soze)",
                                "ComfyDeploy API Float Parameters": "ComfyDeploy API Float Parameters (Soze)",
                                "ComfyDeploy API Image Parameters": "ComfyDeploy API Image Parameters (Soze)",
                                "ComfyDeploy API Mixed Parameters": "ComfyDeploy API Mixed Parameters (Soze)",
                                "ComfyDeploy API Boolean Parameters": "ComfyDeploy API Boolean Parameters (Soze)",
                                "ComfyDeploy API Cached Run Info": "ComfyDeploy API Cached Run Info (Soze)",
                                "ComfyDeploy API Clear Cached Run IDs": "ComfyDeploy API Clear Cached Run IDs (Soze)",
                                
                                #JSON
                                "JSON Value Parser": "JSON Value Parser (Soze)",
                                "JSON Array Iterator": "JSON Array Iterator (Soze)",
                                "JSON Path Extractor": "JSON Path Extractor (Soze)",
                                "JSON File Loader": "JSON File Loader (Soze)",
                                "JSON Formatter": "JSON Formatter (Soze)",
                                "JSON Get Array Count": "JSON Get Array Count (Soze)",
                                "JSON Load File From Folder": "JSON Load File From Folder (Soze)",
                                                                
                                
                                "Load Files From Folder": "Load Files From Folder (Soze)",
                                "Load Images From Folder X Lora": "Load Images From Folder X Lora (Soze)",
                                "File Loader": "File Loader (Soze)",
                                "String Functions": "String Functions (Soze)",
                                "Multi Find And Replace": "Multi Find And Replace (Soze)",
                                "Append To Text File": "Append To Text File (Soze)",
                                "Create Image Batch From JSON Array": "Create Image Batch From JSON Array (Soze)",
                                "Load Image From Filepath": "Load Image From Filepath (Soze)",
                                "Load Images From JSONArray": "Load Images From JSONArray (Soze)",
                                # "Image Resize With Aspect Correction": "Image Resize With Aspect Correction (Soze)",
                                "Multi Image Batch": "Multi Image Batch (Soze)",
                                "Load Text From File": "Load Text From File (Soze)",
                                "Load Random Line From Text File": "Load Random Line From Text File (Soze)",
                                "String Splitter": "String Splitter (Soze)",
                                "Empty String Replacement": "Empty String Replacement (Soze)",
                                "Save Text File To Output": "Save Text File To Output (Soze)",
                                "Veo31 RefImg Video Node": "Veo31 RefImg Video Node (Soze)",
                                
                                #Converters
                                "Int To String": "Int To String (Soze)",
                                "String To Int": "String To Int (Soze)",
                                "String To Float": "String To Float (Soze)",
                                "Float To String": "Float To String (Soze)",
                                "Bool To String": "Bool To String (Soze)",
                                "String To Bool": "String To Bool (Soze)",
                                "Float To Int": "Float To Int (Soze)",
                                "Int To Float": "Int To Float (Soze)",
                                "Boolean Inverter": "Boolean Inverter (Soze)",
                                
                                #Images
                                "Soze Image Size With Maximum": "Soze Image Size With Maximum (Soze)",
                                "Save Image With Absolute Filename": "Save Image With Absolute Filename (Soze)",
                                
                                #Video
                                "Append To Video": "Append To Video (Soze)",
                                "Load Videos From Folder": "Load Videos From Folder (Soze)",
                                
                                #Files
                                "Does File Exist": "Does File Exist (Soze)",
                                "Load Files With Pattern": "Load Files With Pattern (Soze)",
                                
                                
                                
                                
                                # "SAM Audio Text Prompt": "SAM Audio Text Prompt (Soze)",
                                
                              }

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']