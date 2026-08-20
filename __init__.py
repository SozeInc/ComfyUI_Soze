# ComfyUI Mobile Nodes - A collection of ComfyUI quality of life related custom nodes.
# by Soze Inc - 2024-09
# https://github.com/SozeInc/ComfyUI-Soze


import os
import subprocess
import importlib.util
import shutil
import sys
import traceback

from .py.csvreader import Soze_CSVReader, Soze_CSVReaderXCheckpoint, Soze_CSVReaderXLora, Soze_CSVRandomReader
from .py.csvwriter import Soze_CSVWriter
from .py.xy import Soze_UnzippedProductAny
from .py.promptxlora import Soze_PromptFileFromFolderXLora
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
    Soze_AnyConcat,
    Soze_AnyEnumSwitch,
    )

from .py.range_nodes import (
    Soze_IntRangeNode,
    Soze_FloatRangeNode,
    Soze_IntNumStepsRangeNode,
    Soze_FloatNumStepsRangeNode)

from .py.files import (
    Soze_LoadFilesFromFolder,
    Soze_FileLoader,
    Soze_DoesFileExist,
    Soze_LoadFilesWithPattern,
    Soze_DownloadURL,
    Soze_SaveFileToOutput,
    Soze_ExtractZipToOutput,
    )

from .py.json import (
    Soze_ParseValueFromJSONString,
    Soze_JSONArrayIteratorNode,
    Soze_JSONPathExtractorNode,
    Soze_JSONStringParser10X,
    Soze_JSONFileLoader,
    Soze_CreateImageBatchFromJSONArray,
    Soze_LoadImagesFromJSONArray,
    Soze_FormatJson,
    Soze_JSONGetArrayCount,
    Soze_LoadJSONFileFromFolder,
    )

from .py.json_builder import (
    Soze_JSONStringPair,
    Soze_JSONStringPairs10X,
    Soze_JSONIntPair,
    Soze_JSONFloatPair,
    Soze_JSONBoolPair,
    Soze_JSONArrayPair,
    Soze_JSONImagePair,
    Soze_JSONImageEncoder,
    Soze_JSONImageDecoder,
    Soze_JSONImageArrayPair,
    Soze_JSONAudioPair,
    Soze_JSONAudioEncoder,
    Soze_JSONAudioDecoder,
    Soze_JSONAudioArrayPair,
    Soze_JSONObjectPair,
    Soze_JSONRawPair,
    Soze_JSONGenerate,
)

from .py.images import (
    Soze_ImageLabelOverlay,
    Soze_EmptyImages,
    Soze_XYImage,
    Soze_ImageListLoader,
    Soze_VariableImageBuilder,
    #Soze_AlphaCropAndPositionImage,
    Soze_ShrinkImage,
    Soze_ScribbleXDoG,
    Soze_Lineart,
    Soze_LoadImage,
    Soze_LoadImagesFromFolder,
    Soze_LoadRandomImagesFromFolder,
    Soze_BatchProcessSwitch,
    Soze_LoadImageFromUrl, 
    Soze_GetImageColors, 
    Soze_PadMask,
    Soze_LoadImagesFromFolderXLora,
    Soze_LoadImageFromFilepath,
    Soze_MultiImageBatch,
    Soze_ImageSizeWithMaximum,
    Soze_SaveImageWithAbsoluteFilename,
    Soze_SaveImageBatchWithFilenames,
    Soze_LoadImagesFromUrlList,
    Soze_ImageCrop
)

from .py.fal import (
    Soze_FALSeedance2ImageToVideo,
    Soze_FALSeedance2ReferenceToVideo,
    Soze_FALGPTImage2Edit,
    Soze_FALGPTImage2,
    Soze_FALSeedream5LiteEdit,
    Soze_FALSeedream5LiteTextToImage,
    Soze_FALSeedream5ProEdit,
    Soze_FALSeedream45Edit,
    Soze_FALSeedream45TextToImage,
    Soze_FALKlingO3ReferenceToVideo,
    Soze_FALKlingO3FirstLastFrameVideo,
    Soze_FALKlingO3EditVideo,
    Soze_FALKlingO3ReferenceVideoToVideo,
    Soze_FALTopazUpscaleVideo,
    Soze_FALMiniMaxH3ReferenceToVideo,
    Veo31RefImgVideoNode,
)

from .py.fal_video_models import (
    Soze_FALSeedance25ReferenceToVideo,
    Soze_FALGeminiOmniFlashReferenceToVideo,
    Soze_FALKlingV3TurboStandardImageToVideo,
    Soze_FALGrokImagineReferenceToVideo,
    Soze_FALHappyHorse11ReferenceToVideo,
    Soze_FALKlingO34KReferenceToVideo,
    Soze_FALPixVerseC1ReferenceToVideo,
    Soze_FALKlingO1ProReferenceToVideo,
    Soze_FALKlingO1StandardReferenceToVideo,
    Soze_FALViduQ3ReferenceToVideoMix,
    Soze_FALViduQ1ReferenceToVideo,
    Soze_FALMirageAvatarXReferenceToVideo,
)

from .py.modelark import (
    Soze_ModelArkSeedance2,
    Soze_ModelArkSeedreamImages,
)

from .py.oxenai import Soze_OxenAIChatCompletion

from .py.minimax import Soze_MiniMaxH3Video, Soze_MiniMaxH3VideoReference

from .py.tensorscale import (
    Soze_TensorScaleMiniMaxH3,
    Soze_TensorScaleMiniMaxH3Reference,
    Soze_TensorScaleLTX25Fast,
    Soze_TensorScaleLTX23Fast,
    Soze_TensorScaleCosmos3NanoT2V,
    Soze_TensorScaleCosmos3NanoV2V,
    Soze_TensorScaleHunyuanImage3,
    Soze_TensorScaleSenseNovaU15,
)

from .py.rmbg_deploy import Soze_ComfyDeployBiRefNetModelInput

from .py.model_switch import (
    Soze_EnumSwitchCheckpointLoader,
    Soze_EnumSwitchUpscaleModelLoader,
    Soze_EnumSwitchLoraLoader,
    Soze_EnumSwitchDiffusionModelLoader,
)

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

from .py.audio import Soze_LoadAudio

from .py.av_combine import Soze_CombineVideo

from .py.get_frame import Soze_GetFrame

NODE_CLASS_MAPPINGS = { "Output Filename": Soze_OutputFilename,
                        "Load Image": Soze_LoadImage,
                        "Load Audio": Soze_LoadAudio,
                        "Load Images From Folder": Soze_LoadImagesFromFolder,
                        "Load Random Images From Folder": Soze_LoadRandomImagesFromFolder,
                        "Image Batch Process Switch": Soze_BatchProcessSwitch,
                        "Load Image From URL": Soze_LoadImageFromUrl,
                        "CSV Reader": Soze_CSVReader,
                        "CSV Random Reader": Soze_CSVRandomReader,
                        "CSV Reader X Checkpoint": Soze_CSVReaderXCheckpoint,
                        "CSV Writer": Soze_CSVWriter,
                        "Special Character Replacer": Soze_SpecialCharacterReplacer,
                        "Multiline Concatenate Strings": Soze_MultilineConcatenateStrings,
                        "Any Concat": Soze_AnyConcat,
                        "Any Enum Switch": Soze_AnyEnumSwitch,
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
                        "Scribble XDoG": Soze_ScribbleXDoG,
                        "Lineart": Soze_Lineart,
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
                        "JSON String Parser (10X)": Soze_JSONStringParser10X,
                        "JSON File Loader": Soze_JSONFileLoader,
                        "JSON Formatter": Soze_FormatJson,
                        "JSON Get Array Count": Soze_JSONGetArrayCount,
                        "JSON Load File From Folder": Soze_LoadJSONFileFromFolder,
                        "JSON String Pair": Soze_JSONStringPair,
                        "JSON String Pairs (10X)": Soze_JSONStringPairs10X,
                        "JSON Int Pair": Soze_JSONIntPair,
                        "JSON Float Pair": Soze_JSONFloatPair,
                        "JSON Bool Pair": Soze_JSONBoolPair,
                        "JSON Array Pair": Soze_JSONArrayPair,
                        "JSON Image Pair": Soze_JSONImagePair,
                        "JSON Image Encoder": Soze_JSONImageEncoder,
                        "JSON Image Decoder": Soze_JSONImageDecoder,
                        "JSON Image Array Pair": Soze_JSONImageArrayPair,
                        "JSON Audio Pair": Soze_JSONAudioPair,
                        "JSON Audio Encoder": Soze_JSONAudioEncoder,
                        "JSON Audio Decoder": Soze_JSONAudioDecoder,
                        "JSON Audio Array Pair": Soze_JSONAudioArrayPair,
                        "JSON Object Pair": Soze_JSONObjectPair,
                        "JSON Raw Pair": Soze_JSONRawPair,
                        "JSON Generate": Soze_JSONGenerate,
                        
                        "Load Files From Folder": Soze_LoadFilesFromFolder,
                        "Load Images From Folder X Lora": Soze_LoadImagesFromFolderXLora,
                        "File Loader": Soze_FileLoader,
                        "String Functions": Soze_StringFunctions,
                        "Multi Find And Replace": Soze_MultiFindAndReplace,
                        "Append To Text File": Soze_AppendToTextFile,
                        "Create Image Batch From JSON Array": Soze_CreateImageBatchFromJSONArray,
                        "Load Image From Filepath": Soze_LoadImageFromFilepath,
                        "Load Images From JSONArray": Soze_LoadImagesFromJSONArray,
                        "Multi Image Batch": Soze_MultiImageBatch,
                        "Load Text From File": Soze_LoadTextFromFile,
                        "Load Random Line From Text File": Soze_LoadRandomLineFromTextFile,
                        "String Splitter": Soze_StringSplitter,
                        "Empty String Replacement": Soze_EmptyStringReplacement,
                        "Save Text File To Output": Soze_SaveTextFileToOutput,
                        "Veo31 RefImg Video Node": Veo31RefImgVideoNode,
                        "ModelArk Seedance 2.0": Soze_ModelArkSeedance2,
                        "ModelArk Seedream Images": Soze_ModelArkSeedreamImages,
                        "Oxen AI Chat Completion": Soze_OxenAIChatCompletion,
                        "MiniMax H3 Video": Soze_MiniMaxH3Video,
                        "MiniMax H3 Video Reference": Soze_MiniMaxH3VideoReference,
                        "TensorScale MiniMax H3 Video": Soze_TensorScaleMiniMaxH3,
                        "TensorScale MiniMax H3 Video Reference": Soze_TensorScaleMiniMaxH3Reference,
                        "TensorScale LTX-2.5 Fast Video": Soze_TensorScaleLTX25Fast,
                        "TensorScale LTX-2.3 Fast Video": Soze_TensorScaleLTX23Fast,
                        "TensorScale Cosmos3 Nano Text To Video": Soze_TensorScaleCosmos3NanoT2V,
                        "TensorScale Cosmos3 Nano Video To Video": Soze_TensorScaleCosmos3NanoV2V,
                        "TensorScale Hunyuan Image 3 Fast": Soze_TensorScaleHunyuanImage3,
                        "TensorScale SenseNova U1.5": Soze_TensorScaleSenseNovaU15,
                        "ComfyDeploy BiRefNet Model Input": Soze_ComfyDeployBiRefNetModelInput,
                        "Checkpoint Enum Switch": Soze_EnumSwitchCheckpointLoader,
                        "Upscale Model Enum Switch": Soze_EnumSwitchUpscaleModelLoader,
                        "Lora Enum Switch": Soze_EnumSwitchLoraLoader,
                        "Diffusion Model Enum Switch": Soze_EnumSwitchDiffusionModelLoader,
                        "FAL Seedance 2 Image To Video": Soze_FALSeedance2ImageToVideo,
                        "FAL Seedance 2 Reference To Video": Soze_FALSeedance2ReferenceToVideo,
                        "FAL GPT Image 2 Edit": Soze_FALGPTImage2Edit,
                        "FAL GPT Image 2": Soze_FALGPTImage2,
                        "FAL Seedream v5 Lite Edit": Soze_FALSeedream5LiteEdit,
                        "FAL Seedream v5 Lite Text To Image": Soze_FALSeedream5LiteTextToImage,
                        "FAL Seedream v5 Pro Edit": Soze_FALSeedream5ProEdit,
                        "FAL Seedream v4.5 Edit": Soze_FALSeedream45Edit,
                        "FAL Seedream v4.5 Text To Image": Soze_FALSeedream45TextToImage,
                        "FAL Kling O3 Reference To Video": Soze_FALKlingO3ReferenceToVideo,
                        "FAL Kling O3 First-Last Frame Video": Soze_FALKlingO3FirstLastFrameVideo,
                        "FAL Kling O3 Edit Video": Soze_FALKlingO3EditVideo,
                        "FAL Kling O3 Reference Video To Video": Soze_FALKlingO3ReferenceVideoToVideo,
                        "FAL Topaz Upscale Video": Soze_FALTopazUpscaleVideo,
                        "FAL MiniMax H3 Reference To Video": Soze_FALMiniMaxH3ReferenceToVideo,
                        "FAL Seedance 2.5 Reference To Video": Soze_FALSeedance25ReferenceToVideo,
                        "FAL Gemini Omni Flash Reference To Video": Soze_FALGeminiOmniFlashReferenceToVideo,
                        "FAL Kling V3 Turbo Standard Image To Video": Soze_FALKlingV3TurboStandardImageToVideo,
                        "FAL Grok Imagine Reference To Video": Soze_FALGrokImagineReferenceToVideo,
                        "FAL Happy Horse 1.1 Reference To Video": Soze_FALHappyHorse11ReferenceToVideo,
                        "FAL Kling O3 4K Reference To Video": Soze_FALKlingO34KReferenceToVideo,
                        "FAL PixVerse C1 Reference To Video": Soze_FALPixVerseC1ReferenceToVideo,
                        "FAL Kling O1 Pro Reference To Video": Soze_FALKlingO1ProReferenceToVideo,
                        "FAL Kling O1 Standard Reference To Video": Soze_FALKlingO1StandardReferenceToVideo,
                        "FAL Vidu Q3 Reference To Video Mix": Soze_FALViduQ3ReferenceToVideoMix,
                        "FAL Vidu Q1 Reference To Video": Soze_FALViduQ1ReferenceToVideo,
                        "FAL Mirage Avatar X Reference To Video": Soze_FALMirageAvatarXReferenceToVideo,
                        
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
                        "Save Image Batch With Filenames": Soze_SaveImageBatchWithFilenames,
                        "Load Images From URL List": Soze_LoadImagesFromUrlList,
                        "Image Crop": Soze_ImageCrop,
                        
                        #Video
                        "Append To Video": Soze_AppendToVideo,
                        "Load Videos From Folder": Soze_LoadVideosFromFolder,
                        "Combine Video": Soze_CombineVideo,
                        "Get Frame": Soze_GetFrame,
                        
                        #Files
                        "Does File Exist": Soze_DoesFileExist,
                        "Load Files With Pattern": Soze_LoadFilesWithPattern,
                        "Download URL": Soze_DownloadURL,
                        "Save File To Output": Soze_SaveFileToOutput,
                        "Extract Zip To Output": Soze_ExtractZipToOutput,
                        
                        "Load Prompt From Folder X Lora": Soze_PromptFileFromFolderXLora,
                        }

NODE_DISPLAY_NAME_MAPPINGS = { "Output Filename": "Output Filename (Soze)",
                                "Load Image": "Load Image (Soze)",
                                "Load Audio": "Load Audio (Soze)",
                                "Load Images From Folder": "Load Images From Folder (Soze)",
                                "Load Random Images From Folder": "Load Random Images From Folder (Soze)",
                                "Image Batch Process Switch": "Image Batch Process Switch (Soze)",
                                "Load Image From URL": "Load Image From URL (Soze)",
                                "CSV Reader": "CSV Reader (Soze)",
                                "CSV Random Reader": "CSV Random Reader (Soze)",
                                "CSV Writer": "CSV Writer (Soze)",
                                "Special Character Replacer": "Special Character Replacer (Soze)",                               
                                "Multiline Concatenate Strings": "Multiline Concatenate (Soze)",
                                "Any Concat": "Any Concat 10X (Soze)",
                                "Any Enum Switch": "Any Enum Switch 10X (Soze)",
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
                                "Scribble XDoG": "Scribble XDoG (Soze)",
                                "Lineart": "Lineart (Soze)",
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
                                "JSON String Parser (10X)": "JSON String Parser (10X) (Soze)",
                                "JSON File Loader": "JSON File Loader (Soze)",
                                "JSON Formatter": "JSON Formatter (Soze)",
                                "JSON Get Array Count": "JSON Get Array Count (Soze)",
                                "JSON Load File From Folder": "JSON Load File From Folder (Soze)",
                                "JSON String Pair": "JSON Chain String Pair (Soze)",
                                "JSON String Pairs (10X)": "JSON Chain String Pairs (10X) (Soze)",
                                "JSON Int Pair": "JSON Chain Int Pair (Soze)",
                                "JSON Float Pair": "JSON Chain Float Pair (Soze)",
                                "JSON Bool Pair": "JSON Chain Bool Pair (Soze)",
                                "JSON Array Pair": "JSON Chain Array Pair (Soze)",
                                "JSON Image Pair": "JSON Chain Image Pair (Soze)",
                                "JSON Image Encoder": "JSON Image Encoder (Soze)",
                                "JSON Image Decoder": "JSON Image Decoder (Soze)",
                                "JSON Image Array Pair": "JSON Chain Image Array Pair (Soze)",
                                "JSON Audio Pair": "JSON Chain Audio Pair (Soze)",
                                "JSON Audio Encoder": "JSON Audio Encoder (Soze)",
                                "JSON Audio Decoder": "JSON Audio Decoder (Soze)",
                                "JSON Audio Array Pair": "JSON Chain Audio Array Pair (Soze)",
                                "JSON Object Pair": "JSON Chain Object Pair (Soze)",
                                "JSON Raw Pair": "JSON Chain Raw Pair (Soze)",
                                "JSON Generate": "JSON Chain Generate (Soze)",
                                                                
                                
                                "Load Files From Folder": "Load Files From Folder (Soze)",
                                "Load Images From Folder X Lora": "Load Images From Folder X Lora (Soze)",
                                "File Loader": "File Loader (Soze)",
                                "String Functions": "String Functions (Soze)",
                                "Multi Find And Replace": "Multi Find And Replace (Soze)",
                                "Append To Text File": "Append To Text File (Soze)",
                                "Create Image Batch From JSON Array": "Create Image Batch From JSON Array (Soze)",
                                "Load Image From Filepath": "Load Image From Filepath (Soze)",
                                "Load Images From JSONArray": "Load Images From JSONArray (Soze)",
                                "Multi Image Batch": "Multi Image Batch (Soze)",
                                "Load Text From File": "Load Text From File (Soze)",
                                "Load Random Line From Text File": "Load Random Line From Text File (Soze)",
                                "String Splitter": "String Splitter (Soze)",
                                "Empty String Replacement": "Empty String Replacement (Soze)",
                                "Save Text File To Output": "Save Text File To Output (Soze)",
                                "Veo31 RefImg Video Node": "Veo31 RefImg Video Node (Soze)",
                                "ModelArk Seedance 2.0": "ModelArk Seedance 2.0 (Soze)",
                                "ModelArk Seedream Images": "ModelArk Seedream Images (Soze)",
                                "Oxen AI Chat Completion": "Oxen AI Chat Completion (Soze)",
                                "MiniMax H3 Video": "MiniMax H3 Video (Soze)",
                                "MiniMax H3 Video Reference": "MiniMax H3 Video Reference (Soze)",
                                "TensorScale MiniMax H3 Video": "TensorScale MiniMax H3 Video (Soze)",
                                "TensorScale MiniMax H3 Video Reference": "TensorScale MiniMax H3 Video Reference (Soze)",
                                "TensorScale LTX-2.5 Fast Video": "TensorScale LTX-2.5 Fast Video (Soze)",
                                "TensorScale LTX-2.3 Fast Video": "TensorScale LTX-2.3 Fast Video (Soze)",
                                "TensorScale Cosmos3 Nano Text To Video": "TensorScale Cosmos3 Nano Text To Video (Soze)",
                                "TensorScale Cosmos3 Nano Video To Video": "TensorScale Cosmos3 Nano Video To Video (Soze)",
                                "TensorScale Hunyuan Image 3 Fast": "TensorScale Hunyuan Image 3 Fast (Soze)",
                                "TensorScale SenseNova U1.5": "TensorScale SenseNova U1.5 (Soze)",
                                "ComfyDeploy BiRefNet Model Input": "ComfyDeploy BiRefNet Model Input (Soze)",
                                "Checkpoint Enum Switch": "Checkpoint Enum Switch 10X (Soze)",
                                "Upscale Model Enum Switch": "Upscale Model Enum Switch 10X (Soze)",
                                "Lora Enum Switch": "Lora Enum Switch 10X (Soze)",
                                "Diffusion Model Enum Switch": "Diffusion Model Enum Switch 10X (Soze)",
                                "FAL Seedance 2 Image To Video": "FAL Seedance 2 Image To Video (Soze)",
                                "FAL Seedance 2 Reference To Video": "FAL Seedance 2 Reference To Video (Soze)",
                                "FAL GPT Image 2 Edit": "FAL GPT Image 2 Edit (Soze)",
                                "FAL GPT Image 2": "FAL GPT Image 2 (Soze)",
                                "FAL Seedream v5 Lite Edit": "FAL Seedream v5 Lite Edit (Soze)",
                                "FAL Seedream v5 Lite Text To Image": "FAL Seedream v5 Lite Text To Image (Soze)",
                                "FAL Seedream v5 Pro Edit": "FAL Seedream v5 Pro Edit (Soze)",
                                "FAL Seedream v4.5 Edit": "FAL Seedream v4.5 Edit (Soze)",
                                "FAL Seedream v4.5 Text To Image": "FAL Seedream v4.5 Text To Image (Soze)",
                                "FAL Kling O3 Reference To Video": "FAL Kling O3 Reference To Video (Soze)",
                                "FAL Kling O3 First-Last Frame Video": "FAL Kling O3 First-Last Frame Video (Soze)",
                                "FAL Kling O3 Edit Video": "FAL Kling O3 Edit Video (Soze)",
                                "FAL Kling O3 Reference Video To Video": "FAL Kling O3 Reference Video To Video (Soze)",
                                "FAL Topaz Upscale Video": "FAL Topaz Upscale Video (Soze)",
                                "FAL MiniMax H3 Reference To Video": "FAL MiniMax H3 Reference To Video (Soze)",
                                "FAL Seedance 2.5 Reference To Video": "FAL Seedance 2.5 Reference To Video (Soze)",
                                "FAL Gemini Omni Flash Reference To Video": "FAL Gemini Omni Flash Reference To Video (Soze)",
                                "FAL Kling V3 Turbo Standard Image To Video": "FAL Kling V3 Turbo Standard Image To Video (Soze)",
                                "FAL Grok Imagine Reference To Video": "FAL Grok Imagine Reference To Video (Soze)",
                                "FAL Happy Horse 1.1 Reference To Video": "FAL Happy Horse 1.1 Reference To Video (Soze)",
                                "FAL Kling O3 4K Reference To Video": "FAL Kling O3 4K Reference To Video (Soze)",
                                "FAL PixVerse C1 Reference To Video": "FAL PixVerse C1 Reference To Video (Soze)",
                                "FAL Kling O1 Pro Reference To Video": "FAL Kling O1 Pro Reference To Video (Soze)",
                                "FAL Kling O1 Standard Reference To Video": "FAL Kling O1 Standard Reference To Video (Soze)",
                                "FAL Vidu Q3 Reference To Video Mix": "FAL Vidu Q3 Reference To Video Mix (Soze)",
                                "FAL Vidu Q1 Reference To Video": "FAL Vidu Q1 Reference To Video (Soze)",
                                "FAL Mirage Avatar X Reference To Video": "FAL Mirage Avatar X Reference To Video (Soze)",
                                
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
                                "Save Image Batch With Filenames": "Save Image Batch With Filenames (Soze)",
                                "Load Images From URL List": "Load Images From URL List (Soze)",
                                "Image Crop": "Image Crop (Soze)",
                                
                                #Video
                                "Append To Video": "Append To Video (Soze)",
                                "Load Videos From Folder": "Load Videos From Folder (Soze)",
                                "Combine Video": "Combine Video (Soze)",
                                "Get Frame": "Get Frame (Soze)",
                                
                                #Files
                                "Does File Exist": "Does File Exist (Soze)",
                                "Load Files With Pattern": "Load Files With Pattern (Soze)",
                                "Download URL": "Download URL (Soze)",
                                "Save File To Output": "Save File To Output (Soze)",
                                "Extract Zip To Output": "Extract Zip To Output (Soze)",
                                
                                
                                "Load Prompt From Folder X Lora": "Load Prompt From Folder X Lora (Soze)",
                              }

# Add a thumbnail preview strip to the bottom of image-loading nodes.
# Centralized here so the loader functions' many early-return branches are all
# covered without editing each return site.
from .py.preview_utils import enable_image_preview
enable_image_preview(
    Soze_LoadImage,
    Soze_LoadImagesFromFolder,
    Soze_LoadRandomImagesFromFolder,
    Soze_LoadImageFromFilepath,
    Soze_LoadImagesFromFolderXLora,
    Soze_LoadImageFromUrl,
    Soze_ImageListLoader,
    Soze_LoadImagesFromUrlList,
    Soze_LoadImagesFromJSONArray,
    Soze_CreateImageBatchFromJSONArray,
)

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']