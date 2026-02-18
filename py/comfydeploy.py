from email.mime import image
import requests
import os
import torch
import time
import folder_paths
import comfy.model_management

from PIL import Image, ImageOps
from io import BytesIO
from azure.storage.blob import BlobServiceClient
import numpy as np
from PIL import Image
from server import PromptServer
from .images import pil2tensor


def upload_to_azure_storage(image):
    # Get connection string from environment using conn_string_prefix
    env_var_name = "SOZE_AZURE_STORAGE_CONNECTION_STRING"
    connection_string = os.getenv(env_var_name)
    if not connection_string:
        raise EnvironmentError(f"{env_var_name} environment variable not set.")
    

    # Prepare image for upload
    if isinstance(image, torch.Tensor):
        # Remove batch and channel dimensions if present
        arr = image
        if arr.dim() == 4:  # (B, C, H, W)
            arr = arr.squeeze(0)
        if arr.dim() == 3 and arr.shape[0] in [1, 3]:  # (C, H, W)
            arr = arr.permute(1, 2, 0)  # (H, W, C)
        i = 255. * arr.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Unsupported image type for upload.")

    # Save image to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Generate a unique filename
    import uuid
    filename = f"{uuid.uuid4().hex}.png"

    # Set Azure container and blob path
    container_name = "comfyui-output"
    blob_name = f"inputs/{filename}"

    # Upload to Azure
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    container_client.upload_blob(name=blob_name, data=img_bytes, overwrite=True)
    print(f"Uploaded image to {container_name}/{blob_name}")

    # Construct public URL (assuming container is set for public access)
    # Try to extract storage account name from connection string
    try:
        storage_account = [x for x in connection_string.split(";") if x.startswith("AccountName=")][0].split("=")[1]
    except Exception:
        storage_account = "unknown"
    url = f"https://{storage_account}.blob.core.windows.net/{container_name}/{blob_name}"
    return url    

class Soze_ComfyDeployAPINode:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_parameters": ("STRING", {"forceInput": True}),
                "api_url": ("STRING", {"default": "https://api.comfydeploy.com/api/run/deployment/queue"}),
                "deployment_id": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("run_id", )
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, api_parameters, api_url, deployment_id):
        api_key = load_api_key_from_env()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # Parse api_parameters string into a dictionary
        parameters = {}
        if api_parameters.strip():
            for pair in api_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Format values for JSON: bool, int, float as native types; strings and images as quoted
                    if v.lower() in ["true", "false"]:
                        parameters[k] = True if v.lower() == "true" else False
                    else:
                        try:
                            if "." in v:
                                parameters[k] = float(v)
                            else:
                                parameters[k] = int(v)
                        except ValueError:
                            # Strings and image URLs should be quoted in the parameter string, but in JSON they are just strings
                            parameters[k] = v

        payload = {
            "deployment_id": deployment_id,
            "inputs": parameters
        }

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code != 200:
                    raise Exception(f"API call failed: {response.status_code} {response.text}")
                result = response.json()
                return (result,)
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"API call failed (attempt {attempt}/{max_retries}): {e}. Retrying in 5 seconds...")
                time.sleep(5)


class Soze_ComfyDeployAPIStringParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "string_value_1": ("STRING", {"default": "", "forceInput": False}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "string_value_2": ("STRING", {"default": "", "forceInput": False}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "string_value_3": ("STRING", {"default": "", "forceInput": False}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "string_value_4": ("STRING", {"default": "", "forceInput": False}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "string_value_5": ("STRING", {"default": "", "forceInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            name_1, string_value_1,
            name_2, string_value_2,
            name_3, string_value_3,
            name_4, string_value_4,
            name_5, string_value_5,
            more_parameters = ""):
        params = []
        # Parse more_parameters if provided
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        # Only add parameters if name_X is not empty
        for name, value in [
            (name_1, string_value_1),
            (name_2, string_value_2),
            (name_3, string_value_3),
            (name_4, string_value_4),
            (name_5, string_value_5),
        ]:
            if name.strip() and value.strip():
                params.append(f"{name.strip()}={value.strip()}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIIntParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "int_value_1": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "int_value_2": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "int_value_3": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "int_value_4": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "int_value_5": ("INT", {"default": 0, "min": -1, "forceInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            name_1, int_value_1,
            name_2, int_value_2,
            name_3, int_value_3,
            name_4, int_value_4,
            name_5, int_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (name_1, int_value_1),
            (name_2, int_value_2),
            (name_3, int_value_3),
            (name_4, int_value_4),
            (name_5, int_value_5),
        ]:
            if name.strip() and value is not None:
                # Ensure integers are not quoted
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIFloatParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "number_value_1": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "number_value_2": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "number_value_3": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "number_value_4": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "number_value_5": ("FLOAT", {"default": 0.0, "forceInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            name_1, number_value_1,
            name_2, number_value_2,
            name_3, number_value_3,
            name_4, number_value_4,
            name_5, number_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (name_1, number_value_1),
            (name_2, number_value_2),
            (name_3, number_value_3),
            (name_4, number_value_4),
            (name_5, number_value_5),
        ]:
            if name.strip() and value is not None:
                # Ensure floats are not quoted
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIBooleanParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "boolean_value_1": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "boolean_value_2": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "boolean_value_3": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "boolean_value_4": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "boolean_value_5": ("BOOLEAN", {"default": False, "forceInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self,
            name_1, boolean_value_1,
            name_2, boolean_value_2,
            name_3, boolean_value_3,
            name_4, boolean_value_4,
            name_5, boolean_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (name_1, boolean_value_1),
            (name_2, boolean_value_2),
            (name_3, boolean_value_3),
            (name_4, boolean_value_4),
            (name_5, boolean_value_5),
        ]:
            if name.strip() and value is not None:
                # Ensure boolean values are lower case and not quoted
                if isinstance(value, bool):
                    value = "true" if value else "false"
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIMixedParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "string_value_1": ("STRING", {"default": "", "forceInput": False}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "string_value_2": ("STRING", {"default": "", "forceInput": False}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "int_value_3": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "int_value_4": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "int_value_5": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "name_6": ("STRING", {"default": "", "forceInput": False}),
                "number_value_6": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_7": ("STRING", {"default": "", "forceInput": False}),
                "number_value_7": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_8": ("STRING", {"default": "", "forceInput": False}),
                "number_value_8": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "name_9": ("STRING", {"default": "", "forceInput": False}),
                "bool_value_9": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_10": ("STRING", {"default": "", "forceInput": False}),
                "bool_value_10": ("BOOLEAN", {"default": False, "forceInput": False}),
                "name_11": ("STRING", {"default": "", "forceInput": False}),
                "image_value_11": ("IMAGE", {"default": None}),
                "name_12": ("STRING", {"default": "", "forceInput": False}),
                "image_value_12": ("IMAGE", {"default": None}),
                "name_13": ("STRING", {"default": "", "forceInput": False}),
                "image_value_13": ("IMAGE", {"default": None}),
                "name_14": ("STRING", {"default": "", "forceInput": False}),
                "image_value_14": ("IMAGE", {"default": None}),
                "name_15": ("STRING", {"default": "", "forceInput": False}),
                "image_value_15": ("IMAGE", {"default": None}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            name_1, string_value_1,
            name_2, string_value_2,
            name_3 = "", int_value_3 = 0,
            name_4 = "", int_value_4 = 0,
            name_5 = "", int_value_5 = 0,
            name_6 = "", number_value_6 = 0.0,
            name_7 = "", number_value_7 = 0.0,
            name_8 = "", number_value_8 = 0.0,
            name_9 = "", bool_value_9 = False,
            name_10 = "", bool_value_10 = False,
            name_11 = "", image_value_11 = None,
            name_12 = "", image_value_12 = None,
            name_13 = "", image_value_13 = None,
            name_14 = "", image_value_14 = None,
            name_15 = "", image_value_15 = None,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (name_1, string_value_1),
            (name_2, string_value_2),
            (name_3, int_value_3),
            (name_4, int_value_4),
            (name_5, int_value_5),
            (name_6, number_value_6),
            (name_7, number_value_7),
            (name_8, number_value_8),
            (name_9, bool_value_9),
            (name_10, bool_value_10),
            (name_11, image_value_11),
            (name_12, image_value_12),
            (name_13, image_value_13),
            (name_14, image_value_14),
            (name_15, image_value_15),
        ]:
            if name.strip() and value is not None and (not isinstance(value, str) or value.strip()):
                # Ensure booleans, ints, and floats are not quoted
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    pass  # leave as is, don't quote
                elif isinstance(value, torch.Tensor) or (isinstance(value, (Image.Image, bytes))):
                    value = upload_to_azure_storage(value)
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)


class Soze_ComfyDeployAPIMixedParametersV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "save_filename": ("STRING", {"default": "", "forceInput": True}),
                "positive_prompt": ("STRING", {"default": "", "forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "forceInput": True}),
                "batch_count": ("INT", {"default": 1, "min": 1, "forceInput": True}),
                "string_name_1": ("STRING", {"default": "", "forceInput": False}),
                "string_value_1": ("STRING", {"default": "", "forceInput": False}),
                "string_name_2": ("STRING", {"default": "", "forceInput": False}),
                "string_value_2": ("STRING", {"default": "", "forceInput": False}),
                "int_name_3": ("STRING", {"default": "", "forceInput": False}),
                "int_value_3": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "int_name_4": ("STRING", {"default": "", "forceInput": False}),
                "int_value_4": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "int_name_5": ("STRING", {"default": "", "forceInput": False}),
                "int_value_5": ("INT", {"default": 0, "min": -1, "forceInput": False}),
                "float_name_6": ("STRING", {"default": "", "forceInput": False}),
                "float_value_6": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "float_name_7": ("STRING", {"default": "", "forceInput": False}),
                "float_value_7": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "float_name_8": ("STRING", {"default": "", "forceInput": False}),
                "float_value_8": ("FLOAT", {"default": 0.0, "forceInput": False}),
                "bool_name_9": ("STRING", {"default": "", "forceInput": False}),
                "bool_value_9": ("BOOLEAN", {"default": False, "forceInput": False}),
                "bool_name_10": ("STRING", {"default": "", "forceInput": False}),
                "bool_value_10": ("BOOLEAN", {"default": False, "forceInput": False}),
                "image_name_11": ("STRING", {"default": "", "forceInput": False}),
                "image_value_11": ("IMAGE", {"default": None}),
                "image_name_12": ("STRING", {"default": "", "forceInput": False}),
                "image_value_12": ("IMAGE", {"default": None}),
                "image_name_13": ("STRING", {"default": "", "forceInput": False}),
                "image_value_13": ("IMAGE", {"default": None}),
                "image_name_14": ("STRING", {"default": "", "forceInput": False}),
                "image_value_14": ("IMAGE", {"default": None}),
                "image_name_15": ("STRING", {"default": "", "forceInput": False}),
                "image_value_15": ("IMAGE", {"default": None}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self,
            string_name_1, string_value_1,
            string_name_2, string_value_2,
            int_name_3, int_value_3,
            int_name_4, int_value_4,
            int_name_5, int_value_5,
            float_name_6, float_value_6,
            float_name_7, float_value_7,
            float_name_8, float_value_8,
            bool_name_9, bool_value_9,
            bool_name_10, bool_value_10,
            image_name_11 = "", image_value_11 = None,
            image_name_12 = "", image_value_12 = None,
            image_name_13 = "", image_value_13 = None,
            image_name_14 = "", image_value_14 = None,
            image_name_15 = "", image_value_15 = None,
            more_parameters = "",
            save_filename = "",
            positive_prompt = "",
            seed = -1,
            batch_count = 1,):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (string_name_1, string_value_1),
            (string_name_2, string_value_2),
            (int_name_3, int_value_3),
            (int_name_4, int_value_4),
            (int_name_5, int_value_5),
            (float_name_6, float_value_6),
            (float_name_7, float_value_7),
            (float_name_8, float_value_8),
            (bool_name_9, bool_value_9),
            (bool_name_10, bool_value_10),
            (image_name_11, image_value_11),
            (image_name_12, image_value_12),
            (image_name_13, image_value_13),
            (image_name_14, image_value_14),
            (image_name_15, image_value_15),            
        ]:
            if name.strip() and value is not None and (not isinstance(value, str) or value.strip()):
                # Ensure booleans, ints, and floats are not quoted
                if isinstance(value, bool):
                    value = "true" if value else "false"
                elif isinstance(value, (int, float)):
                    pass  # leave as is, don't quote
                elif isinstance(value, torch.Tensor) or (isinstance(value, (Image.Image, bytes))):
                    value = upload_to_azure_storage(value)
                params.append(f"{name.strip()}={value}")
                
        if save_filename.strip():
            params.append(f"save_filename={save_filename.strip()}")
        if positive_prompt.strip():
            params.append(f"positive_prompt={positive_prompt.strip()}")
        if seed is not None:
            params.append(f"seed={seed}")
        if batch_count is not None:
            params.append(f"batch_count={batch_count}")
            
        param_string = ";".join(params)
        return (param_string,)



class Soze_ComfyDeployAPIImageParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "forceInput": False}),
                "image_value_1": ("IMAGE", {"default": None, "forceInput": True}),
                "name_2": ("STRING", {"default": "", "forceInput": False}),
                "image_value_2": ("IMAGE", {"default": None, "forceInput": True}),
                "name_3": ("STRING", {"default": "", "forceInput": False}),
                "image_value_3": ("IMAGE", {"default": None, "forceInput": True}),
                "name_4": ("STRING", {"default": "", "forceInput": False}),
                "image_value_4": ("IMAGE", {"default": None, "forceInput": True}),
                "name_5": ("STRING", {"default": "", "forceInput": False}),
                "image_value_5": ("IMAGE", {"default": None, "forceInput": True}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            name_1, name_2, name_3, name_4, name_5, 
            image_value_1 = None,
            image_value_2 = None,
            image_value_3 = None,
            image_value_4 = None,
            image_value_5 = None,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for name, value in [
            (name_1, image_value_1),
            (name_2, image_value_2),
            (name_3, image_value_3),
            (name_4, image_value_4),
            (name_5, image_value_5),
        ]:
            if name.strip() and value is not None:
                value = upload_to_azure_storage(value)
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)

        return (param_string,)
    
    
class Soze_ComfyDeployCacheAPIRunIDs:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {"forceInput": True}),
                "cache_save_folder": ("STRING", {"default": "", "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }        
        }
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, run_id, cache_save_folder, unique_id=None, extra_pnginfo=None, prompt=None):
        try:
            if isinstance(run_id, dict):
                 if "run_id" in run_id:
                     run_id = run_id["run_id"]
            elif isinstance(run_id, str) and run_id.strip().startswith("{"):
                import json
                run_id_json = json.loads(run_id)
                if "run_id" in run_id_json:
                    run_id = run_id_json["run_id"]
        except Exception:
            pass
            
        user_id = "default"
        
        # Check if we have prompt data (this dictionary comes from the frontend)
        if prompt and isinstance(prompt, dict):
             # 'client_id' is frequently injected here by the frontend
            if "client_id" in prompt:
                 user_id = prompt["client_id"]
        
        # 2. Use the ID to get the directory
        # Note: Ensure folder_paths has this method (it might be in a specific branch/version)
        try:
            if hasattr(folder_paths, "get_public_user_directory"):
                user_dir = folder_paths.get_public_user_directory(user_id)
            else:
                # Fallback for older versions: use standard user directory
                user_dir = os.path.join(folder_paths.user_directory, user_id)
                print(f"Warning: get_public_user_directory not found. Using {user_dir}")
        except Exception as e:
            print(f"Error resolving user directory: {e}")
            user_dir = folder_paths.get_output_directory() # Safe fallback

        run_id_cache_folder = os.path.join(user_dir, "RunIDCache")
        os.makedirs(run_id_cache_folder, exist_ok=True)

        # 3. Create cache folder if specified
        if cache_save_folder.strip():
            relative_folder = cache_save_folder.strip().lstrip("/").lstrip("\\")
            cache_folder = os.path.join(run_id_cache_folder, relative_folder)
            os.makedirs(cache_folder, exist_ok=True)
            # Save run_id to a text file        
            cache_file_path = os.path.join(cache_folder, f"{run_id}")
            with open(cache_file_path, "w") as f:
                f.write("")
        return ()   

class Soze_ComfyDeployCachedAPIRunInfo:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_save_folder": ("STRING", {"default": "", "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }        
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("run_ids", "count", "has_run_ids")
    FUNCTION = "get_info"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True
    
    def get_info(self, cache_save_folder, unique_id=None, extra_pnginfo=None, prompt=None):
        user_id = "default"
        
        # Check if we have prompt data (this dictionary comes from the frontend)
        if prompt and isinstance(prompt, dict):
             # 'client_id' is frequently injected here by the frontend
            if "client_id" in prompt:
                 user_id = prompt["client_id"]
        
        # 2. Use the ID to get the directory
        # Note: Ensure folder_paths has this method (it might be in a specific branch/version)
        try:
            if hasattr(folder_paths, "get_public_user_directory"):
                user_dir = folder_paths.get_public_user_directory(user_id)
            else:
                # Fallback for older versions: use standard user directory
                user_dir = os.path.join(folder_paths.user_directory, user_id)
                print(f"Warning: get_public_user_directory not found. Using {user_dir}")
        except Exception as e:
            print(f"Error resolving user directory: {e}")
            user_dir = folder_paths.get_output_directory() # Safe fallback

        run_id_cache_folder = os.path.join(user_dir, "RunIDCache")
        os.makedirs(run_id_cache_folder, exist_ok=True)

        # 3. Check cache folder if specified
        run_ids = []
        if cache_save_folder.strip():   
            relative_folder = cache_save_folder.strip().lstrip("/").lstrip("\\")
            cache_folder = os.path.join(run_id_cache_folder, relative_folder)
            
            if os.path.exists(cache_folder):
                run_id_files = [f for f in os.listdir(cache_folder) if os.path.isfile(os.path.join(cache_folder, f)) and not f.startswith("_removed_")]
                run_ids = run_id_files

        return (";".join(run_ids), len(run_ids), len(run_ids) > 0)
    
    
    
class Soze_ComfyDeployRetrieveCachedAPIRunIDs:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_save_folder": ("STRING", {"default": "", "forceInput": False}),
                "remove_after_retrieval": ("BOOLEAN", {"default": True, "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }        
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("run_id",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, cache_save_folder, remove_after_retrieval=False, unique_id=None, extra_pnginfo=None, prompt=None):
        user_id = "default"
        
        # Check if we have prompt data (this dictionary comes from the frontend)
        if prompt and isinstance(prompt, dict):
             # 'client_id' is frequently injected here by the frontend
            if "client_id" in prompt:
                 user_id = prompt["client_id"]
        
        # 2. Use the ID to get the directory
        # Note: Ensure folder_paths has this method (it might be in a specific branch/version)
        try:
            if hasattr(folder_paths, "get_public_user_directory"):
                user_dir = folder_paths.get_public_user_directory(user_id)
            else:
                # Fallback for older versions: use standard user directory
                user_dir = os.path.join(folder_paths.user_directory, user_id)
                print(f"Warning: get_public_user_directory not found. Using {user_dir}")
        except Exception as e:
            print(f"Error resolving user directory: {e}")
            user_dir = folder_paths.get_output_directory() # Safe fallback

        run_id_cache_folder = os.path.join(user_dir, "RunIDCache")
        os.makedirs(run_id_cache_folder, exist_ok=True)

        # 3. Check cache folder if specified
        run_id = ""
        if cache_save_folder.strip():   
            relative_folder = cache_save_folder.strip().lstrip("/").lstrip("\\")
            cache_folder = os.path.join(run_id_cache_folder, relative_folder)
            
            if os.path.exists(cache_folder):
                run_id_files = [f for f in os.listdir(cache_folder) if os.path.isfile(os.path.join(cache_folder, f)) and not f.startswith("_removed_")]
                run_id_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_folder, x)))
                
                for f in run_id_files:
                    run_id = f
                    cache_file_path = os.path.join(cache_folder, f"{run_id}")
                    removed_file_path = os.path.join(cache_folder, f"_removed_{run_id}")
                    if os.path.exists(cache_file_path):
                        if remove_after_retrieval:
                            try:
                                os.rename(cache_file_path, removed_file_path)
                            except Exception as e:
                                print(f"Error removing cached file: {e}")
                    break

        if run_id == "":
            raise Exception("No cached run_id found in the specified folder.")
        else:
            return (run_id,)   
        
        
        
class Soze_ComfyDeployClearCachedAPIRunIDs:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cache_save_folder": ("STRING", {"default": "", "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
            }        
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("run_id",)
    FUNCTION = "clear"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def clear(self, cache_save_folder, remove_after_retrieval=False, unique_id=None, extra_pnginfo=None, prompt=None):
        user_id = "default"
        
        # Check if we have prompt data (this dictionary comes from the frontend)
        if prompt and isinstance(prompt, dict):
             # 'client_id' is frequently injected here by the frontend
            if "client_id" in prompt:
                 user_id = prompt["client_id"]
        
        # 2. Use the ID to get the directory
        # Note: Ensure folder_paths has this method (it might be in a specific branch/version)
        try:
            if hasattr(folder_paths, "get_public_user_directory"):
                user_dir = folder_paths.get_public_user_directory(user_id)
            else:
                # Fallback for older versions: use standard user directory
                user_dir = os.path.join(folder_paths.user_directory, user_id)
                print(f"Warning: get_public_user_directory not found. Using {user_dir}")
        except Exception as e:
            print(f"Error resolving user directory: {e}")
            user_dir = folder_paths.get_output_directory() # Safe fallback

        run_id_cache_folder = os.path.join(user_dir, "RunIDCache")
        os.makedirs(run_id_cache_folder, exist_ok=True)

        # 3. Check cache folder if specified
        run_id = ""
        if cache_save_folder.strip():   
            relative_folder = cache_save_folder.strip().lstrip("/").lstrip("\\")
            cache_folder = os.path.join(run_id_cache_folder, relative_folder)
            
            if os.path.exists(cache_folder):
                run_id_files = [f for f in os.listdir(cache_folder) if os.path.isfile(os.path.join(cache_folder, f)) and not f.startswith("_removed_")]
                run_id_files.sort(key=lambda x: os.path.getmtime(os.path.join(cache_folder, x)))
                
                for f in run_id_files:
                    run_id = f
                    cache_file_path = os.path.join(cache_folder, f"{run_id}")
                    removed_file_path = os.path.join(cache_folder, os.path.join(time.strftime("%Y%m%d-%H%M%S"),f"{run_id}"))
                    if os.path.exists(cache_file_path):
                        if remove_after_retrieval:
                            try:
                                os.rename(cache_file_path, removed_file_path)
                            except Exception as e:
                                print(f"Error removing cached file: {e}")
                    break

        if run_id == "":
            raise Exception("No cached run_id found in the specified folder.")
        else:
            return (run_id,)   

    
    
class Soze_ComfyDeployDownloadAPIFiles:
    def IS_CHANGED(self, *args, **kwargs):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {"forceInput": True}),
                "output_save_folder": ("STRING", {"default": "", "forceInput": False}),
                "api_url": ("STRING", {"default": "https://api.comfydeploy.com/api/run"}),
                "wait_max_seconds": ("INT", {"default": 600, "min": 1, "max": 3600, "forceInput": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING","IMAGE","STRING")
    RETURN_NAMES = ("filepaths", "images", "videos")
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, run_id, output_save_folder, api_url, wait_max_seconds, unique_id=None,):
        def send_status(status_text):
            if unique_id:
                try:
                    PromptServer.instance.send_sync("soze.comfydeploy.status", {"node_id": unique_id, "status_text": status_text})
                except:
                    pass

        try:
            if isinstance(run_id, dict):
                 if "run_id" in run_id:
                     run_id = run_id["run_id"]
            elif isinstance(run_id, str) and run_id.strip().startswith("{"):
                import json
                run_id_json = json.loads(run_id)
                if "run_id" in run_id_json:
                    run_id = run_id_json["run_id"]
        except Exception:
            pass
            
        api_key = load_api_key_from_env()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Normalize API URL
        if api_url.endswith("/"):
            api_url = api_url[:-1]

        start_time = time.time()
        send_status("Checking status...")
        while True:
            response = requests.get(f"{api_url}/{run_id}", headers=headers)
            if response.status_code != 200:
                send_status(f"API Error: {response.status_code}")
                raise Exception(f"API call failed: {response.status_code} {response.text}")
            result = response.json()
            status = result.get("status", "")
            
            images = []
            videos = []
            # Map remote URL -> filename provided by API (if any)
            filename_map = {}
            
            if status == "completed" or status == "success":
                send_status("Run completed. Processing outputs...")
                filepaths = []
                
                # Try to extract from 'outputs' (standard structure)
                outputs = result.get("outputs", [])
                for output in outputs:
                    data = output.get("data", {})
                    # Some APIs return a top-level 'files' key on the output itself
                    files_list = output.get("files", [])
                    # Also check inside data for a 'files' key
                    if not files_list and isinstance(data, dict):
                        files_list = data.get("files", [])

                    # Handle images (if present)
                    if "images" in data:
                        for img in data["images"]:
                            if isinstance(img, dict) and "url" in img:
                                filepaths.append(img["url"])
                                images.append(img["url"])
                            elif isinstance(img, str):
                                filepaths.append(img)
                                images.append(img)

                    # Handle single video object (if present)
                    if "video" in data:
                        video = data["video"]
                        # video might be a dict with url, a list, or a simple string
                        if isinstance(video, dict) and "url" in video:
                            filepaths.append(video["url"])
                            videos.append(video["url"])
                        elif isinstance(video, list):
                            for v in video:
                                if isinstance(v, dict) and "url" in v:
                                    filepaths.append(v["url"])
                                    videos.append(v["url"])
                                elif isinstance(v, str):
                                    filepaths.append(v)
                                    videos.append(v)
                        elif isinstance(video, str):
                            filepaths.append(video)
                            videos.append(video)

                    # Handle plural videos array if present
                    if "videos" in data:
                        for v in data["videos"]:
                            if isinstance(v, dict) and "url" in v:
                                filepaths.append(v["url"])
                                videos.append(v["url"])
                            elif isinstance(v, str):
                                filepaths.append(v)
                                videos.append(v)

                    # Handle generic files list (may include mp4s, other assets)
                    if files_list:
                        for f in files_list:
                            # f may be a dict with url/filename or a plain string
                            if isinstance(f, dict):
                                url = f.get("url") or f.get("file")
                                filename_from_api = f.get("filename") or f.get("name")
                            elif isinstance(f, str):
                                url = f
                                filename_from_api = None
                            else:
                                continue

                            if not url:
                                continue

                            # classify by extension when possible
                            _, ext = os.path.splitext(url.split("?")[0])
                            ext = ext.lower().lstrip('.')
                            if ext in ("mp4", "mov", "webm", "mkv", "avi"):
                                videos.append(url)
                                filepaths.append(url)
                            elif ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"):
                                images.append(url)
                                filepaths.append(url)
                            else:
                                # Unknown file type: still download and keep track
                                filepaths.append(url)
                                # store filename if present for later use in saving
                                if filename_from_api:
                                    filename_map[url] = filename_from_api
                
                # Fallback to 'output_files' if 'outputs' yielded nothing (older API structure)
                if not filepaths:
                    output_files = result.get("output_files", [])
                    for file_info in output_files:
                        if "url" in file_info:
                            filepaths.append(file_info["url"])

                # Download files to output_save_folder if specified
                url_to_local_path = {}
                if output_save_folder.strip():
                    relative_folder = output_save_folder.strip().lstrip("/").lstrip("\\")
                    full_output_folder = os.path.join(folder_paths.get_output_directory(), relative_folder)
                    os.makedirs(full_output_folder, exist_ok=True)
                    local_filepaths = []
                    total_files = len(filepaths)
                    from urllib.parse import unquote
                    for i, file_url in enumerate(filepaths):
                        # Prefer filename provided by API (if any), otherwise use basename
                        filename_override = filename_map.get(file_url)
                        if filename_override:
                            filename = filename_override
                        else:
                            filename = os.path.basename(file_url.split('?')[0])
                        
                        filename = unquote(filename)
                        local_path = os.path.join(full_output_folder, filename)
                        # If file exists, follow ComfyUI style and add an incremented suffix: _00001
                        if os.path.exists(local_path):
                            name, ext = os.path.splitext(filename)
                            counter = 1
                            while True:
                                padded = f"_{counter:05d}"
                                new_name = f"{name}{padded}{ext}"
                                candidate = os.path.join(full_output_folder, new_name)
                                if not os.path.exists(candidate):
                                    local_path = candidate
                                    filename = new_name
                                    break
                                counter += 1
                        send_status(f"Downloading {i+1}/{total_files}: {filename}")
                        
                        for attempt in range(3):
                            try:
                                with requests.get(file_url, stream=True) as r:
                                    r.raise_for_status()
                                    with open(local_path, 'wb') as f:
                                        for chunk in r.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                break
                            except Exception as e:
                                if attempt == 2:
                                    send_status(f"Download failed: {filename}")
                                    raise
                                print(f"Download failed (attempt {attempt+1}/3): {e}. Retrying...")
                                time.sleep(2)

                        local_filepaths.append(local_path)
                        url_to_local_path[file_url] = local_path
                    filepaths = local_filepaths
                
                # Load Images
                out_images = []
                for img_source in images:
                    try:
                        # Use local path if downloaded, else URL
                        path = url_to_local_path.get(img_source, img_source)
                        
                        if path.startswith("http"):
                             response = requests.get(path)
                             i = Image.open(BytesIO(response.content))
                        else:
                             i = Image.open(path)

                        i = ImageOps.exif_transpose(i)
                        if i.mode == 'I':
                            i = i.point(lambda i: i * (1 / 255))
                        image = i.convert("RGB")
                        image = np.array(image).astype(np.float32) / 255.0
                        image = torch.from_numpy(image)[None,]
                        out_images.append(image)
                    except Exception as e:
                        print(f"Failed to load image {img_source}: {e}")

                if out_images:
                    out_images_tensor = torch.cat(out_images, dim=0)
                else:
                    out_images_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

                # Preparing Videos
                # If we have local paths, return them, otherwise URLs
                out_videos = [url_to_local_path.get(v, v) for v in videos]
                
                send_status("Done.")
                return (";".join(filepaths), out_images_tensor, ";".join(out_videos))
            elif status == "failed":
                send_status("Run failed.")
                #raise Exception("Run failed according to ComfyDeploy API.")
                return ("", torch.zeros((1, 64, 64, 3), dtype=torch.float32), "")
            elif time.time() - start_time > wait_max_seconds:
                send_status("Timed out.")
                raise TimeoutError("Waiting for run to complete timed out.")
            elif status == "cancelled":
                send_status("Run cancelled, nothing to download.")
                return ("", torch.zeros((1, 64, 64, 3), dtype=torch.float32), "")
            else:
                remaining = int(wait_max_seconds - (time.time() - start_time))
                send_status(f"Status: {status}. Remaining: {remaining}s")
                print(f"Run status: {status}. Waiting...")
                comfy.model_management.throw_exception_if_processing_interrupted()
                time.sleep(5)





def load_api_key_from_env():
    """
    Loads the API key from the CD_API_KEY environment variable.
    Raises an exception if not found.
    """
    api_key = os.getenv("CD_API_KEY")
    if not api_key:
        raise EnvironmentError("CD_API_KEY environment variable not set.")
    return api_key



