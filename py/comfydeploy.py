import requests
import os
import torch

from PIL import Image
from io import BytesIO
from azure.storage.blob import BlobServiceClient
import numpy as np
from PIL import Image


def upload_to_azure_storage(image):
    # Get connection string from environment using conn_string_prefix
    env_var_name = "SOZE_AZURE_STORAGE_CONNECTION_STRING"
    connection_string = os.getenv(env_var_name)
    if not connection_string:
        print(f"Environment variable {env_var_name} not set.")
        return None

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
    blob_name = f"comfyui-input/{filename}"

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
                    if k.strip() and v.strip():
                        parameters[k.strip()] = v.strip()

        payload = {
            "deployment_id": deployment_id,
            "inputs": parameters
        }
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} {response.text}")

        result = response.json()
        return (result,)


class Soze_ComfyDeployAPIStringParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "use_param_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "string_value_1": ("STRING", {"default": "", "isInput": False}),
                "use_param_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "string_value_2": ("STRING", {"default": "", "isInput": False}),
                "use_param_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "string_value_3": ("STRING", {"default": "", "isInput": False}),
                "use_param_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "string_value_4": ("STRING", {"default": "", "isInput": False}),
                "use_param_5": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
                "string_value_5": ("STRING", {"default": "", "isInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            use_param_1, name_1, string_value_1,
            use_param_2, name_2, string_value_2,
            use_param_3, name_3, string_value_3,
            use_param_4, name_4, string_value_4,
            use_param_5, name_5, string_value_5,
            more_parameters = ""):
        params = []
        # Parse more_parameters if provided
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        # Only add parameters if use_param_X is True
        for use, name, value in [
            (use_param_1, name_1, string_value_1),
            (use_param_2, name_2, string_value_2),
            (use_param_3, name_3, string_value_3),
            (use_param_4, name_4, string_value_4),
            (use_param_5, name_5, string_value_5),
        ]:
            if use and name.strip() and value.strip():
                params.append(f"{name.strip()}={value.strip()}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIIntParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "use_param_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "int_value_1": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "int_value_2": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "int_value_3": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "int_value_4": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_5": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
                "int_value_5": ("INT", {"default": 0, "min": -1, "isInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            use_param_1, name_1, int_value_1,
            use_param_2, name_2, int_value_2,
            use_param_3, name_3, int_value_3,
            use_param_4, name_4, int_value_4,
            use_param_5, name_5, int_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for use, name, value in [
            (use_param_1, name_1, int_value_1),
            (use_param_2, name_2, int_value_2),
            (use_param_3, name_3, int_value_3),
            (use_param_4, name_4, int_value_4),
            (use_param_5, name_5, int_value_5),
        ]:
            if use and name.strip() and value is not None:
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIFloatParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "use_param_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "number_value_1": ("FLOAT", {"default": 0.0, "isInput": False}),
                "use_param_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "number_value_2": ("FLOAT", {"default": 0.0, "isInput": False}),
                "use_param_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "number_value_3": ("FLOAT", {"default": 0.0, "isInput": False}),
                "use_param_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "number_value_4": ("FLOAT", {"default": 0.0, "isInput": False}),
                "use_param_5": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
                "number_value_5": ("FLOAT", {"default": 0.0, "isInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            use_param_1, name_1, number_value_1,
            use_param_2, name_2, number_value_2,
            use_param_3, name_3, number_value_3,
            use_param_4, name_4, number_value_4,
            use_param_5, name_5, number_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for use, name, value in [
            (use_param_1, name_1, number_value_1),
            (use_param_2, name_2, number_value_2),
            (use_param_3, name_3, number_value_3),
            (use_param_4, name_4, number_value_4),
            (use_param_5, name_5, number_value_5),
        ]:
            if use and name.strip() and value is not None:
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIBooleanParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "use_param_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "boolean_value_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "use_param_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "boolean_value_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "use_param_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "boolean_value_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "use_param_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "boolean_value_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "use_param_5": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
                "boolean_value_5": ("BOOLEAN", {"default": False, "isInput": False}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self,
            use_param_1, name_1, boolean_value_1,
            use_param_2, name_2, boolean_value_2,
            use_param_3, name_3, boolean_value_3,
            use_param_4, name_4, boolean_value_4,
            use_param_5, name_5, boolean_value_5,
            more_parameters = ""):
        params = []
        if more_parameters.strip():
            for pair in more_parameters.strip().split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    if k.strip() and v.strip():
                        params.append(f"{k.strip()}={v.strip()}")
        for use, name, value in [
            (use_param_1, name_1, boolean_value_1),
            (use_param_2, name_2, boolean_value_2),
            (use_param_3, name_3, boolean_value_3),
            (use_param_4, name_4, boolean_value_4),
            (use_param_5, name_5, boolean_value_5),
        ]:
            if use and name.strip() and value is not None:
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)

class Soze_ComfyDeployAPIMixedParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "use_param_1": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "string_value_1": ("STRING", {"default": "", "isInput": False}),
                "use_param_2": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "string_value_2": ("STRING", {"default": "", "isInput": False}),
                "use_param_3": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "int_value_3": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_4": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "int_value_4": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_5": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
                "int_value_5": ("INT", {"default": 0, "min": -1, "isInput": False}),
                "use_param_6": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_6": ("STRING", {"default": "", "isInput": False}),
                "number_value_6": ("FLOAT", {"default": 0.0, "isInput": True}),
                "use_param_7": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_7": ("STRING", {"default": "", "isInput": False}),
                "number_value_7": ("FLOAT", {"default": 0.0, "isInput": True}),
                "use_param_8": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_8": ("STRING", {"default": "", "isInput": False}),
                "number_value_8": ("FLOAT", {"default": 0.0, "isInput": True}),
                "use_param_9": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_9": ("STRING", {"default": "", "isInput": False}),
                "bool_value_9": ("BOOLEAN", {"default": False, "isInput": False}),
                "use_param_10": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_10": ("STRING", {"default": "", "isInput": False}),
                "bool_value_10": ("BOOLEAN", {"default": False, "isInput": False}),
                "name_11": ("STRING", {"default": "", "isInput": False}),
                "image_value_11": ("IMAGE", {"default": None}),
                "name_12": ("STRING", {"default": "", "isInput": False}),
                "image_value_12": ("IMAGE", {"default": None}),
                "name_13": ("STRING", {"default": "", "isInput": False}),
                "image_value_13": ("IMAGE", {"default": None}),
                "name_14": ("STRING", {"default": "", "isInput": False}),
                "image_value_14": ("IMAGE", {"default": None}),
                "name_15": ("STRING", {"default": "", "isInput": False}),
                "image_value_15": ("IMAGE", {"default": None}),
            }
        }

    RETURN_NAMES = ("api_parameters", )
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "Soze Nodes"
    OUTPUT_NODE = True

    def run(self, 
            use_param_1, name_1, string_value_1,
            use_param_2, name_2, string_value_2,
            use_param_3, name_3, int_value_3,
            use_param_4, name_4, int_value_4,
            use_param_5, name_5, int_value_5,
            use_param_6, name_6, number_value_6,
            use_param_7, name_7, number_value_7,
            use_param_8, name_8, number_value_8,
            use_param_9, name_9, bool_value_9,
            use_param_10, name_10, bool_value_10,
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
        for use, name, value in [
            (use_param_1, name_1, string_value_1),
            (use_param_2, name_2, string_value_2),
            (use_param_3, name_3, int_value_3),
            (use_param_4, name_4, int_value_4),
            (use_param_5, name_5, int_value_5),
            (use_param_6, name_6, number_value_6),
            (use_param_7, name_7, number_value_7),
            (use_param_8, name_8, number_value_8),
            (use_param_9, name_9, bool_value_9),
            (use_param_10, name_10, bool_value_10),
            (name_11.strip() and image_value_11 is not None, name_11, image_value_11),
            (name_12.strip() and image_value_12 is not None, name_12, image_value_12),
            (name_13.strip() and image_value_13 is not None, name_13, image_value_13),
            (name_14.strip() and image_value_14 is not None, name_14, image_value_14),
            (name_15.strip() and image_value_15 is not None, name_15, image_value_15),
        ]:
            if use and name.strip() and value is not None and (not isinstance(value, str) or value.strip()):
                # Convert image values to base64 if needed
                if isinstance(value, torch.Tensor) or (isinstance(value, (Image.Image, bytes))):
                    value = upload_to_azure_storage(value)
                params.append(f"{name.strip()}={value}")
        param_string = ";".join(params)
        return (param_string,)


class Soze_ComfyDeployAPIImageParameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "more_parameters": ("STRING", {"default": "", "forceInput": True}),
                "name_1": ("STRING", {"default": "", "isInput": False}),
                "image_value_1": ("IMAGE", {"default": None, "forceInput": True}),
                "name_2": ("STRING", {"default": "", "isInput": False}),
                "image_value_2": ("IMAGE", {"default": None, "forceInput": True}),
                "name_3": ("STRING", {"default": "", "isInput": False}),
                "image_value_3": ("IMAGE", {"default": None, "forceInput": True}),
                "name_4": ("STRING", {"default": "", "isInput": False}),
                "image_value_4": ("IMAGE", {"default": None, "forceInput": True}),
                "name_5": ("STRING", {"default": "", "isInput": False}),
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

def load_api_key_from_env():
    """
    Loads the API key from the CD_API_KEY environment variable.
    Raises an exception if not found.
    """
    api_key = os.getenv("CD_API_KEY")
    if not api_key:
        raise EnvironmentError("CD_API_KEY environment variable not set.")
    return api_key
