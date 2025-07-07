# import requests
# import os

# from PIL import Image
# from io import BytesIO

# def load_api_key_from_env():
#     # Get the parent directory of the current file
#     env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
#     if not os.path.exists(env_path):
#         raise Exception(f".env file not found at {env_path}")
#     with open(env_path, "r") as f:
#         for line in f:
#             if line.strip().startswith("CD_API_KEY="):
#                 return line.strip().split("=", 1)[1]
#     raise Exception("CD_API_KEY not found in .env file")

# class Soze_ComfyDeployAPINode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "api_url": ("STRING", {"default": "https://api.comfydeploy.com/api/run/deployment/queue"}),
#                 "deployment_id": ("STRING", {"default": ""}),
#                 "input_image_path": ("STRING", {"default": ""}),  # Local file path or URL
#                 "positive_prompt": ("STRING", {"default": ""}),
#             }
#         }

#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "run"
#     CATEGORY = "utils"
#     OUTPUT_NODE = True

#     def run(self, api_url, deployment_id, input_image_path, positive_prompt):
#         api_key = load_api_key_from_env()
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {api_key}"
#         }
#         payload = {
#             "deployment_id": deployment_id,
#             "inputs": {
#                 "input_image": input_image_path,
#                 "positive_prompt": positive_prompt
#             }
#         }
#         response = requests.post(api_url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise Exception(f"API call failed: {response.status_code} {response.text}")

#         result = response.json()
#         output_image_url = result.get("output_image") or result.get("image_url")
#         if not output_image_url:
#             print("API response did not contain an output image. Full response:")
#             print(result)
#             raise Exception("No output image found in API response: " + result)

#         img_response = requests.get(output_image_url)
#         if img_response.status_code != 200:
#             raise Exception(f"Failed to download output image: {img_response.status_code}")

#         img = Image.open(BytesIO(img_response.content))
#         return (img,)
