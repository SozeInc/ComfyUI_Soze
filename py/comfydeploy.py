# import requests
# import os

# from io import BytesIO

# class Soze_ComfyDeployAPINode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "api_url": ("STRING", {"default": "https://api.comfydeploy.com/api/run/deployment/queue"}),
#                 "deployment_id": ("STRING", {"default": ""}),
#                 "input_image": ("STRING", {"default": ""}),  # Local file path or URL
#                 "positive_prompt": ("STRING", {"default": ""}),
#                 "input_number": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
#                 "batch_count": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
#                 "input_batch_size": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),


#             }
#         }

#     RETURN_TYPES = ("STRING",)
#     FUNCTION = "run"
#     CATEGORY = "utils"
#     OUTPUT_NODE = True

#     def run(self, api_url, deployment_id, input_image, positive_prompt, input_number, batch_count, input_batch_size):
#         api_key = self.load_api_key_from_env()
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {api_key}"
#         }
#         # Dynamically build inputs based on non-empty, stripped values
#         inputs = {}
#         if isinstance(input_image, str) and input_image.strip():
#             inputs["input_image"] = input_image.strip()
#         if isinstance(positive_prompt, str) and positive_prompt.strip():
#             inputs["positive_prompt"] = positive_prompt.strip()
#         # Check for additional attributes if present
#         if input_number is not None and str(input_number).strip() != "":
#             inputs["input_number"] = input_number
#         if batch_count is not None and str(batch_count).strip() != "":
#             inputs["batch_count"] = batch_count
#         if input_batch_size is not None and str(input_batch_size).strip() != "":
#             inputs["input_batch_size"] = input_batch_size

#         payload = {
#             "deployment_id": deployment_id,
#             "inputs": inputs
#         }
#         response = requests.post(api_url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise Exception(f"API call failed: {response.status_code} {response.text}")

#         result = response.json()
#         return (result,)

#     @staticmethod
#     def load_api_key_from_env():
#         # Get the parent directory of the current file
#         env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
#         if not os.path.exists(env_path):
#             raise Exception(f".env file not found at {env_path}")
#         with open(env_path, "r") as f:
#             for line in f:
#                 if line.strip().startswith("CD_API_KEY="):
#                     return line.strip().split("=", 1)[1]
#         raise Exception("CD_API_KEY not found in .env file in the nodes root directory")

