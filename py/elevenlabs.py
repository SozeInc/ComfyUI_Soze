import time
import requests
import os

from folder_paths import output_directory

def load_elevenlabs_api_key():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        raise Exception(f".env file not found at {env_path}")
    with open(env_path, "r") as f:
        for line in f:
            if line.strip().startswith("ELEVENLABS_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise Exception("ELEVENLABS_API_KEY not found in .env file")

class Soze_ElevenLabsVoiceRetrieverNode:
    def __init__(self):
        pass

    def IS_CHANGED(self):
        return time.time()    

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_id": ("STRING", {"default": ""}),
                "sample_save_path": ("STRING", {"default": ""}),  # Local directory to save samples
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_data",)
    FUNCTION = "run"
    CATEGORY = "audio"
    OUTPUT_NODE = True

    def run(self, voice_id, sample_save_path):
        import tempfile
        import shutil
        api_key = load_elevenlabs_api_key()
        headers = {
            "xi-api-key": api_key
        }
        url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} {response.text}")
        data = response.json()

        sample_files = []
        # Determine save directory within ComfyUI output directory
        subfolder = sample_save_path.strip() if isinstance(sample_save_path, str) and sample_save_path.strip() else "elevenlabs_samples"
        save_dir = os.path.join(output_directory, subfolder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Download all sample audio files by making a separate API call for each sample
        if "samples" in data and data["samples"]:
            for idx, sample in enumerate(data["samples"]):
                sample_id = sample.get("sample_id")
                if sample_id:
                    sample_url = f"https://api.elevenlabs.io/v1/voices/{voice_id}/samples/{sample_id}/audio"
                    audio_resp = requests.get(sample_url, headers=headers)
                    if audio_resp.status_code == 200:
                        audio_content = audio_resp.content
                        audio_name = data.get('name')
                        if audio_content:
                            file_ext = ".mp3"
                            file_path = os.path.join(save_dir, f"{audio_name}_{voice_id}_sample_{sample_id}{file_ext}")
                            with open(file_path, "wb") as f:
                                f.write(audio_content)
                            sample_files.append(file_path)
                        else:
                            sample_files.append(f"No audio data for sample {sample_id}")
                    else:
                        sample_files.append(f"Failed to fetch sample {sample_id}: {audio_resp.status_code}")
                else:
                    sample_files.append("No sample_id for sample")
        else:
            sample_files.append("No samples found for this voice ID.")

        # Return all other data as formatted text
        import json
        voice_data_text = json.dumps(data, indent=2)
        return (voice_data_text,)


