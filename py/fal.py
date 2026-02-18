from .fal_utils import ApiHandler, FalConfig, ImageUtils
from comfy_api_nodes.util import (download_url_to_video_output)



class Veo31RefImgVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 8}),
            },
            "optional": {
                "generate_audio": ("BOOLEAN", {"default": True}),
                "auto_fix": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate_video"
    CATEGORY = "FAL/VideoGeneration"

    async def generate_video(
        self,
        images,
        prompt,
        resolution,
        duration,
        generate_audio=True,
        auto_fix=True,
    ):
        image_urls = []
        for i in range(images.shape[0]):
            url = ImageUtils.upload_image(images[i])
            if url:
                image_urls.append(url)

        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "resolution": resolution,
            "duration": duration,
            "generate_audio": generate_audio,
            "auto_fix": auto_fix,
        }

        try:
            result = ApiHandler.submit_and_get_result(
                "fal-ai/veo3.1/reference-to-video", arguments
            )
            video_url = result["video"]["url"]
            print(f"Veo3.1 Video URL: {video_url}")
            return (await download_url_to_video_output(video_url),)
        except Exception as e:
            raise Exception(str(e))

