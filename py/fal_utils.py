import configparser
import io
import logging
import os
import tempfile
import time

import numpy as np
import requests
import torch
from fal_client.client import SyncClient
from PIL import Image

logger = logging.getLogger(__name__)

HTTP_TIMEOUT = (10, 120)


class FalConfig:
    """Singleton class to handle FAL configuration and client setup."""

    _instance = None
    _client = None
    _key = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration and API key."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        env_key = os.environ.get("FAL_KEY")
        if env_key:
            logger.info("FAL_KEY found in environment variables")
            self._key = env_key
        else:
            try:
                self._key = config["API"]["FAL_KEY"]
                logger.info("FAL_KEY found in config.ini")
            except KeyError:
                logger.error("FAL_KEY not found in config.ini or environment variables")
                self._key = None
                return

        if self._key == "<your_fal_api_key_here>":
            logger.warning(
                "Default FAL API key placeholder detected. "
                "Set FAL_KEY in environment or in config.ini under [API]. "
                "Get a key at https://fal.ai/dashboard/keys"
            )
            self._key = None

    def get_client(self):
        """Get or create the FAL client."""
        if self._key is None:
            raise RuntimeError(
                "FAL_KEY is not configured. Set FAL_KEY in environment or config.ini."
            )
        if self._client is None:
            self._client = SyncClient(key=self._key)
        return self._client

    def get_key(self):
        return self._key


class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def tensor_to_pil(image):
        try:
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = np.array(image)

            if image_np.ndim == 4:
                image_np = image_np.squeeze(0)
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[0] == 3:
                image_np = np.transpose(image_np, (1, 2, 0))

            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)

            return Image.fromarray(image_np)
        except Exception as e:
            logger.error("Error converting tensor to PIL: %s", e)
            return None

    @staticmethod
    def upload_image(image):
        """Upload image tensor to FAL and return URL."""
        if image is None:
            return None
        pil_image = ImageUtils.tensor_to_pil(image)
        if pil_image is None:
            return None

        # Save the image to a temporary file (delete=False so we control cleanup
        # after the FAL upload finishes; Windows cannot reopen an open temp file).
        fd, temp_file_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            pil_image.save(temp_file_path, format="PNG")
            client = FalConfig().get_client()
            return client.upload_file(temp_file_path)
        except Exception as e:
            logger.error("Error uploading image: %s", e)
            return None
        finally:
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

    @staticmethod
    def upload_file(file_path):
        if not file_path:
            return None
        try:
            client = FalConfig().get_client()
            return client.upload_file(file_path)
        except Exception as e:
            logger.error("Error uploading file: %s", e)
            return None

    @staticmethod
    def upload_audio(audio, suffix: str = ".wav"):
        """Upload a ComfyUI AUDIO dict ({waveform, sample_rate}) to FAL.

        Saves to a temp WAV via torchaudio, uploads, returns the FAL URL or None.
        """
        if audio is None:
            return None
        try:
            import torchaudio  # local import: torchaudio is heavy and only needed here
        except ImportError as e:
            logger.error("torchaudio is required to upload AUDIO inputs: %s", e)
            return None

        try:
            waveform = audio["waveform"]
            sample_rate = int(audio["sample_rate"])
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Unsupported AUDIO payload: %s", e)
            return None

        # Comfy AUDIO is [B, C, S]; torchaudio.save expects [C, S].
        if waveform.dim() == 3:
            waveform = waveform[0]

        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            torchaudio.save(temp_path, waveform.cpu(), sample_rate)
            return FalConfig().get_client().upload_file(temp_path)
        except Exception as e:
            logger.error("Error uploading audio: %s", e)
            return None
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @staticmethod
    def upload_video(video, suffix: str = ".mp4"):
        """Upload a ComfyUI VIDEO object to FAL.

        Uses VideoInput.save_to() to materialize the video to a temp file, then
        uploads via the FAL client. Returns the URL or None on failure.
        """
        if video is None:
            return None

        # Fast path: VideoFromFile exposes the underlying path; reuse it directly.
        existing_path = None
        for attr in ("_VideoFromFile__file", "file", "path", "_path"):
            candidate = getattr(video, attr, None)
            if isinstance(candidate, str) and os.path.isfile(candidate):
                existing_path = candidate
                break

        if existing_path is not None:
            try:
                return FalConfig().get_client().upload_file(existing_path)
            except Exception as e:
                logger.error("Error uploading video (direct path): %s", e)
                return None

        if not hasattr(video, "save_to"):
            logger.error("Unsupported VIDEO object (no save_to): %r", type(video))
            return None

        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            video.save_to(temp_path)
            return FalConfig().get_client().upload_file(temp_path)
        except Exception as e:
            logger.error("Error uploading video: %s", e)
            return None
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    @staticmethod
    def mask_to_image(mask):
        return (
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            .movedim(1, -1)
            .expand(-1, -1, -1, 3)
        )


class ResultProcessor:
    """Utility functions for processing API results."""

    @staticmethod
    def process_image_result(result):
        try:
            images = []
            for img_info in result["images"]:
                img_url = img_info["url"]
                img_response = requests.get(img_url, timeout=HTTP_TIMEOUT)
                img = Image.open(io.BytesIO(img_response.content))
                images.append(np.array(img).astype(np.float32) / 255.0)

            stacked_images = np.stack(images, axis=0)
            return (torch.from_numpy(stacked_images),)
        except Exception as e:
            logger.error("Error processing image result: %s", e)
            return ResultProcessor.create_blank_image()

    @staticmethod
    def process_single_image_result(result):
        try:
            img_url = result["image"]["url"]
            img_response = requests.get(img_url, timeout=HTTP_TIMEOUT)
            img = Image.open(io.BytesIO(img_response.content))
            img_array = np.array(img).astype(np.float32) / 255.0
            stacked_images = np.stack([img_array], axis=0)
            return (torch.from_numpy(stacked_images),)
        except Exception as e:
            logger.error("Error processing single image result: %s", e)
            return ResultProcessor.create_blank_image()

    @staticmethod
    def create_blank_image():
        blank_img = Image.new("RGB", (512, 512), color="black")
        img_array = np.array(blank_img).astype(np.float32) / 255.0
        return (torch.from_numpy(img_array)[None,],)


class ApiHandler:
    """Utility functions for API interactions."""

    # Substrings (matched case-insensitively against the error text) that mark a
    # failure as transient and worth re-submitting. `file_download_error` is the
    # big one: FAL occasionally routes a job to a worker that can't reach the
    # storage host our uploaded file lives on — a fresh submission usually lands
    # on a worker that can.
    _RETRYABLE_HINTS = (
        "file_download_error",
        "failed to download",
        "internal server error",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
        "502 bad gateway",
        "503 service",
        "504 gateway",
    )
    _RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
    # Seconds to wait before attempts 2, 3, ... (last value reused if exceeded).
    _RETRY_DELAYS = (3, 8, 15)

    @staticmethod
    def _is_retryable(exc) -> bool:
        status = getattr(exc, "status_code", None)
        if status in ApiHandler._RETRYABLE_STATUS:
            return True
        msg = str(exc).lower()
        return any(hint in msg for hint in ApiHandler._RETRYABLE_HINTS)

    @staticmethod
    def submit_and_get_result(endpoint, arguments, max_attempts: int = 3):
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                client = FalConfig().get_client()
                handler = client.submit(endpoint, arguments=arguments)
                return handler.get()
            except Exception as e:
                last_exc = e
                if attempt < max_attempts and ApiHandler._is_retryable(e):
                    delay = ApiHandler._RETRY_DELAYS[min(attempt - 1, len(ApiHandler._RETRY_DELAYS) - 1)]
                    logger.warning(
                        "FAL submit to %s failed (attempt %d/%d), retrying in %ds: %s",
                        endpoint, attempt, max_attempts, delay, e,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Error submitting to %s: %s", endpoint, e)
                raise
        # Exhausted retries on a retryable error.
        raise last_exc

    @staticmethod
    def handle_video_generation_error(model_name, error):
        logger.error("Error generating video with %s: %s", model_name, error)
        return ("Error: Unable to generate video.",)

    @staticmethod
    def handle_image_generation_error(model_name, error):
        logger.error("Error generating image with %s: %s", model_name, error)
        return ResultProcessor.create_blank_image()

    @staticmethod
    def handle_text_generation_error(model_name, error):
        logger.error("Error generating text with %s: %s", model_name, error)
        return ("Error: Unable to generate text.",)
