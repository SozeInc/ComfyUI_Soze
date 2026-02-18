# import torch
# import torchaudio
# import os
# import folder_paths
# from sam_audio import SAMAudio, SAMAudioProcessor

# # Cache for model and processor to avoid reloading
# _SAM_AUDIO_MODEL = None
# _SAM_AUDIO_PROCESSOR = None
# _CURRENT_MODEL_NAME = None

# class Soze_SAMAudioTextPrompt:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "input_audio": ("STRING", {"default": "", "multiline": False}),
#                 "prompt": ("STRING", {"default": "", "multiline": True}),
#                 "model_version": (["facebook/sam-audio-small", "facebook/sam-audio-base", "facebook/sam-audio-large"], {"default": "facebook/sam-audio-large"}),
#                 "predict_spans": ("BOOLEAN", {"default": False}),
#                 "reranking_candidates": ("INT", {"default": 1, "min": 1, "max": 10}),
#             }
#         }

#     RETURN_NAMES = ("isolated_audio", "remaining_audio")
#     RETURN_TYPES = ("AUDIO", "AUDIO")
#     FUNCTION = "sam_audio_text_prompt"
#     CATEGORY = "utils"
#     OUTPUT_NODE = True

#     def sam_audio_text_prompt(self, input_audio, prompt, model_version, predict_spans, reranking_candidates):
#         global _SAM_AUDIO_MODEL, _SAM_AUDIO_PROCESSOR, _CURRENT_MODEL_NAME

#         if not os.path.exists(input_audio):
#             raise FileNotFoundError(f"Audio file not found: {input_audio}")

#         # Define model cache directory
#         models_dir = folder_paths.models_dir
#         sam_audio_dir = os.path.join(models_dir, "sam_audio")
        
#         if not os.path.exists(sam_audio_dir):
#             os.makedirs(sam_audio_dir)

#         # Load model if not loaded or if model version changed
#         if _SAM_AUDIO_MODEL is None or _CURRENT_MODEL_NAME != model_version:
#             print(f"Loading SAM Audio model: {model_version}")
#             _SAM_AUDIO_MODEL = SAMAudio.from_pretrained(model_version, cache_dir=sam_audio_dir)
#             _SAM_AUDIO_PROCESSOR = SAMAudioProcessor.from_pretrained(model_version, cache_dir=sam_audio_dir)
#             _CURRENT_MODEL_NAME = model_version
            
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             _SAM_AUDIO_MODEL = _SAM_AUDIO_MODEL.eval().to(device)

#         device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Process input
#         # processor handles loading audio from path if it's a string
#         batch = _SAM_AUDIO_PROCESSOR(
#             audios=[input_audio],
#             descriptions=[prompt],
#         ).to(device)

#         # Inference
#         with torch.inference_mode():
#             result = _SAM_AUDIO_MODEL.separate(
#                 batch, 
#                 predict_spans=predict_spans, 
#                 reranking_candidates=reranking_candidates
#             )

#         # Result structure depends on batch size, here we have 1
#         # result.target and result.residual are likely tensors
        
#         target_audio = result.target.cpu()
#         residual_audio = result.residual.cpu()
        
#         # Get sample rate from processor
#         sample_rate = _SAM_AUDIO_PROCESSOR.audio_sampling_rate

#         # Format for ComfyUI AUDIO type
#         # Usually: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
#         # sam-audio output might be [1, samples] or [1, channels, samples]
        
#         # Ensure 3D tensor [batch, channels, samples]
#         if target_audio.dim() == 2:
#             target_audio = target_audio.unsqueeze(0)
#         if residual_audio.dim() == 2:
#             residual_audio = residual_audio.unsqueeze(0)
            
#         # If it's just [samples], make it [1, 1, samples]
#         if target_audio.dim() == 1:
#             target_audio = target_audio.unsqueeze(0).unsqueeze(0)
#         if residual_audio.dim() == 1:
#             residual_audio = residual_audio.unsqueeze(0).unsqueeze(0)

#         isolated_output = {"waveform": target_audio, "sample_rate": sample_rate}
#         remaining_output = {"waveform": residual_audio, "sample_rate": sample_rate}

#         return (isolated_output, remaining_output)
    