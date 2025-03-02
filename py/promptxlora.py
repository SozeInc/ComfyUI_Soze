import os
import glob
import random

class Soze_PromptXLora:
    def __init__(self):
        self.prompt_index = -1
        self.lora_index = 0

    @classmethod
    def IS_CHANGED(self, prompts, lora_dir, lora_extension, current_prompt_index, current_lora_index, loop_forever):
        return random.random()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "prompts": ("STRING", {"default": "", "multiline": True}),
                "lora_dir": ("STRING", {"default": ""}),
                "lora_extension": ("STRING", {"default": "safetensors"}),
        },
        "optional": {
            "loop_forever": ("BOOLEAN", {"default": False}),    
            "restart_loop": ("BOOLEAN", {"default": False}),    
        }}

    RETURN_TYPES = ("INT","INT", "INT","INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompts_count", "loras_count", "current_prompt_index", "current_lora_index", "prompt_text", "lora_filepath", "lora_filename")
    FUNCTION = "promptxlora"
    CATEGORY = "util"
    
    def promptxlora(self, prompts, lora_dir, lora_extension, loop_forever, restart_loop):
        prompts_count = len(prompts.splitlines()) 
        loras_count = len(glob.glob(os.path.join(lora_dir, '*.' + lora_extension)))
        loras_list = glob.glob(os.path.join(lora_dir, '*.' + lora_extension))
        loras_list.sort()
        lora_filepath = ""
        if (restart_loop):
            self.prompt_index = -1
            self.lora_index = 0
            restart_loop = False
            
        if (self.prompt_index < prompts_count - 1):
            self.prompt_index = self.prompt_index + 1            
        else:
            self.prompt_index = 0
            if (self.lora_index < loras_count - 1):
                self.lora_index = self.lora_index + 1
            else:
                if (loop_forever):
                    self.prompt_index = 0
                    self.lora_index = 0
                else:
                    self.prompt_index = -1
                    self.lora_index = 0
                    raise Exception("End of Loop, current indexes reset")
                
        prompt_text = prompts.splitlines()[self.prompt_index]
        lora_filepath = loras_list[self.lora_index] 
        lora_filename = os.path.basename(lora_filepath)

        return {"ui": {
            "text": [f"{prompts_count}|{loras_count}|{self.prompt_index + 1}|{self.lora_index + 1}"],}, 
            "result": (prompts_count, loras_count, self.prompt_index, self.lora_index, prompt_text, lora_filepath, lora_filename) 
        }
    