import json
import os
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict, FileLocator

configPath = f"{os.getcwd()}/custom_nodes/ComfyUI-daveand-utils/config/prompt-constructor-config.json"

class PromptConstructor:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        config = get_config()
        return { 
            "required":  {
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "format_output": ("BOOLEAN", {"default": True}),
            },
            "optional":  { 
                "01_subject_prompt": ("STRING", {"forceInput": True}),
                "02_subject": ("STRING", {"default": "", "multiline": True}),
                "03_description_prompt": ("STRING", {"forceInput": True}),
                "04_description": ("STRING", {"default": "", "multiline": True}),
                "05_environment_prompt": ("STRING", {"forceInput": True}),
                "06_environment": ("STRING", {"default": "", "multiline": True}),
                "07_composition_prompt": ("STRING", {"forceInput": True}),
                "08_composition": ("STRING", {"default": "", "multiline": True}),
                "09_lighting_prompt": ("STRING", {"forceInput": True}),
                "10_lighting": ("STRING", {"default": "", "multiline": True}),
                "11_camera_prompt": ("STRING", {"forceInput": True}),
                "12_camera": ("STRING", {"default": "", "multiline": True}),
                "13_suffix_prompt": ("STRING", {"forceInput": True}),
                "14_suffix": ("STRING", {"default": "", "multiline": True}),
                "15_negative_prompt": ("STRING", {"forceInput": True}),
                "16_negative": ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = (IO.CONDITIONING, IO.CONDITIONING, "STRING", "STRING",)
    RETURN_NAMES = ("positive_conditioning", "negative_conditioning", "positive_prompt", "negative_prompt",)
    FUNCTION = "build_prompt"
    OUTPUT_NODE = True

    def build_prompt(self, clip, format_output, **kwargs):
        try:
            pos_prompt_list = []
            neg_prompt_list = []
            
            for k in sorted(kwargs.keys()):
                p = kwargs[k]
                

                if isinstance(p, str):
                    if k == "15_negative_prompt" or k == "16_negative":
                        if p is not "":
                            neg_prompt_list.append(p)

                    if p is not "" and not k == "15_negative_prompt" and not k == "16_negative":
                        pos_prompt_list.append(p)

            pos_prompt_string = ", ".join(pos_prompt_list)
            neg_prompt_string = ", ".join(neg_prompt_list)

            if format_output:
                pos_prompt_string = pos_prompt_string.replace(" ", "_")
                neg_prompt_string = neg_prompt_string.replace(" ", "_")
                pos_prompt_string = pos_prompt_string.replace(",_", ", ")
                neg_prompt_string = neg_prompt_string.replace(",_", ", ")
                pos_prompt_string = pos_prompt_string.replace(":_", ": ")
                neg_prompt_string = neg_prompt_string.replace(":_", ": ")

            pos_tokens = clip.tokenize(pos_prompt_string)
            neg_tokens = clip.tokenize(neg_prompt_string)
            
            return (
                clip.encode_from_tokens_scheduled(pos_tokens), 
                clip.encode_from_tokens_scheduled(neg_tokens),
                pos_prompt_string, 
                neg_prompt_string, 
            )

        except Exception as e:
            return (f"Error: {e}",)
        

def get_config():
    if not os.path.isfile(configPath):
        rewrite_json(configSeed)
        print("Config file not found. Seeded a new one.")

    config = {}

    with open(configPath, 'r') as file:
        config = json.load(file)

    return config

def rewrite_json(data, filename=configPath):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


configSeed = {}