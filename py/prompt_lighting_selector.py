import json
import os
import inspect

configPath = f"{os.getcwd()}/custom_nodes/ComfyUI-daveand-utils/config/prompt-lighting-selector-config.json"

class PromptLightingSelector:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        config = get_config()
        return { 
            "required":  { 
                "ambience": (config["ambience"],),
                "artificial_light_enabled": ("BOOLEAN", {"default": False}),
                "natural_light": (config["natural_light"],),
                "artificial_light": (config["artificial_light"],),
                "technique": (config["technique"],),
                "three_point_enabled": ("BOOLEAN", {"default": False}),
                "fill_light": (config["fill_light"],),
                "key_light": (config["key_light"],),
                "rim_light": (config["rim_light"],),
                "direction": (config["direction"],),
                "hardness": (config["hardness"],),
                "intensity": (config["intensity"],),
                "color_temperature": (config["color_temperature"],),
                "brightness": (config["brightness"],),
                "ambient_occlusion": ("BOOLEAN", {"default": False}),
                "specular_lighting": ("BOOLEAN", {"default": False}),
                "subsurface_scattering": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "append_text": ("STRING", {"default": "", "multiline": True}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lighting_prompt",)
    FUNCTION = "lighting_selector"
    OUTPUT_NODE = True

    def lighting_selector(
            self, 
            ambience, 
            artificial_light_enabled, 
            natural_light, 
            artificial_light,
            technique,
            three_point_enabled,
            fill_light,
            key_light,
            rim_light,
            direction,
            hardness,
            intensity,
            color_temperature,
            brightness,
            ambient_occlusion,
            specular_lighting,
            subsurface_scattering,
            append_text
        ):
        try:

            full_prompt_list = [            
                ambience, 
                natural_light, 
                artificial_light,
                technique,
                fill_light,
                key_light,
                rim_light,
                direction,
                hardness,
                intensity,
                color_temperature,
                brightness,
                append_text
            ]

            prompt_list = []

            prompt_list.append(ambience)

            if artificial_light_enabled:
                prompt_list.append(artificial_light)
                prompt_list.append(technique)
            else:
                prompt_list.append(natural_light)
            

            prompt_list.append(fill_light)
            if three_point_enabled:
                prompt_list.append(key_light)
                prompt_list.append(rim_light)
            
            prompt_list.append(direction)
            prompt_list.append(hardness)
            prompt_list.append(intensity)
            prompt_list.append(color_temperature)
            prompt_list.append(brightness)

            if ambient_occlusion:
                prompt_list.append("ambient occlusion lighting")

            if specular_lighting:
                prompt_list.append("specular lighting")

            if subsurface_scattering:
                prompt_list.append("subsurface scattering lighting")

            # if append_text is not "":
            prompt_list.append(append_text)

            # lighting_prompt_string = ", ".join(prompt_list)
            lighting_prompt_string = ""
            for p in prompt_list:
                if p is not "":
                    if lighting_prompt_string is "":
                        lighting_prompt_string = p
                    else:
                        lighting_prompt_string += ", " + p
            
            return (lighting_prompt_string,)

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

def get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

configSeed = {
    "ambience": ["", "well lit room", "white ambient light", "blue ambient light", "dark", "completely dark"],
    "natural_light": ["", "sunlight", "twilight", "moonlight", "golden hour"],
    "artificial_light": ["", "incandecent lighting", "flourencent lighting", "candle lighting", "flash photograpy", "neon lighting"],
    "technique": ["", "studio lighting", "cinematic lighting", "high contrast lighting", "High Dynamic Range lighting"],
    "fill_light": ["", "soft fill light", "hard white fill light"],
    "key_light": ["", "vibrant keylight"],
    "rim_light": ["", "blue rimlight"],
    "direction": ["", "side lighting", "top lighting"],
    "hardness": ["", "hard light", "soft light"],
    "intensity": ["", "high-key lighting", "low-key lighting"],
    "color_temperature": ["", "warm light", "cool light"],
    "brightness": ["", "bright", "dark"]
}