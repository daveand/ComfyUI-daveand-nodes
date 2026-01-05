import json
import os

configPath = f"{os.getcwd()}/custom_nodes/ComfyUI-daveand-nodes/config/prompt-camera-selector-config.json"

class PromptCameraSelector:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        config = get_config()
        return { 
            "required":  { 
                "camera": (config["camera"],),
                "focal_length": (config["focal_length"],),
                "aperture": (config["aperture"],),
                "shutter_speed": (config["shutter_speed"],), 
                "iso": (config["iso"],),
            },
            "optional": {
                "append_text": ("STRING", {"default": "", "multiline": True}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("camera_prompt",)
    FUNCTION = "camera_selector"
    OUTPUT_NODE = True

    def camera_selector(self, camera, focal_length, aperture, shutter_speed, iso, append_text):
        try:
            cameraPrompt = f"Shot on a {camera}, {focal_length} lens, {aperture} aperture, {shutter_speed} sec shutter speed, ISO {iso}"
            if append_text is not "":
                cameraPrompt += f", {append_text}"
            
            return (cameraPrompt,)

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


configSeed = {
    "camera": [
        "DSLR",
        "Canon EOS 5D Mark IV",
        "Canon AE-1 Program",
        "Nikon Z9",
        "Sony A7 III"
    ],
    "focal_length": [
        "10mm fisheye",
        "16mm",
        "36mm",
        "50mm",
        "85mm",
        "200mm"
    ],
    "aperture": [
        "f/1.4",
        "f/2.8",
        "f/4",
        "f/5.6",
        "f/11"
    ],
    "shutter_speed": [
        "1/30", 
        "1/800", 
        "1/4000"
    ],
    "iso": [
        "100", 
        "200", 
        "400", 
        "800", 
        "1600"
    ]
}