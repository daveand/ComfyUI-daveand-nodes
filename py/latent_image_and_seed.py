import torch
import comfy.sample
import latent_preview
import json
import math
import random

class LatentImageAndSeed:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
        return {
            "required": {
                "config_pipe_enabled": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 1216, "min": 16, "max": 10000, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 832, "min": 16, "max": 10000, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "orientation": (["Landscape", "Portrait", "Square"], {"default": "Landscape", "tooltip": "The orientation of the latent images."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                "fixed_seed_enabled": ("BOOLEAN", {"default": False}),
                "fixed_seed": (
                    "INT",
                    {
                        "default": 1984,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                    },
                ),

            },
            "optional": {
                "config_pipe_in": ("*", ),
            },
        }    
    RETURN_TYPES = ("LATENT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "seed", "seed_string", "config_pipe_out")
    FUNCTION = "latent_and_seed"
    OUTPUT_NODE = True

    def IS_CHANGED(**kwargs):
        random.seed()
        return float("NaN")

    def latent_and_seed(self, config_pipe_enabled, width, height, orientation, batch_size, fixed_seed_enabled, fixed_seed, config_pipe_in=None):          
        model_config = json.loads(config_pipe_in) if config_pipe_in else None

        if config_pipe_enabled and model_config:
            width = model_config.get("latent_width", width)
            height = model_config.get("latent_height", height)

        if orientation == "Landscape" and width < height:
            width, height = height, width
        elif orientation == "Portrait" and height < width:
            width, height = height, width
        elif orientation == "Square":
            size = min(width, height)
            width = size
            height = size

        latent = generate_latent(self, width, height, batch_size)
        seed = seed_generator(fixed_seed_enabled, fixed_seed)

        return (latent, seed, str(seed), config_pipe_in,)

def generate_latent(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return {"samples":latent}


def seed_generator(fixed_seed_enabled, fixed_seed):
    if not fixed_seed_enabled:
        random_seed = math.floor(random.random() * 10000000000000000)
        return random_seed
    else:
        return fixed_seed