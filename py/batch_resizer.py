import os
import numpy as np
import torch
import node_helpers
import math
from PIL import Image, ImageSequence, ImageOps
from comfy_extras.nodes_dataset import tensor_to_pil, pil_to_tensor
import comfy.sample
from server import PromptServer
import nodes

class BatchResizer:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        return { 
            "required":  {
                "image_folder": ("STRING", {"default": "", "multiline": False}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "base_width": ("INT", {"default": 1280, "min": 0, "max": 10000 }),
                "square_image": ("BOOLEAN", {"default": False}),
                "crop_center": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "filename", )
    FUNCTION = "batch_resize"
    OUTPUT_NODE = True

    def batch_resize(self, start_index, image_folder, base_width, square_image, crop_center):

        file_list = get_file_list(image_folder)

        
        if start_index < len(file_list):
            print(f"Image: {start_index + 1} of {len(file_list)}")
            file_path = os.path.join(image_folder, file_list[start_index])
            image = node_helpers.pillow(Image.open, file_path)

            if not square_image:
                is_landscape = image.width >= image.height

                if is_landscape:
                    wpercent = (base_width / float(image.size[0]))
                    hsize = int((float(image.size[1]) * float(wpercent)))
                    image = image.resize((base_width, hsize), Image.Resampling.LANCZOS)
                else:
                    wpercent = (base_width / float(image.size[1]))
                    wsize = int((float(image.size[0]) * float(wpercent)))
                    image = image.resize((wsize, base_width), Image.Resampling.LANCZOS)

                image = image.convert("RGB")

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
        else:
            print("Index exceeded number of images. Stopping execution.")
            nodes.interrupt_processing(True)
            return (None, None, )

        return (image, str.split(file_list[start_index], ".")[0], )


def get_file_list(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def resize_image(img, base_width, is_landscape):
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((base_width, hsize), Image.Resampling.LANCZOS)

