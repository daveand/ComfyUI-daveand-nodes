import os
import numpy as np
import json
from PIL import Image

class TrainingDatasetSaver:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required":  {
                "cropped_image": ("IMAGE", ),
                "masked_image": ("IMAGE", ),
                "caption": ("STRING", {"default": "", "multiline": False}),
                "base_folder": ("STRING", {"default": "", "multiline": False}),
                "file_name": ("STRING", {"default": "", "multiline": False}),
                "save_caption": ("BOOLEAN", {"default": False}),
                "save_mask": ("BOOLEAN", {"default": False}),
                "append_to_invoke_jsonl": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("config_file", )
    FUNCTION = "training_dataset_saver"
    OUTPUT_NODE = True

    def training_dataset_saver(self, cropped_image, masked_image, caption, base_folder, file_name, save_caption, save_mask, append_to_invoke_jsonl):

        invoke_file_path = f"{base_folder}invoke-config.jsonl"

        for (batch_number, image) in enumerate(cropped_image):
            save_image(image, f"{base_folder}cropped", file_name)

        mask_name = ""
        
        if save_mask:
            mask_name = f"{base_folder}masks/{file_name}.png"
            for (batch_number, image) in enumerate(masked_image):
                save_image(image, f"{base_folder}masks", file_name)

        if save_caption:
            save_text_file(f"{base_folder}cropped", file_name, caption)

        if append_to_invoke_jsonl:
            json = {
                "image": f"{base_folder}cropped/{file_name}.png",
                "text": caption,
                "mask": mask_name
            }

            json_str = str(json) + "\n"
            json_str = json_str.replace("'", '"')

            if not os.path.isfile(invoke_file_path):
                new_json(json_str, invoke_file_path)
            else:
                append_json(json_str, invoke_file_path)


        config_file = open_file(invoke_file_path) if os.path.isfile(invoke_file_path) else ""

        return (str(config_file), )


def save_image(image, path, file_name):
    os.makedirs(path, exist_ok=True)
    i = 255. * image.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    file = f"{file_name}.png"
    img.save(os.path.join(path, file), compress_level=4)

def save_text_file(path, file_name, caption):
    file = f"{file_name}.txt"
    with open(os.path.join(path, file), 'w') as file:
        file.write(caption)

def append_json(data, filename):
    with open(filename, 'a') as file:
        file.writelines(data)


def new_json(data, filename):
    with open(filename, 'w') as file:
        file.write(data)
        
def open_file(path):
    with open(path, 'r') as file:
        return file.read()
    