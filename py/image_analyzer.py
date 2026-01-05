import os
import inspect
from PIL import Image, ImageStat
import numpy as np
import colour
from comfy_extras.nodes_dataset import tensor_to_pil

class ImageAnalyzer:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required":  {
                "image": ("IMAGE", ),
                "enable_color_temp": ("BOOLEAN", {"default": True}), 
                "temp_weight_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
                "override_color": ("BOOLEAN", {"default": False}),
                "color": ("STRING", {"default": "white", "multiline": False}),
                "enable_brightness": ("BOOLEAN", {"default": True}), 
                "brightness_weight_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("STRING", "*", "*")
    RETURN_NAMES = ("prompt_string", "color_temperature", "brightness",)
    FUNCTION = "image_analyzer"
    OUTPUT_NODE = True

    def image_analyzer(self, image, enable_color_temp, temp_weight_shift, override_color, color, enable_brightness, brightness_weight_shift):
        try:

            prompt_string = ""
            color_temp_prompt = ""
            brightness_prompt = ""
            curr_color = ""

            if override_color:
                curr_color = color + " "

            color_temp = ""

            pil_img = tensor_to_pil(image)

            if enable_color_temp:
                temp = analyze_color_temp(pil_img)

                if temp >= 5000:
                    color_temp = f"cool {curr_color}color temperature {temp}K"
                    rescaled_value = rescale_number(temp, 5000, 10000, 1, 1.5)
                else:
                    color_temp = f"warm {curr_color}color temperature {temp}K"
                    rescaled_value = rescale_number(temp, 5000, 1000, 1, 1.5)

                rescaled_value += temp_weight_shift

                color_temp_prompt = f"({color_temp}:{rescaled_value:.1f})"
                prompt_string += color_temp_prompt

            
            if enable_brightness:

                brightness = analyze_brightness(pil_img)
                brightness_str = ""
                

                if brightness >= 128:
                    brightness_str = f"brightness"
                    rescaled_value = rescale_number(brightness, 128, 255, 1, 1.5)
                else:
                    brightness_str = f"darkness"
                    rescaled_value = rescale_number(brightness, 128, 0, 1, 1.5)

                rescaled_value += brightness_weight_shift

                brightness_prompt = f"({brightness_str}:{rescaled_value:.1f})"

                if enable_color_temp:
                    prompt_string += ", "
                    
                prompt_string += f"{brightness_prompt}"

            return (prompt_string, color_temp_prompt, brightness_prompt)

        except Exception as e:
            return (f"Error: {e}",)


def analyze_color_temp(image):
    img = np.array(image)[:, :, :3]  # Remove alpha if present

    RGB = img.astype(np.float64) / 255.0

    XYZ = colour.sRGB_to_XYZ(RGB / 255)

    xy = colour.XYZ_to_xy(XYZ)

    CCT = colour.xy_to_CCT(xy, 'hernandez1999')

    mean_list = []

    for i in CCT:
        for j in i:
            if j < 10000.0:
                mean_list.append(j)

    average_CCT = np.mean(mean_list)

    return int(average_CCT)


def analyze_brightness( im_file ):
   im = im_file.convert('L')
   stat = ImageStat.Stat(im)
   return stat.rms[0]

def rescale_number(value, original_min, original_max, new_min, new_max):
    return ((value - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min
