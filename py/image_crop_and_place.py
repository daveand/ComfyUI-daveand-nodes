from PIL import Image
from comfy_extras.nodes_dataset import tensor_to_pil
import torch
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
import comfy.sample
import math
from comfy_extras.nodes_dataset import tensor_to_pil, pil_to_tensor

class ImageCropAndPlace:
    CATEGORY = "utils"
    @classmethod    
    def INPUT_TYPES(s):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        return { 
            "required":  {
                "dest_image": ("IMAGE", ),
                "dest_mask": ("MASK", ),
                "crop_image": ("IMAGE", ),
                "crop_mask": ("MASK", ),
            }
        }
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK",)
    RETURN_NAMES = ("image", "placed_image", "crop_mask",)
    FUNCTION = "crop_and_place"
    OUTPUT_NODE = True

    def crop_and_place(self, dest_image, dest_mask, crop_image, crop_mask):

        dest_mask_img = mask_to_image(dest_mask)
        crop_mask_img = mask_to_image(crop_mask)

        cropped_image = cut(crop_image, crop_mask_img, 0, 0)
        cropped_image = scale_to_total_pixels(cropped_image, 0.5, 1)

        placed_image = add_image(dest_image, cropped_image)

        return (dest_image, placed_image, crop_mask)

def mask_to_image(mask):
    result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return result


def add_image(base_img, overlay_img):
    # Open the base and overlay images
    # base_img = Image.open('background.jpg')
    # overlay_img = Image.open('overlay.png')
    base_img = tensor_to_pil(base_img)
    overlay_img = tensor_to_pil(overlay_img)

    # Convert the overlay image to RGBA mode
    overlay_img = overlay_img.convert('RGBA')

    # Define the position where the overlay image will be pasted
    position = (200, 100)

    # Overlay the image over base image
    base_img.paste(overlay_img, position, overlay_img)

    base_img = pil_to_tensor(base_img)

    return base_img
    
def scale_to_total_pixels(image, megapixels, resolution_steps):
    samples = image.movedim(-1,1)
    total = megapixels * 1024 * 1024

    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
    height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

    s = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", "disabled")
    s = s.movedim(1,-1)
    return s

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def cut(image, mask, force_resize_width, force_resize_height, mask_mapping_optional = None):

    # We operate on RGBA to keep the code clean and then convert back after
    image = tensor2rgba(image)
    mask = tensor2mask(mask)


    B, H, W, _ = image.shape
    
    MB, _, _ = mask.shape

    # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
    is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, H * W]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]

    use_width = 853
    use_height = 1280

    alpha_mask = torch.ones((B, H, W, 4))
    alpha_mask[:,:,:,3] = mask

    image = image * alpha_mask

    return image