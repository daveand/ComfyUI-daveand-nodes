from PIL import Image, ImageFilter, ImageOps
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
                "scale_adjust": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1 }),
                "x_adjust": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 10}),
                "y_adjust": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 10}),
                "rotate_degrees": ("INT", {"default": 0, "min": -360, "max": 360}),
                "mirror": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("placed_image", "placed_mask", )
    FUNCTION = "crop_and_place"
    OUTPUT_NODE = True

    def crop_and_place(self, dest_image, dest_mask, crop_image, crop_mask, scale_adjust, x_adjust, y_adjust, rotate_degrees, mirror):

        crop_mask_img = mask_to_image(crop_mask)

        x, y, w, h = find_mask_area(dest_mask)
        #print(x, y, w, h)
        
        rotate = False
        if rotate_degrees != 0:
            rotate = True

        cropped_image, alpha_mask = cut(crop_image, crop_mask_img)
        cropped_image = scale_to_total_pixels(cropped_image, int(w * scale_adjust))
        alpha_mask = scale_to_total_pixels(alpha_mask, int(w * scale_adjust))

        x_pos = int(x + (w / 2) - (cropped_image.shape[2] / 2)) + x_adjust
        y_pos = int(y + (h / 2) - (cropped_image.shape[1] / 2)) + y_adjust
        
        placed_image = add_image(dest_image, cropped_image, x_pos, y_pos, rotate, rotate_degrees, mirror)


        # print(dest_image.shape)
        black_image = torch.zeros(1, dest_image.shape[1], dest_image.shape[2], 3, dtype=torch.uint8)
        placed_mask_img = add_image(black_image, alpha_mask, x_pos, y_pos, rotate, rotate_degrees, mirror)

        placed_mask = image_to_mask(placed_mask_img, "red")

        return (placed_image, placed_mask)

def mask_to_image(mask):
    result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return result

def image_to_mask(image, channel):
    channels = ["red", "green", "blue", "alpha"]
    mask = image[:, :, :, channels.index(channel)]
    return mask


def find_mask_area(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    return x, y, w, h


def add_image(base_img, overlay_img, pos_x, pos_y, rotate, rotate_degrees, mirror):
    # Open the base and overlay images
    # base_img = Image.open('background.jpg')
    # overlay_img = Image.open('overlay.png')
    base_img = tensor_to_pil(base_img)
    overlay_img = tensor_to_pil(overlay_img)


    # Convert the overlay image to RGBA mode
    overlay_img = overlay_img.convert('RGBA')

    if mirror:
        overlay_img = ImageOps.mirror(overlay_img)

    if rotate:
        overlay_img = overlay_img.rotate(rotate_degrees, expand=True)

    # Define the position where the overlay image will be pasted
    position = (pos_x, pos_y)

    # Overlay the image over base image
    base_img.paste(overlay_img, position, overlay_img)

    base_img = pil_to_tensor(base_img)

    return base_img
    
def scale_to_total_pixels(image, base_width):
    samples = image.movedim(-1,1)
    # total = megapixels * 1024 * 1024

    print(image.shape)
    print(image.shape[2])
    
    image_width = image.shape[2]
    image_height = image.shape[1]
    
    wpercent = (base_width / float(image_width))
    hsize = int((float(image_height) * float(wpercent)))


    # scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
    # width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
    # height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

    s = comfy.utils.common_upscale(samples, int(base_width), int(hsize), "lanczos", "disabled")
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


def cut(image, mask):

    # We operate on RGBA to keep the code clean and then convert back after
    image = tensor2rgba(image)
    mask = tensor2mask(mask)

    # if mask_mapping_optional is not None:
    #     image = image[mask_mapping_optional]

    # Scale the mask to be a matching size if it isn't
    B, H, W, _ = image.shape
    # mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
    MB, _, _ = mask.shape

    # if MB < B:
    #     assert(B % MB == 0)
    #     mask = mask.repeat(B // MB, 1, 1)

    # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
    is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, H * W]), dim=1).values, 0.)
    mask[is_empty,0,0] = 1.
    boxes = masks_to_boxes(mask)
    mask[is_empty,0,0] = 0.

    min_x = boxes[:,0]
    min_y = boxes[:,1]
    max_x = boxes[:,2]
    max_y = boxes[:,3]

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    use_width = int(torch.max(width).item())
    use_height = int(torch.max(height).item())

    # if force_resize_width > 0:
    #     use_width = force_resize_width

    # if force_resize_height > 0:
    #     use_height = force_resize_height

    alpha_mask = torch.ones((B, H, W, 4))
    alpha_mask[:,:,:,3] = mask

    image = image * alpha_mask

    result = torch.zeros((B, use_height, use_width, 4))
    result_mask = torch.zeros((B, use_height, use_width, 4))
    for i in range(0, B):
        if not is_empty[i]:
            ymin = int(min_y[i].item())
            ymax = int(max_y[i].item())
            xmin = int(min_x[i].item())
            xmax = int(max_x[i].item())
            single = (image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
            resized = torch.nn.functional.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
            single_mask = (alpha_mask[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
            resized_mask = torch.nn.functional.interpolate(single_mask.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
            
            result[i] = resized[0]
            result_mask[i] = resized_mask[0]


    return result, result_mask