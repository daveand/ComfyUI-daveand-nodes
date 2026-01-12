import torch
import torch.nn.functional as F
import comfy.sample
import latent_preview
import json
import math
from comfy import model_management

class TiledKSamplerWithUpscaler:
    CATEGORY = "sampling"
    @classmethod    
    def INPUT_TYPES(s):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        return {
            "required": {
                "config_pipe_enabled": ("BOOLEAN", {"default": False}),
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "vae": ("VAE", {"tooltip": "The VAE used for decoding the output latent to an image."}),
                "seed": ("INT", {"default": 1984, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm", "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "exponential", "tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "image": ("IMAGE", ),
                "control_net": ("CONTROL_NET", ),
                "tile_resolution": ("INT", {"default": 1024, "min": 512, "max": 10000, "step": 64}),
                "tile_overlap": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 10.0, "step":0.001}),
                "upscale_model": ("UPSCALE_MODEL", ),
                "upscale_method": (upscale_methods, {"default": "lanczos"}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "post_process_enable": ("BOOLEAN", {"default": False}),
                "desaturate": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step":0.01}),
                "sharpen": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step":0.01}),           
            },
            "optional": {
                "config_pipe_in": ("*", ),
            },
        }    
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "config_pipe_out",)
    FUNCTION = "tiled_sampler_upscaler"
    OUTPUT_NODE = True


    def tiled_sampler_upscaler(self, config_pipe_enabled, upscale_model, upscale_method, scale_by, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, image, control_net, tile_resolution, tile_overlap, denoise=0.5, strength=0.5, config_pipe_in=None, post_process_enable=False, desaturate=0.2, sharpen=0.8):          
        model_config = json.loads(config_pipe_in) if config_pipe_in else None

        #Scale image
        image = handle_upscale_model(upscale_model, image)
        image = upscale(image, upscale_method, scale_by)


        samples = image
        untiled_image = image

        counter = 0

        
        tiles = calculate_tiles(samples, tile_resolution, tile_overlap)

        tiled_images = tile_image(samples, tiles[1], tiles[0], tile_overlap, 0, 0)

        processed_tiles = []

        for i in range(len(tiled_images[0])):
            s = tiled_images[0][i].unsqueeze(0)
        
            samples = vae_encode(vae, s)
                                
            control_net_cond = apply_controlnet(self, positive, negative, control_net, s, strength, start_percent=0.0, end_percent=1.0, vae=vae)

            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, control_net_cond[0], control_net_cond[1], samples, denoise)
            samples = vae_decode(vae, samples)
            processed_tiles.append(samples)

        processed_tiles = torch.cat(processed_tiles, dim=0)

        untiled_image = untile_image(processed_tiles, tiled_images[3], tiled_images[4], tiles[1], tiles[0])


        if post_process_enable:
            untiled_image = desaturate_image(untiled_image, desaturate)
            untiled_image = sharpen_image(untiled_image, sharpen)

        return (untiled_image, config_pipe_in,)


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return out

def vae_encode(vae, pixels):
        t = vae.encode(pixels)
        return ({"samples":t})

def vae_decode(vae, samples):
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5: #Combine batches
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae=None, extra_concat=[]):
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1,1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get('control', None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(control_hint, strength, (start_percent, end_percent), vae=vae, extra_concat=extra_concat)
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d['control'] = c_net
            d['control_apply_to_uncond'] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return (out[0], out[1])

def calculate_tiles(image, tile_resolution, tile_overlap):
    _, height, width, _ = image.shape
    tile_size = tile_resolution
    overlap = int(tile_size * tile_overlap)

    y_steps = max(1, math.ceil((height - overlap) / (tile_size - overlap)))
    x_steps = max(1, math.ceil((width - overlap) / (tile_size - overlap)))

    return (x_steps, y_steps, (x_steps * y_steps))

def tile_image(image, rows, cols, overlap, overlap_x, overlap_y):
    h, w = image.shape[1:3]
    tile_h = h // rows
    tile_w = w // cols
    h = tile_h * rows
    w = tile_w * cols
    overlap_h = int(tile_h * overlap) + overlap_y
    overlap_w = int(tile_w * overlap) + overlap_x

    # max overlap is half of the tile size
    overlap_h = min(tile_h // 2, overlap_h)
    overlap_w = min(tile_w // 2, overlap_w)

    if rows == 1:
        overlap_h = 0
    if cols == 1:
        overlap_w = 0
    
    tiles = []
    for i in range(rows):
        for j in range(cols):
            y1 = i * tile_h
            x1 = j * tile_w

            if i > 0:
                y1 -= overlap_h
            if j > 0:
                x1 -= overlap_w

            y2 = y1 + tile_h + overlap_h
            x2 = x1 + tile_w + overlap_w

            if y2 > h:
                y2 = h
                y1 = y2 - tile_h - overlap_h
            if x2 > w:
                x2 = w
                x1 = x2 - tile_w - overlap_w

            tiles.append(image[:, y1:y2, x1:x2, :])
    tiles = torch.cat(tiles, dim=0)

    return(tiles, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h,)

def untile_image(tiles, overlap_x, overlap_y, rows, cols):
    tile_h, tile_w = tiles.shape[1:3]
    tile_h -= overlap_y
    tile_w -= overlap_x
    out_w = cols * tile_w
    out_h = rows * tile_h

    out = torch.zeros((1, out_h, out_w, tiles.shape[3]), device=tiles.device, dtype=tiles.dtype)

    for i in range(rows):
        for j in range(cols):
            y1 = i * tile_h
            x1 = j * tile_w

            if i > 0:
                y1 -= overlap_y
            if j > 0:
                x1 -= overlap_x

            y2 = y1 + tile_h + overlap_y
            x2 = x1 + tile_w + overlap_x

            if y2 > out_h:
                y2 = out_h
                y1 = y2 - tile_h - overlap_y
            if x2 > out_w:
                x2 = out_w
                x1 = x2 - tile_w - overlap_x
            
            mask = torch.ones((1, tile_h+overlap_y, tile_w+overlap_x), device=tiles.device, dtype=tiles.dtype)

            # feather the overlap on top
            if i > 0:
                mask[:, :overlap_y, :] *= torch.linspace(0, 1, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
            # feather the overlap on bottom
            #if i < rows - 1:
            #    mask[:, -overlap_y:, :] *= torch.linspace(1, 0, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
            # feather the overlap on left
            if j > 0:
                mask[:, :, :overlap_x] *= torch.linspace(0, 1, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
            # feather the overlap on right
            #if j < cols - 1:
            #    mask[:, :, -overlap_x:] *= torch.linspace(1, 0, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
            
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
            tile = tiles[i * cols + j] * mask
            out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile
    return out

def desaturate_image(image, factor):
    grayscale = image.mean(dim=3)

    grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
    grayscale = torch.clamp(grayscale, 0, 1)

    return grayscale

def sharpen_image(image, amount):
    epsilon = 1e-5
    img = F.pad(image.permute([0,3,1,2]), pad=(1, 1, 1, 1))

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx + epsilon)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = output.clamp(0, 1)
    #output = torch.nan_to_num(output)

    output = output.permute([0,2,3,1])

    return output

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)

def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

def handle_upscale_model(upscale_model, image):
    device = model_management.get_torch_device()

    memory_required = model_management.module_size(upscale_model.model)
    memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
    memory_required += image.nelement() * image.element_size()
    model_management.free_memory(memory_required, device)

    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)

    tile = 512
    overlap = 32

    oom = True
    try:
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
    finally:
        upscale_model.to("cpu")

    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s

def upscale(image, upscale_method, scale_by):
    samples = image.movedim(-1,1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return s