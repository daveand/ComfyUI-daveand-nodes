import torch
import comfy.sample
import latent_preview
import json

class DoubleKSampler:
    CATEGORY = "sampling"
    @classmethod    
    def INPUT_TYPES(s):
        upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
        return {
            "required": {
                "config_pipe_enabled": ("BOOLEAN", {"default": False}),
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
                "second_sampler_enabled": ("BOOLEAN", {"default": True}),
                "latent_upscale_method": (upscale_methods, {"default": "nearest-exact"}),
                "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),
                "denoise_2": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "second_steps_offset": ("INT", {"default": 0, "min": -1000, "max": 1000}),
                "second_cfg_offset": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step":0.1, "round": 0.01}),
            },
            "optional": {
                "config_pipe_in": ("*", ),
            },
        }    
    RETURN_TYPES = ("LATENT", "LATENT", "STRING",)
    RETURN_NAMES = ("latent_1", "latent_2", "config_pipe_out",)
    FUNCTION = "double_sampler"
    OUTPUT_NODE = True


    def double_sampler(self, config_pipe_enabled, second_sampler_enabled, second_cfg_offset, latent_upscale_method, scale_by, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise_1=1.0, denoise_2=0.6, second_steps_offset=0, config_pipe_in=None):          
        model_config = json.loads(config_pipe_in) if config_pipe_in else None

        if config_pipe_enabled and model_config:
            steps = model_config.get("steps", steps)
            cfg = model_config.get("cfg", cfg)
            sampler_name = model_config.get("sampler", sampler_name)
            scheduler = model_config.get("scheduler", scheduler)

        samples1 = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise_1)
        
        samples2 = samples1

        if second_sampler_enabled:
            scaled_latent = latent_upscale(samples1, latent_upscale_method, scale_by)
            samples2 = common_ksampler(model, seed, steps + second_steps_offset, cfg + second_cfg_offset, sampler_name, scheduler, positive, negative, scaled_latent, denoise=denoise_2)

        return (samples1, samples2, config_pipe_in,)


def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    print("Sampler parameters:")
    print(steps, cfg, sampler_name, scheduler)

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

def latent_upscale(samples, upscale_method, scale_by):
    s = samples.copy()
    width = round(samples["samples"].shape[-1] * scale_by)
    height = round(samples["samples"].shape[-2] * scale_by)
    s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
    return s