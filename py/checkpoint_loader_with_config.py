import json
import os
import comfy.sd
import folder_paths

configPath = f"{os.getcwd()}/custom_nodes/ComfyUI-daveand-utils/config/checkpoint-loader-config.json"

class CheckpointLoaderWithConfig:
    CATEGORY = "loaders"
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required":  { 
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),), 
                "manual_values": ("BOOLEAN", {"default": False}),
                "save_manual_values": ("BOOLEAN", {"default": False}), 
                "steps": ("INT", {"default": 30, "min": 0, "max": 100}), 
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0}), 
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m_sde"}), 
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}), 
                "enable_clipskip": ("BOOLEAN", {"default": True}), 
                "clipskip": ("INT", {"default": -2, "min": -100, "max": 100}), 
                "latent_width": ("INT", {"default": 1216, "min": 16, "max": 10000, "step": 8}),
                "latent_height": ("INT", {"default": 832, "min": 16, "max": 10000, "step": 8}),
            },
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "INT" "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "steps", "cfg", "sampler", "scheduler", "latent_width", "latent_height", "config_pipe_out", "available_configs")
    FUNCTION = "load_checkpoint_and_config"
    OUTPUT_NODE = True

    def load_checkpoint_and_config(self, ckpt_name, manual_values, save_manual_values, steps, cfg, sampler, scheduler, clipskip, enable_clipskip, latent_width, latent_height):
        
        print(f"Current directory: {os.getcwd()}")

        if not os.path.isfile(configPath):
            emptyJson = []
            rewrite_json(emptyJson)

        with open(configPath, 'r') as file:
            models = json.load(file)

        selectedModel = {
            "model": ckpt_name,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "enable_clipskip": enable_clipskip,
            "clipskip": clipskip,
            "latent_width": latent_width,
            "latent_height": latent_height,
        }

        found = False
        availableModelsStr = ""

        for model in models:
            availableModelsStr += f"{model['model']}\n"
            if not found and model['model'] == ckpt_name:
                # print(model)
                found = True
                if not manual_values:
                    selectedModel = model
                if manual_values and save_manual_values:
                    model['steps'] = selectedModel['steps']
                    model['cfg'] = selectedModel['cfg']
                    model['sampler'] = selectedModel['sampler']
                    model['scheduler'] = selectedModel['scheduler']
                    model['enable_clipskip'] = selectedModel['enable_clipskip']
                    model['clipskip'] = selectedModel['clipskip']
                    model['latent_width'] = selectedModel['latent_width']
                    model['latent_height'] = selectedModel['latent_height']
                    rewrite_json(models)


        if not found:
            print(f"Model '{ckpt_name}' not found in config. Using manual values and saving to config file.")
            write_json(selectedModel)
        
        print(f"Name:\t\t\t{selectedModel['model']}")
        print(f"Steps:\t\t\t{selectedModel['steps']}")
        print(f"Cfg:\t\t\t{selectedModel['cfg']}")
        print(f"Sampler:\t\t{selectedModel['sampler']}")
        print(f"Scheduler:\t\t{selectedModel['scheduler']}")
        print(f"Enable clip skip:\t{selectedModel['enable_clipskip']}")
        print(f"Clip Skip:\t\t{selectedModel['clipskip']}")
        print(f"Latent width:\t\t{selectedModel['latent_width']}")
        print(f"Latent height:\t\t{selectedModel['latent_height']}")

        model = load_checkpoint(selectedModel['model'], enable_clipskip, clipskip)

        return (model[0], model[1], model[2], selectedModel['steps'], selectedModel['cfg'], selectedModel['sampler'], selectedModel['scheduler'], selectedModel['latent_width'], selectedModel['latent_height'], json.dumps(selectedModel, indent=4), availableModelsStr)


        
def write_json(new_data, filename=configPath):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data.append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent=4)

def rewrite_json(data, filename=configPath):
    print("Updating config with manual values")
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def load_checkpoint(ckpt_name, enable_clipskip, stop_at_clip_layer):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        if enable_clipskip:
            clip = out[1].clone()
            clip.clip_layer(stop_at_clip_layer)
            out = (out[0], clip, out[2])

        
        return out