import json
import os
import comfy.sd
import folder_paths

configPath = f"{os.getcwd()}/custom_nodes/ComfyUI-daveand-utils/config/model-config-selector-config.json"

class ModelConfigSelector:
    CATEGORY = "utils"
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
                "clipskip": ("INT", {"default": -2, "min": -100, "max": 100}), 
                "bypass_clipskip": ("BOOLEAN", {"default": False}), 
            },
        }
    RETURN_TYPES = (folder_paths.get_filename_list("checkpoints"), "INT", "FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("ckpt_name", "steps", "cfg", "sampler", "scheduler", "clipskip", "bypass_clipskip", "config_pipe_out", "available_configs")
    FUNCTION = "load_config"
    OUTPUT_NODE = True

    def load_config(self, ckpt_name, manual_values, save_manual_values, steps, cfg, sampler, scheduler, clipskip, bypass_clipskip) :
        try:
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
                "clipskip": clipskip,
                "bypassClipSkip": bypass_clipskip
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
                        model['clipskip'] = selectedModel['clipskip']
                        model['bypassClipSkip'] = selectedModel['bypassClipSkip']
                        #print(json.dumps(models, indent=4))
                        rewrite_json(models)


            if not found:
                print(f"Model '{ckpt_name}' not found in config. Using manual values and saving to config file.")
                write_json(selectedModel)
            
            print(f"Name:\t\t\t{selectedModel['model']}")
            print(f"Steps:\t\t\t{selectedModel['steps']}")
            print(f"Cfg:\t\t\t{selectedModel['cfg']}")
            print(f"Sampler:\t\t{selectedModel['sampler']}")
            print(f"Scheduler:\t\t{selectedModel['scheduler']}")
            print(f"Clip Skip:\t\t{selectedModel['clipskip']}")
            print(f"Bypass Clip Skip:\t{selectedModel['bypassClipSkip']}")

            return (selectedModel['model'], selectedModel['steps'], selectedModel['cfg'], selectedModel['sampler'], selectedModel['scheduler'], selectedModel['clipskip'], selectedModel['bypassClipSkip'], json.dumps(selectedModel, indent=4), availableModelsStr)

        except Exception as e:
            return (f"Error: {e}",)
        
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
