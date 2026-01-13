from .py.prompt_constructor import PromptConstructor
from .py.prompt_camera_selector import PromptCameraSelector
from .py.prompt_lighting_selector import PromptLightingSelector
from .py.prompt_lighting_selector import PromptLightingSelector
from .py.image_analyzer import ImageAnalyzer
from .py.double_ksampler import DoubleKSampler
from .py.checkpoint_loader_with_config import CheckpointLoaderWithConfig
from .py.latent_image_and_seed import LatentImageAndSeed
from .py.cascaded_tile_upscaler import CascadedTileUpscaler
from .py.training_dataset_saver import TrainingDatasetSaver
from .py.batch_resizer import BatchResizer
from .py.image_crop_and_place import ImageCropAndPlace
from .py.tiled_ksampler import TiledKSampler
from .py.tiled_ksampler_upscaler import TiledKSamplerWithUpscaler
from .py.test1 import Test1

NODE_CLASS_MAPPINGS = {
    "PromptConstructor" : PromptConstructor,
    "PromptCameraSelector" : PromptCameraSelector,
    "PromptLightingSelector" : PromptLightingSelector,
    "ImageAnalyzer" : ImageAnalyzer,
    "DoubleKSampler" : DoubleKSampler,
    "CheckpointLoaderWithConfig" : CheckpointLoaderWithConfig,
    "LatentImageAndSeed" : LatentImageAndSeed,
    "CascadedTileUpscaler" : CascadedTileUpscaler,
    "TrainingDatasetSaver" : TrainingDatasetSaver,
    "BatchResizer" : BatchResizer,
    "ImageCropAndPlace" : ImageCropAndPlace,
    "TiledKSampler" : TiledKSampler,
    "TiledKSamplerWithUpscaler" : TiledKSamplerWithUpscaler,
    "Test1" : Test1,
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
