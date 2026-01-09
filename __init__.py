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
}
