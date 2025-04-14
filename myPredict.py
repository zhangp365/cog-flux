from pathlib import Path
from typing import List

from cog import BasePredictor, Input, Path 
from predict import Predictor, Inputs
from bfl_predictor import BflReduxPredictor, BflBf16Predictor
from predict import FLUX_DEV, LoadedModels, WeightsDownloadCache
from flux.sampling import (
    prepare_redux,
)
from flux.util import (
    load_redux,
)

class BflReduxLoraPredictor(BflBf16Predictor):
    """
    Works for dev and schnell.
    To use, pass path to redux image into predict as a prepare_kwargs - e.g.:
        redux_predictor.predict(..., prepare_kwargs={"redux_img_path": redux_image}
    """

    def __init__(
        self,
        flow_model_name: str,
        loaded_models: LoadedModels | None = None,
        device: str = "cuda",
        offload: bool = False,
        weights_download_cache: WeightsDownloadCache | None = None,
        restore_lora_from_cloned_weights: bool = False,
    ):
        super().__init__(
            flow_model_name,
            loaded_models,
            device=device,
            offload=offload,
            weights_download_cache=weights_download_cache,
            restore_lora_from_cloned_weights=restore_lora_from_cloned_weights,
        )
        self.redux_image_encoder = load_redux(device="cuda")

    def prepare(self, x, prompt, redux_img_path=None):
        """Overrides prepare in order to properly preprocess redux image"""
        return prepare_redux(
            self.t5,
            self.clip,
            x,
            prompt=prompt,
            encoder=self.redux_image_encoder,
            img_cond_path=redux_img_path,
        )


class DevReduxLoraPredictor(Predictor):
    def setup(self):
        self.base_setup()
        cache = WeightsDownloadCache()
        self.model = BflReduxLoraPredictor(FLUX_DEV, 
                                       restore_lora_from_cloned_weights=True,
                                       weights_download_cache=cache,
                                       offload=self.should_offload())
        

    def predict(
        self,
        prompt: str = Inputs.prompt,
        redux_image: Path = Input(
            description="Input image to condition your output on. This replaces prompt for FLUX.1 Redux models",
        ),
        aspect_ratio: str = Inputs.aspect_ratio,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended range is 28-50",
            ge=1,
            le=50,
            default=28,
        ),
        guidance: float = Input(
            description="Guidance for generated image", ge=0, le=10, default=3
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        megapixels: str = Inputs.megapixels,
        lora_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
    ) -> List[Path]:
        self.model.handle_loras(lora_weights, lora_scale, self.model.device)

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.model.predict(
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            prepare_kwargs={"redux_img_path": redux_image},
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )