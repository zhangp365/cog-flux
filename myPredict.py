from pathlib import Path
from typing import List

from cog import BasePredictor, Input, Path 
from predict import Predictor, Inputs
from bfl_predictor import BflReduxPredictor
from predict import FLUX_DEV

class DevReduxLoraPredictor(Predictor):
    def setup(self):
        self.base_setup()
        self.model = BflReduxPredictor(FLUX_DEV, 
                                       restore_lora_from_cloned_weights=True,
                                       offload=self.should_offload())
        

    def predict(
        self,
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
        prompt = ""
        self.model.handle_loras(lora_weights, lora_scale)

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