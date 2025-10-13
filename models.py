# models.py

import torch
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipelineLegacy
from PIL import Image

# try:
#     from transformers import PreTrainedModel
# except ImportError:  # transformers is optional until image generation runs
#     PreTrainedModel = None
# else:
#     if not getattr(PreTrainedModel, "_patched_for_offload_state_dict", False):
#         _original_from_pretrained = PreTrainedModel.from_pretrained.__func__

#         def _from_pretrained_without_offload(
#             cls,
#             pretrained_model_name_or_path,
#             *model_args,
#             **kwargs,
#         ):
#             kwargs.pop("offload_state_dict", None)
#             return _original_from_pretrained(
#                 cls, pretrained_model_name_or_path, *model_args, **kwargs
#             )

#         PreTrainedModel.from_pretrained = classmethod(_from_pretrained_without_offload)
#         PreTrainedModel._patched_for_offload_state_dict = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model() -> StableDiffusionInpaintPipelineLegacy:
    pipe = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    return pipe


def generate_image(
    pipe: StableDiffusionInpaintPipelineLegacy, prompt: str
) -> Image.Image:
    output = pipe(prompt, num_inference_steps=10).images[0]
    return output
