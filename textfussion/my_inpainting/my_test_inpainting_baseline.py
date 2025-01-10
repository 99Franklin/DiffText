import requests
import torch
from io import BytesIO
from PIL import Image

from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline

init_image = Image.open("./demo_images/demo_images_100/1/img.png").convert("RGB").resize((512, 512))
mask_image = Image.open("./demo_images/demo_images_100/1/mask.png").convert("RGB").resize((512, 512))

# init_image = Image.open("./demo_images/img_street.png").convert("RGB").resize((512, 512))
# mask_image = Image.open("./demo_images/mask_street.png").convert("RGB").resize((512, 512))

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    # "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
# prompt = "A road sign standing by the street, displaying the text 'STOP'."
# prompt = "A bunch of flowers with a card that says 'happy birthday'"
# prompt = "fill the region with the text 'Hello'"
prompt = "fill the region with the text 'Hello', and keep the background unchanged"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

image.save("demo/try.png")
