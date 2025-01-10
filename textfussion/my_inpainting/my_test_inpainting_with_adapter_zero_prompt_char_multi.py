import os
import PIL
import ipdb
import torch
import shutil
import random
from tqdm import tqdm

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline

from src.models.adapter import Adapter
from src.models.unet_2d_with_adapter import UNet2DConditionWithAdapterModel
from src.pipelines.stable_diffusion_inpainting_with_adapter import StableDiffusionInpaintWithAdapterPipeline
from src.pipelines.stable_diffusion_inpainting_with_adapter_zero_prompt import StableDiffusionInpaintWithAdapterZeroPromptPipeline


# input_data_dir = "./demo_images/demo_images_char_10"
input_data_dir = "./demo_images/demo_images_100"
# output_data_dir = "./demo/inpainting_with_adapter_char_text_guide_zero_prompt_epoch_4e_lr_1e-6_10"
output_data_dir = "./demo/inpainting_with_adapter_char_text_guide_zero_prompt_epoch_4e_lr_1e-6_100_2"

checkpoint_dir = "./output/0325_inpainting_with_adapter_char_text_guide_zero_prompt_epoch_4e_lr_1e-6"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, img_index, prompt_text, adapter=None):

    random_index = random.randint(0, len(prompt_text) - 1)

    prompt_text = prompt_text[random_index]

    img_path = os.path.join(input_data_dir, str(img_index), "img.png")
    mask_path = os.path.join(input_data_dir, str(img_index), "mask.png")

    init_image = PIL.Image.open(img_path).convert("RGB").resize((512, 512))
    mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))
    mask_viz_image = PIL.Image.open(os.path.join(input_data_dir, str(img_index), "mask_viz.png")).convert("RGB").resize((512, 512))
    init_image.save(os.path.join(output_data_dir, str(img_index) + ".png"))
    mask_viz_image.save(os.path.join(output_data_dir, str(img_index) + "_mask_viz.png"))

    # prompt = "fill the region with the text '" + prompt_text + "'"
    # prompt = "fill the region with the text '" + prompt_text + "', and keep the background unchanged."
    prompt = "Fill the region, and keep the background unchanged."

    image = pipe(prompt=prompt, text=prompt_text, image=init_image, mask_image=mask_image, adapter=adapter).images[0]

    res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
    image.save(res_path)


if __name__ == '__main__':
    # with open("./demo_images/demo_images_char_10_label.txt", "r") as f:
    with open("./demo_images/demo_images_100_label.txt", "r") as f:
        data = f.read().splitlines()
    input_list = []

    for i, string in enumerate(data):
        input_list.append([i + 1, string.split(" ")[-1]])

    # pipe = StableDiffusionInpaintWithAdapterPipeline.from_pretrained(
    #     checkpoint_dir,
    #     torch_dtype=torch.float16,
    # )

    MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", revision=None)
    unet = UNet2DConditionWithAdapterModel.from_pretrained(
        MODEL_NAME, subfolder="unet", revision=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", revision=None
    )

    pipe = StableDiffusionInpaintWithAdapterZeroPromptPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler, None)
    pipe = pipe.to("cuda")
    adapter = Adapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    adapter.load_state_dict(torch.load(os.path.join(checkpoint_dir, "adapter.pth")))
    adapter = adapter.to("cuda")

    print("model loaded!")

    for item in input_list:
        image_inpainting(pipe, item[0], item[1], adapter=adapter)
