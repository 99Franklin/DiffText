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

from src.dataset.text_mapper import TextMapper
from src.dataset.crop_image_for_test import crop_image_for_test, get_rotated_rect_mask
from src.models.adapter_with_pre_prompt import WithPrePromptAdapter
from src.models.unet_2d_with_adapter import UNet2DConditionWithAdapterModel
from src.pipelines.stable_diffusion_inpainting_with_adapter_with_fussion_te \
    import StableDiffusionInpaintWithAdapterWithFussionTEPipeline


# input_data_dir = "./demo_images/demo_images_char_10"
input_data_dir = "./demo_images/demo_images_100"
# output_data_dir = "./demo/inpainting_data_v3_with_adapter_with_2_encoder_pre_prompt_crop_for_test"
output_data_dir = "./demo/try_refine_mask"

checkpoint_dir = "./output/0408_inpainting_with_adapter_with_pre_prompt"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, img_index, prompt_text, adapter=None, text_mapper=None):

    img_path = os.path.join(input_data_dir, str(img_index), "img.png")
    mask_path = os.path.join(input_data_dir, str(img_index), "mask.png")

    # init_image = PIL.Image.open(img_path).convert("RGB").resize((512, 512))
    # mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))

    init_image = PIL.Image.open(img_path).convert("RGB")
    mask_image = PIL.Image.open(mask_path).convert("L")
    mask_image = get_rotated_rect_mask(mask_image, len(prompt_text))

    mask_image.save(os.path.join(output_data_dir, str(img_index) + "_refine_mask.png"))
    ori_image = init_image
    crop_poly, init_image, mask_image = crop_image_for_test(init_image, mask_image)

    init_image.save(os.path.join(output_data_dir, str(img_index) + "_cropped_mask.png"))
    mask_viz_image = PIL.Image.open(os.path.join(input_data_dir, str(img_index), "mask_viz.png"))
    # mask_viz_image = PIL.Image.open(os.path.join(input_data_dir, str(img_index), "mask_viz.png")).convert("RGB").resize((512, 512))
    ori_image.save(os.path.join(output_data_dir, str(img_index) + ".png"))
    mask_viz_image.save(os.path.join(output_data_dir, str(img_index) + "_mask_viz.png"))

    # prompt = "fill the region with the text '" + prompt_text + "'"
    prompt = "fill the region with the text '" + prompt_text + "', and keep the background unchanged."

    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, adapter=adapter,
                 ori_text=prompt_text, text_mapper=text_mapper).images[0]

    image = image.resize((crop_poly[2] - crop_poly[0], crop_poly[3] - crop_poly[1]))
    ori_image.paste(image, (crop_poly[0], crop_poly[1]))
    image = ori_image

    res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
    image.save(res_path)
    # ipdb.set_trace()
    # print("Here")


if __name__ == '__main__':
    # with open("./demo_images/demo_images_char_10_label.txt", "r") as f:
    with open("./demo_images/demo_images_100_label.txt", "r") as f:
        data = f.read().splitlines()
    input_list = []

    for i, string in enumerate(data):
        input_list.append([i + 1, string.split(" ")[-1]])

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

    pipe = StableDiffusionInpaintWithAdapterWithFussionTEPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler, None)
    pipe = pipe.to("cuda")
    adapter = WithPrePromptAdapter(channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False)
    adapter.load_state_dict(torch.load(os.path.join(checkpoint_dir, "adapter.pth")))
    adapter = adapter.to("cuda")
    text_mapper = TextMapper()

    print("model loaded!")

    for item in input_list:
        image_inpainting(pipe, item[0], item[1], adapter=adapter, text_mapper=text_mapper)
