import os
import PIL
import ipdb
import torch
import shutil
from tqdm import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from src.dataset.text_mapper import TextMapper
from src.models.only_prefix_prompt import OnlyPrefixPromptAdapter
from src.pipelines.stable_diffusion_inpainting_only_pre_prompt \
    import StableDiffusionInpaintOnlyPrePromptPipeline
from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
from src.dataset.crop_image_for_test import crop_image_for_test
from src.dataset.batch_utils import get_input_batch_list, get_batch_data, batch_save

input_data_dir = "./demo_images/demo_images_20"
output_data_dir = "./demo/0421_cross_prefix_te_cross_prefix_clip_batch_test_with_adjust"

checkpoint_dir = "./output/0419_inpainting_data_v5_cross_prefix_te_cross_prefix_clip"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, batch_data_index, batch_data_input, TEST_BATCH_SIZE, adapter=None, text_mapper=None):
    init_image, mask_image, prompt, polys, ori_texts = get_batch_data(batch_data_input)

    # init_image = PIL.Image.open(img_path).convert("RGB")
    # mask_image = PIL.Image.open(mask_path).convert("L")
    # # ipdb.set_trace()
    # ori_image = init_image
    # ori_image.save(os.path.join(output_data_dir, str(img_index) + ".png"))
    #
    # crop_poly, init_image, mask_image = crop_image_for_test(init_image, mask_image)
    # mask_image.save(os.path.join(output_data_dir, str(img_index) + "_mask.png"))
    #
    # init_image = init_image
    # init_image.save(os.path.join(output_data_dir, str(img_index) + "_cropped.png"))
    #
    # mask_viz_image = PIL.Image.open(os.path.join(input_data_dir, str(img_index), "mask_viz.png")).convert("RGB")
    # mask_viz_image.save(os.path.join(output_data_dir, str(img_index) + "_mask_viz.png"))

    # prompt = "fill the region with the text '" + prompt_text + "'"
    # prompt = "fill the region with the text '" + prompt_text + "', and keep the background unchanged."
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, adapter=adapter,
                 ori_text=ori_texts, text_mapper=text_mapper).images

    # image = image.resize((crop_poly[2] - crop_poly[0], crop_poly[3] - crop_poly[1]))
    # ori_image.paste(image, (crop_poly[0], crop_poly[1]))
    # image = ori_image

    batch_save(image, output_data_dir, batch_data_index, polys, TEST_BATCH_SIZE)
    # res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
    # image.save(res_path)


if __name__ == '__main__':
    TEST_BATCH_SIZE = 6
    with open("./demo_images/demo_images_20_story.txt", "r") as f:
        word_data = f.read().splitlines()
    input_list = []

    MODEL_NAME = "stabilityai/stable-diffusion-2-inpainting"
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", revision=None)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_NAME, subfolder="unet", revision=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", revision=None
    )

    pipe = StableDiffusionInpaintOnlyPrePromptPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler, None)
    pipe = pipe.to("cuda")
    adapter = OnlyPrefixPromptAdapter()
    adapter.load_state_dict(torch.load(os.path.join(checkpoint_dir, "adapter.pth")))
    adapter = adapter.to("cuda")
    text_mapper = TextMapper()
    print("model loaded!")

    input_batch_list = get_input_batch_list(input_data_dir, word_data, TEST_BATCH_SIZE)
    for batch_index, batch_item in enumerate(input_batch_list):
        image_inpainting(pipe, batch_index, batch_item, TEST_BATCH_SIZE, adapter=adapter, text_mapper=text_mapper)
