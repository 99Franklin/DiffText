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
from src.models.char_encoder import CharEncoder
from src.models.dual_controlnet import DualControlNet
from src.models.unet_2d_with_dual_text_controlnet import UNet2DConditionWithDualTextControlNetModel
from src.pipelines.stable_diffusion_inpainting_dual_text_full_controlnet \
    import StableDiffusionInpaintDualTextFullControlnetPipeline
from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
from src.dataset.crop_image_for_test import crop_image_for_test
from src.dataset.batch_utils import get_input_batch_list, get_batch_data, batch_save

input_data_dir = "./demo_images/demo_images_20"
output_data_dir = "./demo/0506_full_controlnet_dual_text_with_adjust"

checkpoint_dir = "./output/0502_full_controlnet_dual_text"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, batch_data_index, batch_data_input, TEST_BATCH_SIZE,
                     adapter=None, char_encoder=None, text_mapper=None):
    init_image, mask_image, prompt, polys, ori_texts = get_batch_data(batch_data_input)

    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, adapter=adapter, char_encoder=char_encoder,
                 text_mapper=text_mapper, ori_text=ori_texts).images

    batch_save(image, output_data_dir, batch_data_index, polys, TEST_BATCH_SIZE)


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
    unet = UNet2DConditionWithDualTextControlNetModel.from_pretrained(
        MODEL_NAME, subfolder="unet", revision=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", revision=None
    )

    ipdb.set_trace()
    unet.load_state_dict(torch.load(os.path.join(checkpoint_dir, "controledunet_unt.pth")))
    pipe = StableDiffusionInpaintDualTextFullControlnetPipeline(
        vae, text_encoder, tokenizer, unet, noise_scheduler, None)
    pipe = pipe.to("cuda")

    adapter = DualControlNet()
    adapter.load_state_dict(torch.load(os.path.join(checkpoint_dir, "controledunet_controlnet.pth")))
    adapter = adapter.to("cuda")

    char_encoder = CharEncoder()
    char_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "controledunet_char_encoder.pth")))
    char_encoder = char_encoder.to("cuda")

    text_mapper = TextMapper()

    print("model loaded!")

    input_batch_list = get_input_batch_list(input_data_dir, word_data, TEST_BATCH_SIZE)
    for batch_index, batch_item in tqdm(enumerate(input_batch_list)):
        image_inpainting(pipe, batch_index, batch_item, TEST_BATCH_SIZE,
                         adapter=adapter, char_encoder=char_encoder, text_mapper=text_mapper)
