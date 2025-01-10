import os
import PIL
import ipdb
import json
import torch
import random
import shutil
import numpy as np

from tqdm import tqdm
from mmocr.apis import TextRecInferencer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
from src.models.unet_2d_with_dual_text import UNet2DConditionWithDualTextModel
from src.models.union_net import CharEncoder
from src.pipelines.new_paradigm_inpainting_dual_text_encoder import DualTextInpaintPipeline
from src.build_synth_data.batch_utils import get_input_batch_list, save_img_and_metas, split_batch_data
from src.build_synth_data.crop_tools import crop_image_regions, replace_crop_region, get_chosen_regions
from src.build_synth_data.rec_inferencer import get_valid_repaint
from src.dataset.text_mapper import TextMapper
input_data_json_path = "/data2/lfu/datasets/scene_text/build_synthtext/0518_redefine_angle_35.json"
input_bg_path = "/data2/lfu/datasets/scene_text/diffste_ocr_dataset/SynthText/bg_data/bg_img/"
output_data_dir = \
    "/data2/lfu/datasets/scene_text/build_synthtext/my_synth_data/new_paradigm_0614_dual_text"

checkpoint_dir = "./output/new_paradigm_0607_inpainting_dual_text"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
    os.mkdir(output_data_dir + "/train")
else:
    os.mkdir(output_data_dir)
    os.mkdir(output_data_dir + "/train")


def image_inpainting(pipe, images, masks, prompts, char_encoder, text_mapper):
    res_image = pipe(prompt=prompts, image=images, mask_image=masks, char_encoder=char_encoder,
                     text_mapper=text_mapper).images

    return res_image


if __name__ == '__main__':

    TEST_BATCH_SIZE = 4
    # TEST_BATCH_SIZE = 2
    with open(input_data_json_path, "r") as f:
        total_data_list = json.load(f)
    MODEL_NAME = checkpoint_dir
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", revision=None
    )
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae", revision=None)
    unet = UNet2DConditionWithDualTextModel.from_pretrained(
        MODEL_NAME, subfolder="unet", revision=None
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", revision=None
    )

    char_encoder = CharEncoder()
    char_encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, "union_net_char_encoder.pth")))
    char_encoder = char_encoder.to("cuda")
    pipe = DualTextInpaintPipeline(vae, text_encoder, tokenizer, unet, noise_scheduler, None)
    pipe = pipe.to("cuda")

    text_mapper = TextMapper()
    rec_inferencer = TextRecInferencer(model='abinet', device='cuda:0')
    print("model loaded!")

    random.shuffle(total_data_list)

    total_empty = 0
    res_data_list = []
    for input_item in tqdm(total_data_list, ncols=80):
        img_name = input_item["bg_img_path"]

        img_name = img_name.replace("/home/lfu/project/DiffSTE/ocr-dataset/SynthText/bg_data/bg_img",
                                    "/data2/lfu/datasets/scene_text/diffste_ocr_dataset/SynthText/bg_data/bg_img")
        ori_image = PIL.Image.open(img_name).convert("RGB")

        used_index = [0] * len(input_item["text_region"])
        upper_batch_num = 0
        total_chosen_index = []
        total_valid_index = []
        while upper_batch_num < 16:

            chosen_index, used_index = split_batch_data(input_item, used_index, TEST_BATCH_SIZE)

            if not chosen_index:
                upper_batch_num += 1
                continue

            cropped_images, cropped_masks, cropped_labels = get_chosen_regions(ori_image, input_item, chosen_index)

            repaint_image = image_inpainting(pipe, cropped_images, cropped_masks, cropped_labels,
                                             char_encoder, text_mapper)
            repaint_index = get_valid_repaint(repaint_image, cropped_masks, cropped_labels, rec_inferencer)
            total_chosen_index.append(chosen_index)
            total_valid_index.append(repaint_index)

            ori_image = replace_crop_region(ori_image, input_item, chosen_index, repaint_image,
                                            repaint_index)
            upper_batch_num += 1

        res_data_item = save_img_and_metas(output_data_dir, ori_image, input_item, total_chosen_index,
                                           total_valid_index, TEST_BATCH_SIZE)

        if res_data_item is None:
            total_empty += 1
            continue
        res_data_list.append(res_data_item)
        # ipdb.set_trace()

    print("total_empty: ", total_empty)
    with open(os.path.join(output_data_dir, 'label_list.json'), 'w') as f:
        json.dump(res_data_list, f)
