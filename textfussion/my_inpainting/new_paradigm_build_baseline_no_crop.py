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
from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
# from src.pipelines.stable_diffusion_inpainting_text_glyph import TextGlyphInpaintPipeline
from src.build_synth_data.batch_utils import get_input_batch_list, save_img_and_metas, split_batch_data, no_crop_split_batch_data
from src.build_synth_data.crop_tools import crop_image_regions, no_crop_replace_crop_region, get_chosen_regions, no_crop_get_chosen_regions
from src.build_synth_data.rec_inferencer import get_valid_repaint

input_data_json_path = "/data2/lfu/datasets/scene_text/build_synthtext/0518_redefine_angle_35.json"
input_bg_path = "/data2/lfu/datasets/scene_text/diffste_ocr_dataset/SynthText/bg_data/bg_img/"
# output_data_dir = "/data2/lfu/datasets/scene_text/build_synthtext/my_synth_data/new_paradigm_0605_text_glyph_90_conf_35_angle"
output_data_dir = \
    "/data2/lfu/datasets/scene_text/build_synthtext/my_synth_data/new_paradigm_0726_baseline_no_crop_no_rec"

# checkpoint_dir = "./output/new_paradigm_0601_add_glyph_2gpu"
checkpoint_dir = "./output/0416_inpainting_data_v5_zoom_up_baseline"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
    os.mkdir(output_data_dir + "/train")
else:
    os.mkdir(output_data_dir)
    os.mkdir(output_data_dir + "/train")


def image_inpainting(pipe, images, masks, prompts):
    res_image = pipe(prompt=prompts, image=images, mask_image=masks).images

    return res_image


if __name__ == '__main__':

    TEST_BATCH_SIZE = 8
    # TEST_BATCH_SIZE = 2
    with open(input_data_json_path, "r") as f:
        total_data_list = json.load(f)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
    )
    # pipe = TextGlyphInpaintPipeline.from_pretrained(
    #     checkpoint_dir,
    #     torch_dtype=torch.float16,
    # )
    pipe = pipe.to("cuda")
    rec_inferencer = TextRecInferencer(model='abinet', device='cuda:0')
    print("model loaded!")

    # input_batch_list = get_input_batch_list(total_data_list, TEST_BATCH_SIZE)

    # total_data_list_1 = total_data_list[::200]
    # total_data_list_2 = total_data_list[::200]
    # total_data_list = total_data_list_1 + total_data_list_2
    # total_data_list_1 = total_data_list[::500]
    # total_data_list_2 = total_data_list[::500]
    # total_data_list_3 = total_data_list[::500]
    # total_data_list = total_data_list_1 + total_data_list_2 + total_data_list_3
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

            chosen_index, used_index = no_crop_split_batch_data(input_item, used_index, TEST_BATCH_SIZE)

            if not chosen_index:
                upper_batch_num += 1
                continue

            cropped_images, cropped_masks, cropped_labels = no_crop_get_chosen_regions(ori_image, input_item, chosen_index)

            repaint_image = image_inpainting(pipe, cropped_images, cropped_masks, cropped_labels)
            repaint_index = get_valid_repaint(repaint_image, cropped_masks, cropped_labels, rec_inferencer)
            total_chosen_index.append(chosen_index)
            total_valid_index.append(repaint_index)

            ori_image = no_crop_replace_crop_region(ori_image, input_item, chosen_index, repaint_image,
                                            repaint_index)
            upper_batch_num += 1

        res_data_item = save_img_and_metas(output_data_dir, ori_image, input_item, total_chosen_index,
                                           total_valid_index, TEST_BATCH_SIZE)

        if res_data_item is None:
            total_empty += 1
            continue
        res_data_list.append(res_data_item)

    print("total_empty: ", total_empty)
    with open(os.path.join(output_data_dir, 'label_list.json'), 'w') as f:
        json.dump(res_data_list, f)
