import os
import PIL
import ipdb
import json
import torch
import shutil
import numpy as np

from tqdm import tqdm
from mmocr.apis import TextRecInferencer
from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
from src.build_synth_data.batch_utils import get_input_batch_list, save_img_and_metas
from src.build_synth_data.crop_tools import crop_image_regions, replace_crop_region
from src.build_synth_data.rec_inferencer import get_valid_repaint

input_data_json_path = "/data2/lfu/datasets/scene_text/build_synthtext/no_small_total_data.json"
input_bg_path = "/data2/lfu/datasets/scene_text/diffste_ocr_dataset/SynthText/bg_data/bg_img/"
# output_data_dir = "/data2/lfu/datasets/scene_text/build_synthtext/my_synth_data/baseline_with_large_num_with_v2_rec"
output_data_dir = "/data2/lfu/datasets/scene_text/build_synthtext/my_synth_data/try"

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

    TEST_BATCH_SIZE = 1
    # TEST_BATCH_SIZE = 2
    with open(input_data_json_path, "r") as f:
        total_data_list = json.load(f)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    rec_inferencer = TextRecInferencer(model='abinet', device='cuda:0')
    print("model loaded!")

    input_batch_list = get_input_batch_list(total_data_list, TEST_BATCH_SIZE)

    # input_batch_list = input_batch_list[::200]
    res_data_list = []
    for input_item in tqdm(input_batch_list, ncols=80):
        img_name = input_item["img_name"]
        img_name = img_name.replace("/home/lfu/project/DiffSTE/ocr-dataset/SynthText/bg_data/bg_img",
                                    "/data2/lfu/datasets/scene_text/diffste_ocr_dataset/SynthText/bg_data/bg_img")
        ori_image = PIL.Image.open(img_name).convert("RGB")
        cropped_images, cropped_masks = crop_image_regions(ori_image, input_item["batched_crop_regions"], input_item["batched_text_masks"])
        total_repaint_image = []
        total_repaint_index = []
        for batch_images, batch_masks, batch_labels in zip(cropped_images, cropped_masks, input_item["batched_text_labels"]):
            repaint_image = image_inpainting(pipe, batch_images, batch_masks, batch_labels)
            repaint_index = get_valid_repaint(repaint_image, batch_masks, rec_inferencer)
            total_repaint_image.append(repaint_image)
            total_repaint_index.append(repaint_index)
            # for index, item in enumerate(repaint_image):
            #     item.save("vis_" + str(index) + ".png")

        res_image = replace_crop_region(ori_image, input_item["batched_crop_regions"], total_repaint_image,
                                        total_repaint_index)

        res_data_item = save_img_and_metas(output_data_dir, res_image, input_item, total_repaint_index, TEST_BATCH_SIZE)
        if res_data_item is None:
            continue
        res_data_list.append(res_data_item)

    with open(os.path.join(output_data_dir, 'label_list.json'), 'w') as f:
        json.dump(res_data_list, f)
