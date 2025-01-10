import os
import PIL
import ipdb
import json
import torch
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from mmocr.apis import TextRecInferencer
from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
# from src.pipelines.stable_diffusion_inpainting_text_glyph import TextGlyphInpaintPipeline
from src.build_synth_data.batch_utils import get_input_batch_list, save_img_and_metas, split_batch_data
from src.build_synth_data.crop_tools import crop_image_regions, replace_crop_region, get_chosen_regions
from src.build_synth_data.rec_inferencer import get_valid_repaint

# output_data_dir = \
#     "./img_for_paper/any_demo_img"

output_data_dir = \
    "./img_for_paper/any_demo_img_second_gen"

checkpoint_dir = "./output/0416_inpainting_data_v5_zoom_up_baseline"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)

os.mkdir(output_data_dir)


def get_text_masks(img_size=(1024, 1024), text_region=None):
    img_size = img_size
    polygon_points = text_region

    background_color = 0
    foreground_color = 255

    img = Image.new("L", img_size, background_color)
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon_points, fill=foreground_color)
    img.save(os.path.join(output_data_dir, "demo_mask.png"))

    return img


def image_inpainting(pipe, images, masks, prompts):
    res_image = pipe(prompt=prompts, image=images, mask_image=masks).images
    return res_image


if __name__ == '__main__':

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    # rec_inferencer = TextRecInferencer(model='abinet', device='cuda:0')
    print("model loaded!")

    # img_name = "./demo_images/demo_images_20/2/img.png"
    # img_name = "./demo_images/for_paper/1.jpg"
    img_name = "./demo_images/for_paper/difftext_water.png"
    ori_image = PIL.Image.open(img_name).convert("RGB")

    ori_image = ori_image.resize((512, 512))
    cropped_images = [ori_image]

    # demo_images_20/19
    # text_region = [(165, 235), (380, 235), (380, 285), (165, 285)]
    # text_region = [(190, 325), (400, 325), (380, 385), (175, 385)]
    text_region = [(50, 360), (250, 360), (240, 385), (35, 385)]
    cropped_masks = [get_text_masks(img_size=ori_image.size, text_region=text_region)]
    cropped_labels = ["WELCOME"]

    for i in range(0, 31):
        repaint_image = image_inpainting(pipe, cropped_images, cropped_masks, cropped_labels)[0]
        repaint_image.save(os.path.join(output_data_dir, "demo_final_" + str(i) + ".png"))

    # repaint_index = get_valid_repaint(repaint_image, cropped_masks, cropped_labels, rec_inferencer)
    # total_chosen_index.append(chosen_index)
    # total_valid_index.append(repaint_index)
    #
    # ori_image = replace_crop_region(ori_image, input_item, chosen_index, repaint_image,
    #                                 repaint_index)
    # ori_image.save(os.path.join(output_data_dir, "demo.png"))

