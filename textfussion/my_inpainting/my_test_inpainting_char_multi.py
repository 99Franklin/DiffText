import os
import PIL
import ipdb
import torch
import shutil
import random
from tqdm import tqdm

from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline

input_data_dir = "./demo_images/demo_images_100"
output_data_dir = "./demo/inpainting_char_100"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, img_index, prompt_text):

    random_index = random.randint(0, len(prompt_text) - 1)

    prompt_text = prompt_text[random_index]

    img_path = os.path.join(input_data_dir, str(img_index), "img.png")
    mask_path = os.path.join(input_data_dir, str(img_index), "mask.png")

    init_image = PIL.Image.open(img_path).convert("RGB").resize((512, 512))
    mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))

    init_image.save(os.path.join(output_data_dir, str(img_index) + ".png"))

    # prompt = "fill the region with the text '" + prompt_text + "'"
    prompt = "fill the region with the text '" + prompt_text + "', and keep the background unchanged."

    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

    res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
    image.save(res_path)


if __name__ == '__main__':
    with open("./demo_images/demo_images_100_label.txt", "r") as f:
        data = f.read().splitlines()
    input_list = []

    for i, string in enumerate(data):
        input_list.append([i + 1, string.split(" ")[-1]])

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "./output/0318_inpainting_baseline_char",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    print("model loaded!")

    for item in input_list:
        image_inpainting(pipe, item[0], item[1])
