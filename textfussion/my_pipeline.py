# # #!/usr/bin/env python3
from diffusers import DiffusionPipeline
import os
import PIL
import requests
from io import BytesIO
from tqdm import tqdm
import torch
import ipdb
import shutil


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "json" in file:
                continue
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", custom_pipeline="/home/lfu/project/textfussion/examples/community/stable_diffusion_mega.py",
                                         torch_dtype=torch.float16, revision="fp16")
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", custom_pipeline="stable_diffusion_mega",
#                                          torch_dtype=torch.float16, revision="fp16")

pipe.to("cuda")
pipe.enable_attention_slicing()

batch_size = 1
img_list = get_file_paths("/data2/lfu/datasets/scene_text/build_synthtext/10k_synthtext/")
output_data_dir = "/data2/lfu/datasets/scene_text/build_synthtext/filtered_synthtext/"
if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)

batched_img_paths = [img_list[i:i + batch_size] for i in range(0, len(img_list), batch_size)]

for batch_paths in tqdm(batched_img_paths, ncols=80):
    batch_images = []
    for single_path in batch_paths:
        single_image = PIL.Image.open(single_path).convert("RGB")
        batch_images.append(single_image)

    prompt = ["A scene image with english texts."] * batch_size

    images = pipe.img2img(prompt=prompt, image=batch_images, strength=0.1, guidance_scale=7.5).images

    for res_image, single_path in zip(images, batch_paths):
        res_path = single_path.replace("10k_synthtext", "filtered_synthtext")
        folder_name = os.path.dirname(res_path)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        res_image.save(res_path)


################
# import PIL
# import ipdb
# import requests
# from PIL import Image
# from io import BytesIO
# from diffusers import StableDiffusionUpscalePipeline
# import torch
#
# # load model and scheduler
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
# pipeline = StableDiffusionUpscalePipeline.from_pretrained(
#     model_id, revision="fp16", torch_dtype=torch.float16
# )
# pipeline = pipeline.to("cuda")
#
# # let's download an  image
# init_image = PIL.Image.open("vis_ori.png").convert("RGB")
# w, h = init_image.size
# print(init_image.size)
# # ipdb.set_trace()
# low_res_img = init_image.resize((192, 192))
# prompt = ""
#
# upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
# print(upscaled_image.size)
# upscaled_image = upscaled_image.resize((w, h))
# upscaled_image.save("vis_res_upsampled.png")
