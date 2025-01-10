# import os
# import PIL
# import ipdb
# import torch
# import shutil
# from tqdm import tqdm
#
# from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
# from src.dataset.crop_image_for_test import crop_image_for_test
# from src.dataset.batch_utils import get_input_batch_list
#
# input_data_dir = "./demo_images/demo_images_100"
# output_data_dir = "./demo/try_batch"
#
# checkpoint_dir = "./output/0411_inpainting_with_blank_data_v5_baseline"
#
# if os.path.exists(output_data_dir):
#     shutil.rmtree(output_data_dir)
#     os.mkdir(output_data_dir)
# else:
#     os.mkdir(output_data_dir)
#
#
# def image_inpainting(pipe, img_index, prompt_text):
#     ipdb.set_trace()
#
#     img_path = os.path.join(input_data_dir, str(img_index), "img.png")
#     mask_path = os.path.join(input_data_dir, str(img_index), "mask.png")
#
#     init_image = PIL.Image.open(img_path).convert("RGB").resize((512, 512))
#     mask_image = PIL.Image.open(mask_path).convert("RGB").resize((512, 512))
#
#     # init_image = PIL.Image.open(img_path).convert("RGB")
#     # mask_image = PIL.Image.open(mask_path).convert("L")
#     # # ipdb.set_trace()
#     # ori_image = init_image
#     # ori_image.save(os.path.join(output_data_dir, str(img_index) + ".png"))
#     #
#     # crop_poly, init_image, mask_image = crop_image_for_test(init_image, mask_image)
#     # mask_image.save(os.path.join(output_data_dir, str(img_index) + "_mask.png"))
#     #
#     # init_image = init_image
#     # init_image.save(os.path.join(output_data_dir, str(img_index) + "_cropped.png"))
#     #
#     # mask_viz_image = PIL.Image.open(os.path.join(input_data_dir, str(img_index), "mask_viz.png")).convert("RGB")
#     # mask_viz_image.save(os.path.join(output_data_dir, str(img_index) + "_mask_viz.png"))
#
#     # prompt = "fill the region with the text '" + prompt_text + "'"
#     # prompt = "fill the region with the text '" + prompt_text + "', and keep the background unchanged."
#
#     # ipdb.set_trace()
#     text = [char for char in prompt_text]
#     text = " ".join(text)
#     prompt = "Write some characters '" + text + "'"
#
#     image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
#
#     # image = image.resize((crop_poly[2] - crop_poly[0], crop_poly[3] - crop_poly[1]))
#     # ori_image.paste(image, (crop_poly[0], crop_poly[1]))
#     # image = ori_image
#
#     res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
#     image.save(res_path)
#
#
# if __name__ == '__main__':
#     with open("./demo_images/demo_images_100_label.txt", "r") as f:
#         data = f.read().splitlines()
#     input_list = []
#
#     for i, string in enumerate(data):
#         input_list.append([i + 1, string.split(" ")[-1]])
#
#     # pipe = StableDiffusionInpaintPipeline.from_pretrained(
#     #     "stabilityai/stable-diffusion-2-inpainting",
#     #     torch_dtype=torch.float16,
#     # )
#     # ipdb.set_trace()
#     pipe = StableDiffusionInpaintPipeline.from_pretrained(
#         checkpoint_dir,
#         torch_dtype=torch.float16,
#     )
#     pipe = pipe.to("cuda")
#
#     print("model loaded!")

#     for item in input_list:
#         image_inpainting(pipe, item[0], item[1])

# batch test

import os
import PIL
import ipdb
import torch
import shutil
from tqdm import tqdm

from src.pipelines.stable_diffusion_inpainting import StableDiffusionInpaintPipeline
from src.dataset.crop_image_for_test import crop_image_for_test
from src.dataset.batch_utils import get_input_batch_list, get_batch_data, batch_save

input_data_dir = "./demo_images/demo_images_20"
output_data_dir = "./demo/try"

checkpoint_dir = "./output/0416_inpainting_data_v5_zoom_up_baseline"

if os.path.exists(output_data_dir):
    shutil.rmtree(output_data_dir)
    os.mkdir(output_data_dir)
else:
    os.mkdir(output_data_dir)


def image_inpainting(pipe, batch_data_index, batch_data_input, TEST_BATCH_SIZE):
    init_image, mask_image, prompt, polys, _ = get_batch_data(batch_data_input)

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

    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images

    # image = image.resize((crop_poly[2] - crop_poly[0], crop_poly[3] - crop_poly[1]))
    # ori_image.paste(image, (crop_poly[0], crop_poly[1]))
    # image = ori_image

    batch_save(image, output_data_dir, batch_data_index, polys, TEST_BATCH_SIZE)
    # res_path = os.path.join(output_data_dir, str(img_index) + "_" + prompt_text + ".png")
    # image.save(res_path)


if __name__ == '__main__':
    TEST_BATCH_SIZE = 8
    with open("./demo_images/demo_images_20_story.txt", "r") as f:
        word_data = f.read().splitlines()
    input_list = []

    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float16,
    # )
    # ipdb.set_trace()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    # print("model loaded!")

    input_batch_list = get_input_batch_list(input_data_dir, word_data, TEST_BATCH_SIZE)
    for batch_index, batch_item in enumerate(input_batch_list):
        image_inpainting(pipe, batch_index, batch_item, TEST_BATCH_SIZE)
