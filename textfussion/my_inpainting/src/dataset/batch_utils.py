import os
import PIL
from src.dataset.crop_image_for_test import crop_image_for_test, get_rotated_rect_mask


def get_input_batch_list(img_dir, word_data, batch_size=1):
    total_data_list = []
    total_label_list = []
    temp_index = 0
    for img_index in range(1, 21):
        for word_index in range(50):
            img_path = os.path.join(img_dir, str(img_index), "img.png")
            mask_path = os.path.join(img_dir, str(img_index), "mask.png")
            word_label = word_data[word_index]
            total_data_list.append([img_path, mask_path, word_label])
            total_label_list.append([str(temp_index), word_label])
            temp_index += 1
    batched_list = [total_data_list[i:i + batch_size] for i in range(0, len(total_data_list), batch_size)]

    f = open("/home/lfu/project/textfussion/my_inpainting/demo_images/label_list.txt", "w")
    for item in total_label_list:
        f.write(item[0] + " " + item[1] + "\n")

    return batched_list


def get_batch_data(batch_data_info):

    batch_image, batch_mask, batch_prompt, batch_polys, ori_texts = [], [], [], [], []
    for item in batch_data_info:

        init_image = PIL.Image.open(item[0]).convert("RGB").resize((512, 512))
        mask_image = PIL.Image.open(item[1]).convert("L").resize((512, 512))

        mask_image = get_rotated_rect_mask(mask_image, len(item[2]))

        crop_poly, _, _ = crop_image_for_test(init_image, mask_image)

        text_prompt = item[2]

        prompt = "Write english text '" + text_prompt + "'."
        # prompt = text_prompt

        batch_image.append(init_image)
        batch_mask.append(mask_image)
        batch_prompt.append(prompt)
        batch_polys.append(crop_poly)
        ori_texts.append(item[2])

    return batch_image, batch_mask, batch_prompt, batch_polys, ori_texts


def batch_save(images, res_dir, batch_index, polys, TEST_BATCH_SIZE):
    for image_index, image in enumerate(images):
        image = image.crop(polys[image_index]).resize((512, 512))
        res_index = TEST_BATCH_SIZE * batch_index + image_index
        res_path = os.path.join(res_dir, str(res_index) + ".png")
        image.save(res_path)
