import os
import cv2
import PIL
import ipdb
import numpy as np
from shapely.geometry import Polygon
from src.dataset.crop_image_for_test import crop_image_for_test

PIL.Image.MAX_IMAGE_PIXELS = None


def get_input_batch_list(total_data, batch_size=1):
    res_list = []
    for item in total_data:
        temp_dict = {"img_name": item["bg_img_path"]}
        crop_regions = item["text_region"]
        text_masks = item["text_masks"]
        text_labels = item["text_label"]

        res_text_labels = []
        for ori_word in text_labels:
            res_text_labels.append("Write english text '" + ori_word + "'.")
        text_labels = res_text_labels

        batched_crop_regions = [crop_regions[i:i + batch_size] for i in range(0, len(crop_regions), batch_size)]
        batched_text_masks = [text_masks[i:i + batch_size] for i in range(0, len(text_masks), batch_size)]
        batched_text_labels = [text_labels[i:i + batch_size] for i in range(0, len(text_labels), batch_size)]

        temp_dict["batched_crop_regions"] = batched_crop_regions
        temp_dict["batched_text_masks"] = batched_text_masks
        temp_dict["batched_text_labels"] = batched_text_labels
        temp_dict["ori_text_labels"] = item["text_label"]
        temp_dict["ori_text_bbox"] = item["text_masks"]

        res_list.append(temp_dict)

    return res_list


def get_batch_data(batch_data_info):

    batch_image, batch_mask, batch_prompt, batch_polys, ori_texts = [], [], [], [], []
    for item in batch_data_info:

        init_image = PIL.Image.open(item[0]).convert("RGB").resize((512, 512))
        mask_image = PIL.Image.open(item[1]).convert("L").resize((512, 512))

        crop_poly, _, _ = crop_image_for_test(init_image, mask_image)

        text_prompt = item[2]

        # prompt = "Write english text '" + text_prompt + "'."
        prompt = text_prompt

        batch_image.append(init_image)
        batch_mask.append(mask_image)
        batch_prompt.append(prompt)
        batch_polys.append(crop_poly)
        ori_texts.append(item[2])

    return batch_image, batch_mask, batch_prompt, batch_polys, ori_texts


def save_img_and_metas(output_dir, image, img_metas, chosen_index, valid_index, batch_size):
    res_img_meta = {}
    filter_valid_index = []
    for group_chosen_index, group_valid_index in zip(chosen_index, valid_index):
        for i, single_index in enumerate(group_chosen_index):
            if i in group_valid_index:
                filter_valid_index.append(single_index)

    total_valid_index = []
    for i in range(len(img_metas["text_region"])):
        if i in filter_valid_index:
            total_valid_index.append(1)
        else:
            total_valid_index.append(0)

    w, h = image.size
    if w >= h:
        ratio = 600 / w
        h = int(h * ratio)
        image = image.resize((600, h))
    else:
        ratio = 600 / h
        w = int(w * ratio)
        image = image.resize((w, 600))

    ori_text_bbox = [np.int0(ratio * np.array(bbox)) for bbox in img_metas["text_masks"]]
    res_text_bbox = []
    for bbox_index, bbox in enumerate(ori_text_bbox):
        if total_valid_index[bbox_index] == 1:
            res_text_bbox.append(bbox)

    if len(res_text_bbox) == 0:
        print("empty!")
        return None

    ori_text_labels = img_metas["text_label"]
    res_text_label = []
    for label_index, text_label in enumerate(ori_text_labels):
        if total_valid_index[label_index] == 1:
            res_text_label.append(text_label)

    image_name = img_metas["bg_img_path"].split("/")[-1]
    res_img_path = os.path.join("train", image_name)
    if os.path.exists(os.path.join(output_dir, res_img_path)):
        replica_time = 1
        front, last = os.path.splitext(res_img_path)
        while True:
            temp_img_path = front + "_" + str(replica_time) + last
            if os.path.exists(os.path.join(output_dir, temp_img_path)):
                replica_time += 1
            else:
                res_img_path = temp_img_path
                break

    res_img_meta["img_name"] = res_img_path

    image.save(os.path.join(output_dir, res_img_path))

    vis_image = cv2.imread(os.path.join(output_dir, res_img_path))
    for bbox in res_text_bbox:
        cv2.polylines(vis_image, [bbox], True, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, res_img_path.replace(".", "_vis.")), vis_image)

    if len(res_text_bbox) == 0:
        print("empty!")
        return None
    res_img_meta["text_labels"] = res_text_label
    res_text_bbox = [bbox_item.tolist() for bbox_item in res_text_bbox]
    res_img_meta["text_bbox"] = res_text_bbox

    return res_img_meta


def split_batch_data(data_item, used_index, batch_size):
    text_region = []
    text_region_index = []
    for number_item, index in enumerate(used_index):
        if index == 0:
            text_region.append(data_item["text_region"][number_item])
            text_region_index.append(number_item)

    chosen_index, _ = filter_rect(text_region)
    if len(chosen_index) > batch_size:
        chosen_index = chosen_index[:batch_size]

    chosen_index = [text_region_index[i] for i in chosen_index]

    filter_chosen_index = []
    for single_index in chosen_index:
        single_poly = data_item["text_masks"][single_index]
        # single_poly = [[single_poly[0], single_poly[1]], [single_poly[2], single_poly[1]],
        #                [single_poly[2], single_poly[3]], [single_poly[0], single_poly[3]]]
        single_poly = Polygon(single_poly)
        filter_flag = True
        for used_num, single_used_index in enumerate(used_index):
            if used_num == single_index:
                continue
            if single_used_index == 1:
                used_poly = data_item["text_masks"][used_num]
                # used_poly = [[used_poly[0], used_poly[1]], [used_poly[2], used_poly[1]],
                #              [used_poly[2], used_poly[3]], [used_poly[0], used_poly[3]]]
                used_poly = Polygon(used_poly)
                try:
                    if single_poly.intersects(used_poly):
                        intersection_area = single_poly.intersection(used_poly).area
                        if intersection_area > 0.2 * used_poly.area:
                            filter_flag = False
                            break
                        elif intersection_area > 0.2 * single_poly.area:
                            filter_flag = False
                            break
                        else:
                            continue
                    else:
                        continue
                except:
                    print("try-except: ")
                    print(single_poly)
                    print(used_poly)
                    filter_flag = False
            else:
                continue
        if filter_flag:
            filter_chosen_index.append(single_index)

    for index in chosen_index:
        used_index[index] = 1

    return filter_chosen_index, used_index


def no_crop_split_batch_data(data_item, used_index, batch_size):
    text_region = []
    text_region_index = []

    for number_item, index in enumerate(used_index):
        if index == 0:
            text_region.append(data_item["text_region"][number_item])
            text_region_index.append(number_item)
            used_index[number_item] = 1
            break

    filter_chosen_index = text_region_index

    return filter_chosen_index, used_index


def filter_rect(rectangles):
    polygons = []
    for rect in rectangles:
        rect = np.array([[rect[0], rect[1]], [rect[2], rect[1]], [rect[2], rect[3]], [rect[0], rect[3]]])
        polygons.append(Polygon(rect))

    # 定义结果列表和当前处理的矩形框集合
    result = []
    result_index = []
    remaining = set(range(len(polygons)))

    while remaining:
        # 选取一个未处理的矩形框
        i = remaining.pop()
        p = polygons[i]

        # 找到与当前矩形框相交的所有矩形框
        neighbors = set()
        for j, q in enumerate(polygons):
            if i != j and p.intersects(q):
                neighbors.add(j)

        # 将与当前矩形框相交的所有矩形框从未处理集合中移除
        remaining.difference_update(neighbors)

        # 将当前矩形框加入结果列表中
        result.append(np.array(p.exterior.coords))
        result_index.append(i)

    try:
        result = np.array(result)
        result = np.int0(result[:, :4, :])
    except:
        print("debug")
    return result_index, result
