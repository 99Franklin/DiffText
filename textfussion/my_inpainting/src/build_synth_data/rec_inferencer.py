import os
import re
import cv2
import ipdb
import numpy as np

rule = re.compile(u"[^a-zA-Z0-9]")


def get_valid_repaint(input_images, input_masks, input_labels, rec_inferencer):
    valid_index = []
    valid_images = []

    for img, mask in zip(input_images, input_masks):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        mask_bbox = mask.getbbox()

        # ipdb.set_trace()
        # cv2.imwrite("vis_ori.png", img)
        # res = img[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]]
        # cv2.imwrite("vis_res.png", res)

        if mask_bbox is not None:
            img = img[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]]

        valid_images.append(img)

    rec_pred = rec_inferencer(valid_images)["predictions"]
    for index, rec_item in enumerate(rec_pred):
        gt_label = input_labels[index]
        gt_label = gt_label.split("'")
        gt_label = "'".join(gt_label[1:-1])
        shorted_label = rule.sub('', gt_label)
        if len(shorted_label) <= 0:
            print(shorted_label)
            continue
        elif rec_item["scores"] < 0.9:
            continue
        elif len(rec_item["text"]) < 0.5 * len(gt_label):
            print(rec_item["text"])
            print(gt_label)
            continue
        elif len(rec_item["text"]) > 1.5 * len(gt_label):
            print(rec_item["text"])
            print(gt_label)
            continue
        else:
            valid_index.append(index)

    return valid_index

# def get_valid_repaint(input_images, input_masks, input_labels, rec_inferencer):
#     valid_index = []
#
#     for index, image in enumerate(input_images):
#         valid_index.append(index)
#
#     return valid_index

