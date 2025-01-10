import cv2
import math
import ipdb
import random
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS = None


def crop_image_regions(image, region_pts, mask_pts):
    res_batch_images, res_batch_masks = [], []
    for batch_region_pts, batch_mask_pts in zip(region_pts, mask_pts):

        res_images = []
        res_masks = []
        for region_pt, mask_pt in zip(batch_region_pts, batch_mask_pts):
            region_pt = np.array(region_pt)
            mask_pt = np.array(mask_pt)

            # cv2_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            # draw_region = region_pt[:, None, :]
            # draw_mask = mask_pt[:, None, :]
            # cv2.polylines(cv2_img, [draw_region], True, (0, 255, 0), 3)
            # cv2.polylines(cv2_img, [draw_mask], True, (0, 0, 255), 3)
            # cv2.imwrite("vis_img_2.png", cv2_img)
            x_min, y_min = min(region_pt[:, 0]), min(region_pt[:, 1])
            x_max, y_max = max(region_pt[:, 0]), max(region_pt[:, 1])

            image_patch = image.crop([x_min, y_min, x_max, y_max])

            mask_pt[:, 0] = mask_pt[:, 0] - x_min
            mask_pt[:, 1] = mask_pt[:, 1] - y_min
            mask_patch = np.zeros(image_patch.size)
            # image_patch.save("vis_crop.png")
            cv2.fillPoly(mask_patch, [mask_pt], 255)
            # cv2.imwrite("vis_patch.png", mask_patch)

            mask_patch = Image.fromarray(np.uint8(mask_patch))
            # mask_patch.save("vis_patch_2.png")
            # ipdb.set_trace()
            image_patch = image_patch.resize((512, 512))
            mask_patch = mask_patch.resize((512, 512))
            res_images.append(image_patch)
            res_masks.append(mask_patch)
        res_batch_images.append(res_images)
        res_batch_masks.append(res_masks)
    return res_batch_images, res_batch_masks


def replace_crop_region(old_image, input_item, chosen_index, batch_crop_images, batch_valid_index):
    text_regions = input_item["text_region"]
    chosen_region = []
    for index in range(len(text_regions)):
        if index in chosen_index:
            chosen_region.append(text_regions[index])

    for valid_index in batch_valid_index:
        crop_poly = chosen_region[valid_index]
        crop_image = batch_crop_images[valid_index]

        x_min, y_min, x_max, y_max = int(crop_poly[0]), int(crop_poly[1]), int(crop_poly[2]), int(crop_poly[3])

        crop_image = crop_image.resize((x_max - x_min, y_max - y_min))
        old_image.paste(crop_image, (x_min, y_min))

    # old_image.save("vis_paste.png")

    return old_image


def no_crop_replace_crop_region(old_image, input_item, chosen_index, batch_crop_images, batch_valid_index):
    text_regions = input_item["text_region"]
    chosen_region = []
    for index in range(len(text_regions)):
        if index in chosen_index:
            chosen_region.append(text_regions[index])

    for valid_index in batch_valid_index:
        crop_poly = chosen_region[valid_index]
        crop_image = batch_crop_images[valid_index]

        x_min, y_min, x_max, y_max = int(crop_poly[0]), int(crop_poly[1]), int(crop_poly[2]), int(crop_poly[3])

        crop_image = crop_image.resize(old_image.size)
        old_image.paste(crop_image, (0, 0))

    # old_image.save("vis_paste.png")

    return old_image


def bezier_curve(points, num_points=50):
    # B(t) = (1 - t) ^ 3 * P0 + 3t(1 - t) ^ 2 * P1 + 3t ^ 2(1 - t) * P2 + t ^ 3 * P3
    t = np.linspace(0., 1., num_points)
    curve_points = np.zeros((num_points, 2))
    for i in range(num_points):
        curve_points[i] = (1 - t[i]) ** 3 * points[0] + 3 * t[i] * (1 - t[i]) ** 2 * points[1] + 3 * t[i] ** 2 * (1 - t[i]) * points[2] + (t[i] ** 3) * points[3]
    return curve_points[:,0], curve_points[:,1]


def get_chosen_regions(image, data_item, chosen_index):
    res_batch_images, res_batch_masks, res_batch_labels = [], [], []

    for index in chosen_index:
        region_pt = np.array(data_item["text_region"][index])
        region_pt = np.uint(np.array([[region_pt[0], region_pt[1]], [region_pt[2], region_pt[1]],
                                      [region_pt[2], region_pt[3]], [region_pt[0], region_pt[3]]]))
        mask_pt = np.array(data_item["text_masks"][index])
        label = data_item["text_label"][index]
        # label = "Write english text '" + label + "'."
        label = "A scene image with english text '" + label + "'."

        # cv2_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # vis_img = cv2_img.copy()
        # draw_region = region_pt[:, None, :]
        # draw_mask = mask_pt[:, None, :]
        # cv2.polylines(cv2_img, [draw_region], True, (0, 255, 0), 3)
        # cv2.polylines(cv2_img, [draw_mask], True, (0, 0, 255), 3)
        # cv2.imwrite("vis_img_2.png", cv2_img)
        #
        # print(mask_pt)

        # mid_upper = (mask_pt[0] + mask_pt[1]) / 2
        # mid_lower = (mask_pt[2] + mask_pt[3]) / 2
        # offset = (mask_pt[2][1] + mask_pt[3][1] - mask_pt[1][1] - mask_pt[0][1]) / 3 # 偏移量，可以根据需要调整
        # random_offset = random.uniform(-offset, offset)
        # ctrl_p1 = mid_upper + np.array([0, random_offset])
        # ctrl_p2 = mid_lower + np.array([0, random_offset])
        #
        # x1, y1 = bezier_curve(np.array([mask_pt[0], ctrl_p1, ctrl_p1, mask_pt[1]]))
        # x2, y2 = bezier_curve(np.array([mask_pt[2], ctrl_p2, ctrl_p2, mask_pt[3]]))
        #
        # upper_edge_points = np.column_stack((x1, y1))
        # lower_edge_points = np.column_stack((x2, y2))
        # mask_pt = np.concatenate((upper_edge_points, lower_edge_points), axis=0)

        # print(mask_pt)
        # mask_pt = np.uint(mask_pt)
        # draw_region = mask_pt[:, None, :]
        # cv2.polylines(vis_img, [draw_region], True, (0, 255, 0), 2)
        # cv2.imwrite("vis_img_3.png", vis_img)
        #
        # import ipdb
        # ipdb.set_trace()

        x_min, y_min = min(region_pt[:, 0]), min(region_pt[:, 1])
        x_max, y_max = max(region_pt[:, 0]), max(region_pt[:, 1])

        image_patch = image.crop([x_min, y_min, x_max, y_max])

        mask_pt[:, 0] = mask_pt[:, 0] - x_min
        mask_pt[:, 1] = mask_pt[:, 1] - y_min

        w, h = image_patch.size

        mask_pt = np.clip(mask_pt, a_min=[0, 0], a_max=[w, h])
        # mask_pt = np.uint(mask_pt)

        mask_patch = np.zeros((h, w))
        # image_patch.save("vis_crop.png")
        cv2.fillPoly(mask_patch, [mask_pt], 255)
        # cv2.imwrite("vis_patch.png", mask_patch)

        mask_patch = Image.fromarray(np.uint8(mask_patch))
        # mask_patch.save("vis_patch_2.png")
        # ipdb.set_trace()
        image_patch = image_patch.resize((512, 512))
        mask_patch = mask_patch.resize((512, 512))

        res_batch_images.append(image_patch)
        res_batch_masks.append(mask_patch)
        res_batch_labels.append(label)
    return res_batch_images, res_batch_masks, res_batch_labels


def no_crop_get_chosen_regions(image, data_item, chosen_index):
    res_batch_images, res_batch_masks, res_batch_labels = [], [], []

    for index in chosen_index:
        region_pt = np.array(data_item["text_region"][index])
        region_pt = np.uint(np.array([[region_pt[0], region_pt[1]], [region_pt[2], region_pt[1]],
                                      [region_pt[2], region_pt[3]], [region_pt[0], region_pt[3]]]))
        mask_pt = np.array(data_item["text_masks"][index])
        label = data_item["text_label"][index]
        # label = "Write english text '" + label + "'."
        label = "A scene image with english text '" + label + "'."

        # cv2_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # draw_region = region_pt[:, None, :]
        # draw_mask = mask_pt[:, None, :]
        # cv2.polylines(cv2_img, [draw_region], True, (0, 255, 0), 3)
        # cv2.polylines(cv2_img, [draw_mask], True, (0, 0, 255), 3)
        # cv2.imwrite("vis_img_2.png", cv2_img)

        x_min, y_min = min(region_pt[:, 0]), min(region_pt[:, 1])
        x_max, y_max = max(region_pt[:, 0]), max(region_pt[:, 1])

        image_patch = image

        mask_pt[:, 0] = mask_pt[:, 0]
        mask_pt[:, 1] = mask_pt[:, 1]

        w, h = image_patch.size

        mask_patch = np.zeros((h, w))
        # image_patch.save("vis_crop.png")
        cv2.fillPoly(mask_patch, [mask_pt], 255)
        # cv2.imwrite("vis_patch.png", mask_patch)

        mask_patch = Image.fromarray(np.uint8(mask_patch))
        # mask_patch.save("vis_patch_2.png")
        # ipdb.set_trace()
        image_patch = image_patch.resize((512, 512))
        mask_patch = mask_patch.resize((512, 512))

        res_batch_images.append(image_patch)
        res_batch_masks.append(mask_patch)
        res_batch_labels.append(label)
    return res_batch_images, res_batch_masks, res_batch_labels


def bezier_edge(label_item, size):
    text_polygon = label_item["text_masks"]
    width, height = size
    curved_polygon = []

    for single_polygon in text_polygon:
        single_polygon = np.array(single_polygon)
        mid_upper = (single_polygon[0] + single_polygon[1]) / 2
        mid_lower = (single_polygon[2] + single_polygon[3]) / 2

        random_strength = random.uniform(2, 6)
        offset = (single_polygon[2][1] + single_polygon[3][1]
                  - single_polygon[1][1] - single_polygon[0][1]) / random_strength
        random_offset = random.uniform(-offset, offset)
        ctrl_p1 = mid_upper + np.array([0, random_offset])
        ctrl_p2 = mid_lower + np.array([0, random_offset])

        x1, y1 = bezier_curve(np.array([single_polygon[0], ctrl_p1, ctrl_p1, single_polygon[1]]))
        x2, y2 = bezier_curve(np.array([single_polygon[2], ctrl_p2, ctrl_p2, single_polygon[3]]))

        upper_edge_points = np.column_stack((x1, y1))
        lower_edge_points = np.column_stack((x2, y2))
        mask_pt = np.concatenate((upper_edge_points, lower_edge_points), axis=0)

        mask_pt = np.clip(mask_pt, a_min=[0, 0], a_max=[width, height])
        mask_pt = np.uint(mask_pt)
        mask_pt = mask_pt.tolist()
        curved_polygon.append(mask_pt)
    label_item["text_masks"] = curved_polygon

    return label_item
