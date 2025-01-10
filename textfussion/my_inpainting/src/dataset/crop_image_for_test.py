import cv2
import math
import ipdb
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def crop_image_for_test(image, mask):
    bbox = mask.getbbox()

    if bbox is None:
        return image, mask.convert("RGB")

    x1, y1, x2, y2 = bbox
    # 获取mask区域的宽和高
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    bound_len = 256
    if width > 196 or height > 196:
        bound_len = max(2 * width, 2 * height)

    new_x1, new_y1 = center_x - bound_len // 2 + 1, center_y - bound_len // 2 + 1
    new_x2, new_y2 = new_x1 + bound_len, new_y1 + bound_len

    if new_x1 < 0:
        new_x2 += abs(new_x1)
        new_x1 = 0
    if new_y1 < 0:
        new_y2 += abs(new_y1)
        new_y1 = 0
    if new_x2 > mask.width:
        new_x1 -= new_x2 - mask.width
        new_x2 = mask.width
    if new_y2 > mask.height:
        new_y1 -= new_y2 - mask.height
        new_y2 = mask.height

    new_x1 = max(1, new_x1)
    new_y1 = max(1, new_y1)
    new_x2 = min(mask.width - 1, new_x2)
    new_y2 = min(mask.height - 1, new_y2)

    zoom_up_polys = [new_x1, new_y1, new_x2, new_y2]

    image = image.crop(zoom_up_polys).resize((512, 512))
    mask = mask.crop(zoom_up_polys).resize((512, 512)).convert("RGB")

    return zoom_up_polys, image, mask


def get_rotated_rect_mask(mask, ratio):
    # ipdb.set_trace()
    # mask.save("r_ori_mask.png")
    ratio = 0.6 * ratio
    mask_array = np.array(mask)
    mask_array = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)

    zero_image = np.zeros_like(mask)

    # 计算最小外接矩形
    gray = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    bbox = cv2.boxPoints(rect)

    # ipdb.set_trace()
    # temp_bbox = np.int0(bbox)[:, None, :]
    # cv2.drawContours(mask_array, [temp_bbox], 0, 255, 3)

    bbox = np.int0(bbox)

    cx, cy = rect[0][0], rect[0][1]

    if bbox[0][0] < cx and bbox[1][0] < cx:
        bbox = [bbox[1], bbox[2], bbox[3], bbox[0]]

    angle = (math.atan2(bbox[1][1] - bbox[0][1], bbox[1][0] - bbox[0][0])
             + math.atan2(bbox[2][1] - bbox[3][1], bbox[2][0] - bbox[3][0])) / 2
    mask_w = 0.9 * (math.sqrt((bbox[1][1] - bbox[0][1]) ** 2 + (bbox[1][0] - bbox[0][0]) ** 2) +
                    math.sqrt((bbox[3][1] - bbox[2][1]) ** 2 + (bbox[3][0] - bbox[2][0]) ** 2)) / 2
    mask_h = 0.9 * (math.sqrt((bbox[0][1] - bbox[3][1]) ** 2 + (bbox[0][0] - bbox[3][0]) ** 2) +
                    math.sqrt((bbox[1][1] - bbox[2][1]) ** 2 + (bbox[1][0] - bbox[2][0]) ** 2)) / 2

    rect_ratio = mask_w / mask_h
    if rect_ratio > ratio:
        new_width = mask_h * ratio
        new_height = mask_h
    else:
        new_width = mask_w
        new_height = mask_w / ratio

    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    new_half_width = new_width / 2
    new_half_height = new_height / 2

    new_top_left_x = cx - cos_val * new_half_width + sin_val * new_half_height
    new_top_left_y = cy - sin_val * new_half_width - cos_val * new_half_height
    new_top_right_x = cx + cos_val * new_half_width + sin_val * new_half_height
    new_top_right_y = cy + sin_val * new_half_width - cos_val * new_half_height

    # temp_new_top_left_x, temp_new_top_left_y = np.int0(new_top_left_x), np.int0(new_top_left_y)
    # cv2.circle(mask_array, (temp_new_top_left_x, temp_new_top_left_y), 10, 255, -1)

    # temp_new_top_right_x, temp_new_top_right_y = np.int0(new_top_right_x), np.int0(new_top_right_y)
    # cv2.circle(mask_array, (temp_new_top_right_x, temp_new_top_right_y), 10, 255, -1)

    # 计算旋转后的左下角和右下角坐标
    new_bottom_left_x = new_top_left_x - sin_val * new_height
    new_bottom_left_y = new_top_left_y + cos_val * new_height
    new_bottom_right_x = new_top_right_x - sin_val * new_height
    new_bottom_right_y = new_top_right_y + cos_val * new_height

    # temp_new_bottom_left_x, temp_new_bottom_left_y = np.int0(new_bottom_left_x), np.int0(new_bottom_left_y)
    # cv2.circle(mask_array, (temp_new_bottom_left_x, temp_new_bottom_left_y), 10, 255, -1)

    # temp_new_bottom_right_x, temp_new_bottom_right_y = np.int0(new_bottom_right_x), np.int0(new_bottom_right_y)
    # cv2.circle(mask_array, (temp_new_bottom_right_x, temp_new_bottom_right_y), 10, 255, -1)
    # cv2.imwrite("r_circle.png", mask_array)
    rect_new = [[new_top_left_x, new_top_left_y], [new_top_right_x, new_top_right_y],
                [new_bottom_right_x, new_bottom_right_y], [new_bottom_left_x, new_bottom_left_y]]

    rect_new = np.array(rect_new)[:, None, :]
    rect_new = np.int0(rect_new)
    cv2.fillPoly(zero_image, [rect_new], 255)
    # cv2.imwrite("r_refine_mask.png", zero_image)

    # ipdb.set_trace()
    return Image.fromarray(zero_image)
