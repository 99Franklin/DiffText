import cv2
import ipdb
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


glyph_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 224)


def decide_box_seq(points):
    centre_x, centre_y = sum([x for x, _ in points])/len(points), sum([y for _, y in points])/len(points)
    # print("\n")
    # print(centre_x)
    # print(points)
    if points[1][0] > centre_x:
        # print("top_left")
        points = [points[0], points[1], points[2], points[3]]
    else:
        # print("bottom_left")
        points = [points[1], points[2], points[3], points[0]]

    # print(points)
    # print("\n")
    return points


def get_test_mask_image(ori_image, mask_image, text):
    text = "'".join(text.split("'")[1:-1])
    font_w, font_h = glyph_font.getsize(text)

    img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
    img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

    # Set up image properties
    text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(text_image)

    # Determine text position
    text_width, text_height = draw.textsize(text, font=glyph_font)

    x_pos = (img_width - text_width) / 2
    y_pos = (img_height - text_height) / 2 - 10

    # Draw text on image
    draw.text((x_pos, y_pos), text, font=glyph_font, fill=(255, 255, 255))
    text_image = text_image.resize((512, 256))
    text_image = text_image.convert("L")

    text_image = np.asarray(text_image)

    mask_image = mask_image.resize((512, 512))
    mask_image = np.asarray(mask_image)

    ori_image = ori_image.resize((512, 512))
    ori_image = cv2.cvtColor(np.asarray(ori_image), cv2.COLOR_RGB2BGR)

    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        print("A mask error!")

        ori_image[mask_image == 255] = [0, 0, 0]

        ori_image = ori_image.transpose(2, 0, 1)
        # ori_image = ori_image.astype(np.float32) / 255.0
        ori_image = ori_image.astype(np.float32) / 127.5 - 1.0
        ori_image = torch.from_numpy(ori_image)

        return ori_image

    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # vis_mask_image = mask_image
    # cv2.circle(vis_mask_image, box[0], 5, (255, 0, 0), 2)
    # cv2.polylines(vis_mask_image, pts=[box], isClosed=True, color=(255, 0, 0), thickness=2)
    # cv2.imwrite("vis_mask_image.png", vis_mask_image)

    box = decide_box_seq(box)
    # box = sort_counterclockwise(box)
    box = np.array(box)

    res_img = np.zeros_like(mask_image)
    # cv2.fillPoly(res_img, [box], 127)

    pts_dst = np.array(
        [[0, 0], [text_image.shape[1], 0], [text_image.shape[1], text_image.shape[0]], [0, text_image.shape[0]]])

    M = cv2.getPerspectiveTransform(pts_dst.astype(np.float32), box.astype(np.float32))
    text_image_warped = cv2.warpPerspective(text_image, M, (res_img.shape[1], res_img.shape[0]))

    cv2.fillPoly(ori_image, [box], [0, 0, 0])

    ori_image[text_image_warped == 255] = [255, 255, 255]

    ori_image = np.array(ori_image)
    # cv2.imwrite("res.png", ori_image)
    vis_image = ori_image
    # ori_image = ori_image.transpose(2, 0, 1)
    #
    # # ori_image = ori_image.astype(np.float32) / 255.0
    # ori_image = ori_image.astype(np.float32) / 127.5 - 1.0
    # ori_image = torch.from_numpy(ori_image)

    ori_image = ori_image.transpose(2, 0, 1)
    ori_image = ori_image.astype(np.float32) / 127.5 - 1.0

    return ori_image, vis_image
