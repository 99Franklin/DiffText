import os
import cv2
import ipdb
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

template_font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 224)


def get_text_image(text, is_blank=None):
    if is_blank:
        text_image = Image.new('RGB', (512, 256), color=(0, 0, 0))

        text_image = text_image.convert("L")
        text_image = np.array(text_image)
        text_image = text_image.astype(np.float32) / 255.0

        text_image = text_image[None]
        text_image[text_image < 0.5] = 0
        text_image[text_image >= 0.5] = 1
        text_image = torch.from_numpy(text_image)

        return text_image

    font_w, font_h = template_font.getsize(text)

    img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
    img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

    # Set up image properties
    text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
    draw = ImageDraw.Draw(text_image)

    # Determine text position
    text_width, text_height = draw.textsize(text, font=template_font)

    x_pos = (img_width - text_width) / 2
    y_pos = (img_height - text_height) / 2 - 10

    # Draw text on image
    draw.text((x_pos, y_pos), text, font=template_font, fill=(255, 255, 255))
    text_image = text_image.resize((512, 256))

    text_image = text_image.convert("L")
    text_image = np.array(text_image)
    text_image = text_image.astype(np.float32) / 255.0

    text_image = text_image[None]
    text_image[text_image < 0.5] = 0
    text_image[text_image >= 0.5] = 1
    text_image = torch.from_numpy(text_image)

    return text_image


def sort_counterclockwise(points):
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
    angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    counterclockwise_points = [points[i] for i in counterclockwise_indices]

    return counterclockwise_points


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


def get_pos_image(text, mask_image, is_blank=None):
    if is_blank:
        if isinstance(mask_image, list):
            text_images = []
            for single_mask_image in mask_image:
                text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))

                text_image = text_image.convert("L")
                text_image = np.array(text_image)
                # cv2.imwrite("temp_zero_pose.png", text_image)
                text_image = text_image.astype(np.float32) / 255.0

                text_image = text_image[None]
                text_image[text_image < 0.5] = 0
                text_image[text_image >= 0.5] = 1
                text_image = torch.from_numpy(text_image)
                text_images.append(text_image)

            text_images = torch.stack(text_images)
            # text_images = torch.cat(text_images, dim=0)
            return text_images

        else:
            text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))

            text_image = text_image.convert("L")
            text_image = np.array(text_image)
            # cv2.imwrite("temp_zero_pose.png", text_image)
            text_image = text_image.astype(np.float32) / 255.0

            text_image = text_image[None]
            text_image[text_image < 0.5] = 0
            text_image[text_image >= 0.5] = 1
            text_image = torch.from_numpy(text_image)

            return text_image

    if isinstance(text, list):
        res_imgs = []
        for (single_text, single_mask_image) in zip(text, mask_image):
            # single_mask_image.save("temp_mask.png")
            font_w, font_h = template_font.getsize(single_text)

            img_height = int(font_h * (1 + 0.05 * (len(single_text) - 1)))
            img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(single_text)))))

            # Set up image properties
            text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
            draw = ImageDraw.Draw(text_image)

            # Determine text position
            text_width, text_height = draw.textsize(single_text, font=template_font)

            x_pos = (img_width - text_width) / 2
            y_pos = (img_height - text_height) / 2 - 10

            # Draw text on image
            draw.text((x_pos, y_pos), single_text, font=template_font, fill=(255, 255, 255))
            # text_image = text_image.resize((512, 256))
            text_image = text_image.resize((512, 512))

            text_image = np.asarray(text_image.convert("L"))
            single_mask_image = np.asarray(single_mask_image.convert("L"))

            contours, hierarchy = cv2.findContours(single_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) < 1:
                print("A mask error!")
                text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))

                text_image = text_image.convert("L")
                text_image = np.array(text_image)
                text_image = text_image.astype(np.float32) / 255.0

                text_image = text_image[None]
                text_image[text_image < 0.5] = 0
                text_image[text_image >= 0.5] = 1
                text_image = torch.from_numpy(text_image)

                res_imgs.append(text_image)
                continue

            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # box = sort_counterclockwise(box)
            box = decide_box_seq(box)
            box = np.array(box)

            res_img = np.zeros_like(single_mask_image)
            # cv2.fillPoly(res_img, [box], 127)

            pts_dst = np.array(
                [[0, 0], [text_image.shape[1], 0], [text_image.shape[1], text_image.shape[0]], [0, text_image.shape[0]]])
            M = cv2.getPerspectiveTransform(pts_dst.astype(np.float32), box.astype(np.float32))
            text_image_warped = cv2.warpPerspective(text_image, M, (res_img.shape[1], res_img.shape[0]))

            res_img[text_image_warped == 255] = 255

            # cv2.imwrite("temp_pose_img.png", res_img)
            # ipdb.set_trace()
            res_img = res_img.astype(np.float32) / 255.0
            res_img = torch.from_numpy(res_img).unsqueeze(0)

            res_imgs.append(res_img)
        # res_imgs = torch.cat(res_imgs, dim=0)
        res_imgs = torch.stack(res_imgs)

        return res_imgs

    else:

        font_w, font_h = template_font.getsize(text)

        img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
        img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

        # Set up image properties
        text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Determine text position
        text_width, text_height = draw.textsize(text, font=template_font)

        x_pos = (img_width - text_width) / 2
        y_pos = (img_height - text_height) / 2 - 10

        # Draw text on image
        draw.text((x_pos, y_pos), text, font=template_font, fill=(255, 255, 255))
        # text_image = text_image.resize((512, 256))
        text_image = text_image.resize((512, 512))

        text_image = np.asarray(text_image.convert("L"))
        mask_image = np.asarray(mask_image.convert("L"))

        contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            print("A mask error!")
            text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))

            text_image = text_image.convert("L")
            text_image = np.array(text_image)
            text_image = text_image.astype(np.float32) / 255.0

            text_image = text_image[None]
            text_image[text_image < 0.5] = 0
            text_image[text_image >= 0.5] = 1
            text_image = torch.from_numpy(text_image)

            return text_image

        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # box = sort_counterclockwise(box)
        box = decide_box_seq(box)
        box = np.array(box)

        res_img = np.zeros_like(mask_image)
        # cv2.fillPoly(res_img, [box], 127)

        pts_dst = np.array(
            [[0, 0], [text_image.shape[1], 0], [text_image.shape[1], text_image.shape[0]], [0, text_image.shape[0]]])
        M = cv2.getPerspectiveTransform(pts_dst.astype(np.float32), box.astype(np.float32))
        text_image_warped = cv2.warpPerspective(text_image, M, (res_img.shape[1], res_img.shape[0]))

        res_img[text_image_warped == 255] = 255

        # cv2.imwrite("temp_pose_img.png", res_img)

        res_img = res_img.astype(np.float32) / 255.0
        res_img = torch.from_numpy(res_img).unsqueeze(0)

        return res_img


def get_mask_pose_image(text, mask_images, ori_images, is_blank=None):
    if is_blank:

        text_images = []
        for single_mask, single_image in zip(mask_images, ori_images):
            # ipdb.set_trace()
            single_image = cv2.cvtColor(np.asarray(single_image), cv2.COLOR_RGB2BGR)
            single_mask = np.asarray(single_mask)
            cv2.imwrite("vis_ori.png", single_image)
            single_image[single_mask == 255] = [0, 0, 0]

            cv2.imwrite("temp_blank.png", single_image)
            ipdb.set_trace()
            single_image = single_image.transpose(2, 0, 1)

            single_image = single_image.astype(np.float32) / 255.0
            single_image = torch.from_numpy(single_image)
            text_images.append(single_image)

        text_images = torch.stack(text_images)
        # text_images = torch.cat(text_images, dim=0)
        return text_images

    res_imgs = []
    for (single_text, single_mask_image, single_ori_image) in zip(text, mask_images, ori_images):
        # single_mask_image.save("temp_mask.png")
        font_w, font_h = template_font.getsize(single_text)

        img_height = int(font_h * (1 + 0.05 * (len(single_text) - 1)))
        img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(single_text)))))

        # Set up image properties
        text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Determine text position
        text_width, text_height = draw.textsize(single_text, font=template_font)

        x_pos = (img_width - text_width) / 2
        y_pos = (img_height - text_height) / 2 - 10

        # Draw text on image
        draw.text((x_pos, y_pos), single_text, font=template_font, fill=(255, 255, 255))
        # text_image = text_image.resize((512, 256))
        text_image = text_image.resize((512, 512))

        text_image = np.asarray(text_image.convert("L"))
        single_mask_image = np.asarray(single_mask_image.convert("L"))

        single_ori_image = cv2.cvtColor(np.asarray(single_ori_image), cv2.COLOR_RGB2BGR)

        contours, hierarchy = cv2.findContours(single_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 1:
            print("A mask error!")
            text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))

            text_image = text_image.convert("L")
            text_image = np.array(text_image)
            text_image = text_image.astype(np.float32) / 255.0

            text_image = text_image[None]
            text_image[text_image < 0.5] = 0
            text_image[text_image >= 0.5] = 1
            text_image = torch.from_numpy(text_image)

            res_imgs.append(text_image)
            continue

        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # box = sort_counterclockwise(box)
        box = decide_box_seq(box)
        box = np.array(box)

        res_img = np.zeros_like(single_mask_image)
        # cv2.fillPoly(res_img, [box], 127)

        pts_dst = np.array(
            [[0, 0], [text_image.shape[1], 0], [text_image.shape[1], text_image.shape[0]], [0, text_image.shape[0]]])
        M = cv2.getPerspectiveTransform(pts_dst.astype(np.float32), box.astype(np.float32))
        text_image_warped = cv2.warpPerspective(text_image, M, (res_img.shape[1], res_img.shape[0]))

        cv2.fillPoly(single_ori_image, [box], [0, 0, 0])
        single_ori_image[text_image_warped == 255] = [255, 255, 255]
        # res_img[text_image_warped == 255] = 255

        # cv2.imwrite("temp_pose_img.png", res_img)
        # ipdb.set_trace()
        # res_img = res_img.astype(np.float32) / 255.0
        # res_img = torch.from_numpy(res_img).unsqueeze(0)

        # cv2.imwrite("temp_image.png", single_ori_image)
        # ipdb.set_trace()

        single_ori_image = single_ori_image.transpose(2, 0, 1)

        single_ori_image = single_ori_image.astype(np.float32) / 255.0
        single_ori_image = torch.from_numpy(single_ori_image)

        res_imgs.append(single_ori_image)
    # res_imgs = torch.cat(res_imgs, dim=0)
    res_imgs = torch.stack(res_imgs)

    return res_imgs
