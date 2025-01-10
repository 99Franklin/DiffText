import os
import cv2
import ipdb
import math
import random
import jsonlines
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip
from torchvision import utils as vutils


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


class NewParadigmTextDataset(Dataset):

    def __init__(self, data_dir, resolution=512):

        self.data_dir = data_dir

        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 224)
        self.transforms = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
            ]
        )

        self.data_list = []
        with open(os.path.join(data_dir, "metadata.jsonl"), "r", encoding='utf-8') as f:
            for line_item in jsonlines.Reader(f):
                self.data_list.append(line_item)

        # self.data_list = self.data_list[::100]
        # self.data_list = self.data_list[::50]
        # self.data_list = self.data_list[::7]
        print("Total data num: ", len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_item = self.data_list[idx]

        image = Image.open(os.path.join(self.data_dir, data_item["file_name"])).convert("RGB")
        mask = Image.open(os.path.join(self.data_dir, data_item["mask_name"])).convert("L")

        zoom_up_polys = self.get_zoom_up_polys(mask)
        if zoom_up_polys is not None:
            # image.save("ori_image.png")
            # mask.save("ori_mask.png")
            image = image.crop(zoom_up_polys).resize((512, 512))
            mask = mask.crop(zoom_up_polys).resize((512, 512))
            # image.save("res_image.png")
            # mask.save("res_mask.png")
            # print(data_item)
            # ipdb.set_trace()
        # skeleton = self.get_text_skeleton(data_item["text"])

        # vis_image = Image.open(os.path.join(self.data_dir, data_item["file_name"])).convert("RGB").resize((512,512))
        # vis_image.save("res_ori.png")

        skeleton = self.get_mask_position_guide(data_item["text"], mask, image)

        image, mask = self.do_image_transform(image, mask)

        # text = "Fill the region, and keep the background unchanged."

        # text = [char for char in data_item["text"]]
        # text = " ".join(text)
        # text = "Write some characters '" + text + "'"

        # text = "Write english text '" + data_item["text"] + "'."

        #################################
        # pure text
        # text = data_item["text"]
        text = "A scene image with english text '" + data_item["text"] + "'."
        #################################

        sample = {'pixel_values': image, 'input_ids': text, 'mask': mask, 'ori_text': data_item["text"], 'masked_image': skeleton}

        return sample

    def do_image_transform(self, image, mask):
        image = self.transforms(image)
        mask = self.transforms(mask)

        # if random.random() > 0.5:
        #     image = hflip(image)
        #     mask = hflip(mask)

        image = np.array(image).transpose(2, 0, 1)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask)
        mask = mask.astype(np.float32) / 255.0

        mask = mask[None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        return image, mask

    def get_text_skeleton(self, text):
        font_w, font_h = self.font.getsize(text)

        img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
        img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

        # Set up image properties
        text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Determine text position
        text_width, text_height = draw.textsize(text, font=self.font)

        x_pos = (img_width - text_width) / 2
        y_pos = (img_height - text_height) / 2 - 10

        # Draw text on image
        draw.text((x_pos, y_pos), text, font=self.font, fill=(255, 255, 255))
        text_image = text_image.resize((512, 256))

        text_image = text_image.convert("L")
        text_image = np.array(text_image)
        text_image = text_image.astype(np.float32) / 255.0

        text_image = text_image[None]
        text_image[text_image < 0.5] = 0
        text_image[text_image >= 0.5] = 1
        text_image = torch.from_numpy(text_image)

        return text_image

    def get_position_guide(self, text, mask_image):
        font_w, font_h = self.font.getsize(text)

        img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
        img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

        # Set up image properties
        text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Determine text position
        text_width, text_height = draw.textsize(text, font=self.font)

        x_pos = (img_width - text_width) / 2
        y_pos = (img_height - text_height) / 2 - 10

        # Draw text on image
        draw.text((x_pos, y_pos), text, font=self.font, fill=(255, 255, 255))
        text_image = text_image.resize((512, 256))
        text_image = text_image.convert("L")

        text_image = np.asarray(text_image)

        mask_image = mask_image.resize(((512, 512)))
        mask_image = np.asarray(mask_image)

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

        res_img[text_image_warped == 255] = 255

        # cv2.imwrite("res.png", res_img)

        res_img = res_img.astype(np.float32) / 255.0
        res_img = torch.from_numpy(res_img).unsqueeze(0)

        return res_img

    def get_mask_position_guide(self, text, mask_image, ori_image):
        font_w, font_h = self.font.getsize(text)

        img_height = int(font_h * (1 + 0.05 * (len(text) - 1)))
        img_width = int(font_w * (1 + math.exp(0.8 * (2 - len(text)))))

        # Set up image properties
        text_image = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(text_image)

        # Determine text position
        text_width, text_height = draw.textsize(text, font=self.font)

        x_pos = (img_width - text_width) / 2
        y_pos = (img_height - text_height) / 2 - 10

        # Draw text on image
        draw.text((x_pos, y_pos), text, font=self.font, fill=(255, 255, 255))
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
            # text_image = Image.new('RGB', (512, 512), color=(0, 0, 0))
            #
            # # text_image = text_image.convert("L")
            # text_image = np.array(text_image).transpose(2, 0, 1)
            # text_image = text_image.astype(np.float32) / 255.0
            #
            # # text_image = text_image[None]
            # text_image[text_image < 0.5] = 0
            # text_image[text_image >= 0.5] = 1
            # text_image = torch.from_numpy(text_image)
            # return text_image

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
        # ipdb.set_trace()
        # cv2.imwrite("res.png", ori_image)
        # ipdb.set_trace()

        ori_image = ori_image.transpose(2, 0, 1)

        # ori_image = ori_image.astype(np.float32) / 255.0
        ori_image = ori_image.astype(np.float32) / 127.5 - 1.0
        ori_image = torch.from_numpy(ori_image)

        return ori_image

    def get_zoom_up_polys(self, ori_mask):
        # img_width, img_height = ori_mask.size
        bbox = ori_mask.getbbox()

        if bbox is None:
            print("Error Mask")
            return None

        x1, y1, x2, y2 = bbox
        # 获取mask区域的宽和高
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        bound_len = 256
        if width > 256 or height > 256:
            bound_len = max(width, height)

        new_x1, new_y1 = center_x - bound_len // 2 + 1, center_y - bound_len // 2 + 1
        new_x2, new_y2 = new_x1 + bound_len, new_y1 + bound_len

        if new_x1 < 0:
            new_x2 += abs(new_x1)
            new_x1 = 0
        if new_y1 < 0:
            new_y2 += abs(new_y1)
            new_y1 = 0
        if new_x2 > ori_mask.width:
            new_x1 -= new_x2 - ori_mask.width
            new_x2 = ori_mask.width
        if new_y2 > ori_mask.height:
            new_y1 -= new_y2 - ori_mask.height
            new_y2 = ori_mask.height

        new_x1 = max(1, new_x1)
        new_y1 = max(1, new_y1)
        new_x2 = min(ori_mask.width - 1, new_x2)
        new_y2 = min(ori_mask.height - 1, new_y2)

        return new_x1, new_y1, new_x2, new_y2
