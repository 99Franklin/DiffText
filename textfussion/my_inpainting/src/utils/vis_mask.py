import cv2
import ipdb
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image

img_path = "/data/lfu/datasets/scene_text_detection/for_diffusers_inpainting_char/mask/100/leopard_140_0_2.png"

mask = Image.open(img_path).convert("L").resize((512,512))

mask = np.array(mask)
mask = mask.astype(np.float32) / 255.0

mask = mask[None]
mask[mask < 0.5] = 0
mask[mask >= 0.5] = 1
mask = torch.from_numpy(mask)

vutils.save_image(mask, 'ori.png', normalize=True)

mask = mask.unsqueeze(0)
mask = torch.nn.functional.interpolate(mask, size=(64, 64))
# mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode="area")
# mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode="bilinear")

vutils.save_image(mask[0], 'res_trilinear.png', normalize=True)

