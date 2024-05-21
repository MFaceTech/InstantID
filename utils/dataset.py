import os
import random
import json
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import numpy as np
import cv2
import math
import PIL


def crop_with_expanded_size(image, crop_coords, expand_factor=1.1):
    # 打开图像文件
    # img = Image.open(path)

    # 原始图像尺寸
    original_width, original_height = image.size

    # 已知的裁剪坐标 (left, top, right, bottom)
    # crop_coords = (left, top, right, bottom)

    # 计算原始裁剪区域的中心点
    center_x = (crop_coords[0] + crop_coords[2]) / 2
    center_y = (crop_coords[1] + crop_coords[3]) / 2

    # 计算原始裁剪区域的宽度和高度
    original_crop_width = crop_coords[2] - crop_coords[0]
    original_crop_height = crop_coords[3] - crop_coords[1]

    # 计算新的裁剪区域的宽度和高度
    new_crop_width = original_crop_width * expand_factor
    new_crop_height = original_crop_height * expand_factor

    # 计算新的裁剪坐标，确保不会超出图像边界
    new_left = max(center_x - new_crop_width / 2, 0)
    new_top = max(center_y - new_crop_height / 2, 0)
    new_right = min(center_x + new_crop_width / 2, original_width)
    new_bottom = min(center_y + new_crop_height / 2, original_height)

    # 新的裁剪坐标
    new_crop_coords = (int(new_left), int(new_top), int(new_right), int(new_bottom))

    # 裁剪图像
    cropped_img = image.crop(new_crop_coords)

    return cropped_img


class CropToRatioTransform(object):
    def __init__(self, target_aspect_ratio=512 / 640):
        self.target_aspect_ratio = target_aspect_ratio

    def __call__(self, img):
        # 计算当前宽高比
        current_w, current_h = img.size
        current_aspect_ratio = current_w / current_h
        # print(current_aspect_ratio)

        # 如果当前宽高比大于目标宽高比，则截取宽度至目标宽高比
        if current_aspect_ratio > self.target_aspect_ratio:
            # 计算目标宽度
            target_w = int(current_h * self.target_aspect_ratio)
            # 计算需要截取的区域
            left = (current_w - target_w) // 2
            right = left + target_w
            # 截取图像
            img = img.crop((left, 0, right, current_h))

        return img


class TopCropTransform(object):
    def __init__(self, crop_size):
        # crop_size可以是单个整数或包含两个整数的元组/列表
        if isinstance(crop_size, int):
            self.crop_height = crop_size
            self.crop_width = crop_size
        elif isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
            self.crop_height, self.crop_width = crop_size
        else:
            raise TypeError('crop_size must be an int or a list/tuple of length 2.')

    def __call__(self, img):
        # 检查提供的crop_size是否不大于图像的尺寸
        w, h = img.size
        if self.crop_width > w or self.crop_height > h:
            raise ValueError('crop_size must be smaller than the dimensions of the image.')

        top = 0
        center = w // 2
        crop_width, crop_height = self.crop_width, self.crop_height
        left = center - crop_width // 2

        # 防止坐标超出图像边界
        left = max(0, left)
        right = min(w, left + crop_width)
        bottom = min(h, top + crop_height)

        # 执行裁剪
        img = img.crop((left, top, right, bottom))
        return img


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0,
                                   360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, crop_size=(640, 512), center_crop=False, image_root_path=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size  #短边缩放到size
        self.image_root_path = image_root_path
        # 创建一个空列表来存储解析后的数据
        self.data = []
        # 读取并解析JSON文件的每一行
        with open(json_file, 'r') as f:
            for line in f:
                # 解析JSON数据并添加到列表中
                self.data.append(json.loads(line))

        self.transform = transforms.Compose([
            CropToRatioTransform(target_aspect_ratio=crop_size[1] / crop_size[0]),
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else TopCropTransform(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.kps_transform = transforms.Compose([
            CropToRatioTransform(target_aspect_ratio=crop_size[1] / crop_size[0]),
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else TopCropTransform(crop_size),
            transforms.ToTensor(),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        image_file = item["file_name"]
        text = item["additional_feature"]
        bbox = item['bbox']
        landmarks = item['landmarks']
        feature_file = item["penult_id_embed_file"]
        clip_from_seg_file = item["clip_from_seg_file"]
        clip_from_orig_file = item["clip_from_orig_file"]
        seg_map_orig_file = item["seg_map_orig_file"]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        kps_image = draw_kps(raw_image.convert("RGB"), landmarks)
        kps_image = self.kps_transform(kps_image)

        # crop image to clip
        crop_image = crop_with_expanded_size(raw_image, bbox)
        clip_image = self.clip_image_processor(images=crop_image, return_tensors="pt").pixel_values
        # load face feature
        face_id_embed = torch.load(os.path.join(self.image_root_path, feature_file), map_location="cpu")
        face_id_embed = torch.from_numpy(face_id_embed)

        # 定义所有可能的丢弃组合及其概率
        drop_combinations = {
            ('text',): 0.05,
            ('feature',): 0.05,
            ('feature', 'text'): 0.05,
        }
        # drop_combinations = {
        #     ('text',): 0.05,
        #     ('feature',): 0.04,
        #     ('image',): 0.04,
        #     ('image', 'feature'): 0.03,
        #     ('image', 'text'): 0.03,
        #     ('feature', 'text'): 0.03,
        #     ('image', 'feature', 'text'): 0.03
        # }
        # 计算剩余概率
        remaining_probability = 1 - sum(drop_combinations.values())
        # 添加新的键值对，对应不丢弃任何条件
        drop_combinations[()] = remaining_probability
        # 根据概率选择一个丢弃组合
        drop_choice = random.choices(list(drop_combinations.keys()), weights=list(drop_combinations.values()), k=1)[0]
        # 根据选择的组合来丢弃对象
        drop_text_embed = int('text' in drop_choice)
        drop_feature_embed = int('feature' in drop_choice)
        drop_image_embed = int('image' in drop_choice)

        # CFG处理
        if drop_text_embed:
            text = ""
        if drop_feature_embed:
            face_id_embed = torch.zeros_like(face_id_embed)
        if drop_image_embed:
            pass  # drop in train loop

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "kps_image": kps_image,
            "clip_image": clip_image,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    kps_images = torch.stack([example["kps_image"] for example in data])

    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)

    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "kps_images": kps_images,
        "clip_images": clip_images,
        "text_input_ids": text_input_ids,
        "face_id_embed": face_id_embed,
        "drop_image_embeds": drop_image_embeds
    }
