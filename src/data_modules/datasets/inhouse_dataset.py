import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
import cv2

import albumentations as A
from albumentations.pytorch import transforms as A_transforms

from .util.mask import (
    bbox2mask,
    brush_stroke_mask,
    get_irregular_mask,
    random_bbox,
    random_cropping_bbox,
)
from .util.degradation import DE_process


IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, split):
    df = pd.read_csv(
        f"/data1/fundus_dataset/inhouse_dataset/label_files/finding_{split}.csv"
    )
    df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", dir))
    images = df["filename"]
    return images


def pil_loader(path):
    return Image.open(path).convert("RGB")


def cv2_loader(path):
    images = cv2.imread(path)
    return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)


class Fundus_InpaintDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        mask_mode=None,
        data_len=-1,
        image_size=[256, 256],
        loader=pil_loader,
    ):
        imgs = make_dataset(data_dir, split)
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.loader = loader
        self.mask_mode = mask_mode
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["mask_image"] = mask_img
        ret["mask"] = mask
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == "bbox":
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == "center":
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h // 4, w // 4, h // 2, w // 2))
        elif self.mask_mode == "irregular":
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == "free_form":
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == "hybrid":
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(
                self.image_size,
            )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1)


class Fundus_UncropDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        mask_mode=None,
        data_len=-1,
        image_size=[256, 256],
        loader=pil_loader,
    ):
        imgs = make_dataset(data_dir, split)
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.loader = loader
        self.mask_mode = mask_mode
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img)
        mask_img = img * (1.0 - mask) + mask

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["mask_image"] = mask_img
        ret["mask"] = mask
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == "manual":
            mask = bbox2mask(self.image_size, self.mask_config["shape"])
        elif self.mask_mode == "fourdirection" or self.mask_mode == "onedirection":
            mask = bbox2mask(
                self.image_size, random_cropping_bbox(mask_mode=self.mask_mode)
            )
        elif self.mask_mode == "hybrid":
            if np.random.randint(0, 2) < 1:
                mask = bbox2mask(
                    self.image_size, random_cropping_bbox(mask_mode="onedirection")
                )
            else:
                mask = bbox2mask(
                    self.image_size, random_cropping_bbox(mask_mode="fourdirection")
                )
        elif self.mask_mode == "file":
            pass
        else:
            raise NotImplementedError(
                f"Mask mode {self.mask_mode} has not been implemented."
            )
        return torch.from_numpy(mask).permute(2, 0, 1)


class Fundus_EnhancementDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        data_len=-1,
        image_size=[224, 224],
        loader=cv2_loader,
        **kwargs,
    ):
        self.data_dir = data_dir

        imgs = make_dataset(data_dir, split)
        if data_len > 0:
            self.imgs = imgs[: int(data_len)]
        else:
            self.imgs = imgs

        self.tfs = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=0.5, std=0.5),
                A_transforms.ToTensorV2(),
            ]
        )
        self.tfs_de = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.OneOf(
                    [
                        DE_process(de_type="001", p=1),  # 0.8),
                        DE_process(de_type="010", p=1),  # 0.8),
                        DE_process(de_type="100", p=1),  # 0.8),
                        DE_process(de_type="011", p=1),  # 0.8),
                        DE_process(de_type="101", p=1),  # 0.8),
                        DE_process(de_type="110", p=1),  # 0.8),
                        DE_process(de_type="111", p=1),  # 0.8),
                    ],
                    p=0.8,
                ),
                A.Normalize(mean=0.5, std=0.5),
                A_transforms.ToTensorV2(),
            ]
        )
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img_np = self.loader(path)
        img = self.tfs(image=img_np)["image"]
        cond_image = self.tfs_de(image=img_np)["image"]

        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = Fundus_EnhancementDataset(
        data_dir="/data1/fundus_dataset/inhouse_dataset",
        split="train",
    )
    print(len(dataset))
