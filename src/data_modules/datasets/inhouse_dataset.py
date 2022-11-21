import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import glob
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
from .util.region_mask_utils import generate_single_region_mask, encoding_region

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


def make_vessel_dataset(dir, split):
    df = pd.read_csv(
        f"/data1/fundus_dataset/inhouse_dataset/label_files/finding_{split}.csv"
    )
    df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext/resized_1024x1024", dir))
    fn_list = df["filename"]
    return fn_list

def make_optic_fovea_dataset(dir, split):
    df = pd.read_csv(
        f"/data1/fundus_dataset/inhouse_dataset/label_files/finding_{split}.csv"
    )
    df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", dir))
    info_list = df[["filename", "x_disc", "y_disc", "x_fovea", "y_fovea"]]
    return info_list


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
        **kwargs,
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
        **kwargs,
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
        ret_de_info=True,
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
        if split == 'train':
            prob = .5
        else:
            prob = .8
        self.tfs_de = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                DE_process(prob_illumination=prob, prob_spot=prob, prob_blur=prob),  # 0.8),
                A.Normalize(mean=0.5, std=0.5),
                A_transforms.ToTensorV2(),
            ]
        )
        self.ToTensor = transforms.ToTensor()
        self.tfs_de_mask = A.Compose(
            [ 
                A_transforms.ToTensorV2(),
            ]
        )
        self.loader = loader
        self.image_size = image_size
        self.ret_de_info = ret_de_info

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img_np = self.loader(path)
        img = self.tfs(image=img_np)["image"]
        cond_image = self.tfs_de(image=img_np)["image"]
        de_info = self.tfs_de.transforms[1].degradation_info
        # normalizing for [5,15] to [0, 1]
        blur_mask = np.ones((img.shape[1], img.shape[2])) * (
            de_info["blurness"] * 0.1 - 0.5
        )
        # spot mask highlight
        spot_mask = np.minimum(de_info["spot_mask"] * 10, 1)
        de_mask = [
            blur_mask,
            spot_mask,
            de_info["halo_mask"],
            de_info["hole_mask"],
        ]
        de_mask = self.tfs_de_mask(image=np.array(de_mask).transpose(1, 2, 0))["image"]
        ret["gt_image"] = img
        ret["cond_image"] = cond_image
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1]
        if self.ret_de_info:
            ret["de_mask"] = de_mask  # (4, 8, 8)
        return ret

    def __len__(self):
        return len(self.imgs)


class Fundus_DownstreamDataset(data.Dataset):
    def __init__(
        self,
        data_dir, 
        split,
        data_len=-1,
        image_size=[224, 224],
        loader=cv2_loader,
        ret_de_info=True,
        task=None,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.task = task
        self.split = split

        if split == 'train':
            prob = .5
        else:
            prob = .8

        if task == 'vessel':
            fn_list = make_vessel_dataset(data_dir, 'train')
            num = len(fn_list)
            if split=='train':
                self.imgs = list(fn_list[:int(num*0.8)])
            elif split == 'val':
                self.imgs = list(fn_list[int(num*0.8):int(num*0.9)])
            elif split == 'test':
                self.imgs = list(fn_list[int(num*0.9):])
            
            if data_len > 0:
                self.imgs = self.imgs[: int(data_len)]

            self.label_dir = "/data1/vessel_dataset/gt"
            if split == 'test':
                prob = 0.8
                self.tfs = A.Compose(
                    [
                        A.Resize(image_size[0], image_size[1]), 
                        DE_process(prob_illumination=prob, prob_spot=prob, prob_blur=prob),
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    additional_targets={f"vessel_mask": "mask"},
                )
                self.tfs_clean = A.Compose(
                    [
                        A.Resize(image_size[0], image_size[1]),  
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    additional_targets={f"vessel_mask": "mask"},
                )
            else:
                self.tfs = A.Compose(
                    [
                        A.RandomResizedCrop(image_size[0], image_size[1]),
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            rotate_limit=180, border_mode=0, value=0, mask_value=0, p=0.9
                        ),
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    additional_targets={f"vessel_mask": "mask"},
                )
        elif task == 'optic_fovea':  
            self.imgs = make_optic_fovea_dataset(data_dir, split) # df
            if data_len > 0:
                self.imgs = self.imgs[: int(data_len)]
            if split == 'test':
                prob = 0.8
                self.tfs = A.Compose(
                    [
                        A.Resize(image_size[0], image_size[1]), 
                        DE_process(prob_illumination=prob, prob_spot=prob, prob_blur=prob),
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"], remove_invisible=False),
                )
                self.tfs_clean = A.Compose(
                    [
                        A.Resize(image_size[0], image_size[1]),  
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"], remove_invisible=False),
                )
            else:
                self.tfs = A.Compose(
                    [
                        A.Resize(image_size[0], image_size[1]),
                        A.HorizontalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            rotate_limit=180, border_mode=0, value=0, mask_value=0, p=0.9
                        ),
                        A.Normalize(mean=0.5, std=0.5),
                        A_transforms.ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"], remove_invisible=False),
                ) 

        self.loader = loader  
 
    def __getitem__(self, index):
        ret = {}
        if self.task == 'vessel':
            path = self.imgs[index]
            img_np = self.loader(path)
            vessel_dir = path.replace("/data1/vessel_dataset/img", self.label_dir) 
            vessels = cv2.imread(vessel_dir)
            vessels_np = cv2.cvtColor(vessels, cv2.COLOR_RGB2GRAY) 

            # aug = self.tfs(image=img_np, vessel_mask=vessels_np)
            # mask = aug["vessel_mask"].float() / 255
            if self.split == 'test':
                fn_name = path.rsplit("/")[-1].rsplit("\\")[-1] 
                
                clean = self.tfs_clean(image=img_np, vessel_mask=vessels_np) 
                ret['clean_image'] = clean["image"]
                mask = clean["vessel_mask"].float() / 255
            mask[mask > 0.1] = 1
            mask[mask < 0.1] = 0
            ret["mask"] = mask.unsqueeze(0)
        elif self.task == 'optic_fovea':
            info = self.imgs.iloc[index]
            path = info['filename']
            img_np = self.loader(info['filename'])
            keypoints = [self.get_fovea_xy(info), self.get_disc_xy(info)]
            aug = self.tfs(
                image=img_np,
                keypoints=keypoints,
                class_labels=["fovea", "disc"], 
            )
            if self.split == 'test':
                clean = self.tfs_clean(image=img_np,
                    keypoints=keypoints,
                    class_labels=["fovea", "disc"]
                )
                ret['clean_image'] = clean["image"]
            ret['gt'] = torch.Tensor(aug["keypoints"]).view(-1)
            ret['gt'] = ret['gt'] / 256 
        img = aug["image"]

        ret["gt_image"] = img
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1] 

        return ret

    def __len__(self):
        return len(self.imgs)
  
    def get_disc_xy(self, info):
        return tuple(info[["x_disc", "y_disc"]].values.astype(int))

    def get_fovea_xy(self, info):
        return tuple(info[["x_fovea", "y_fovea"]].values.astype(int))

if __name__ == "__main__":
    dataset = Fundus_EnhancementDataset(
        data_dir="/data1/fundus_dataset/inhouse_dataset",
        split="train",
    )
    print(len(dataset))


def make_finding_inpainting_dataset(dir, split, mask_mode, task): 
    df = pd.read_csv(
        f'/data1/fundus_dataset/inhouse_dataset/label_files/finding_{split}.csv' 
    )
    df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", dir)) 

    if task == 'finding_inpainting_inference':
        neg_df = df.loc[df[f'{mask_mode}_region'] != df[f'{mask_mode}_region']]
        return neg_df[['filename', "x_disc", "y_disc", "x_fovea", "y_fovea"]]

    else:
        pos_df = df.loc[df[f'{mask_mode}_region'] == df[f'{mask_mode}_region']]
        return pos_df[['filename', mask_mode, f'{mask_mode}_region']]
 
class Fundus_Finding_InpaintDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        mask_mode=None,
        data_len=-1,
        image_size=[256, 256],
        loader=cv2_loader,
        task='finding_inpainting',
        **kwargs,
    ): 
        self.split = split
        self.task = task
        df = make_finding_inpainting_dataset(data_dir, split, mask_mode, task) 

        if data_len > 0:
            self.df = df.sample(n=data_len) 
        else:
            self.df = df
        self.tfs = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                A_transforms.ToTensorV2(), 
            ],
            additional_targets = {f"mask_{mask_mode}": "mask"}
        )
        self.loader = loader
        self.mask_mode = mask_mode
        # self.pseudo_mask_dir = '/home/sunhokim/repo/FUNDUS_FAI_DEV/pseudo_mask'
        self.image_size = image_size
        
    def __getitem__(self, index):
        info = self.df.iloc[index]
        fn = info['filename']
        if self.task == 'finding_inpainting_inference':  
            land_marks = list(
                map(int, info[["x_disc", "y_disc", "x_fovea", "y_fovea"]].tolist())
            ) 
            if self.mask_mode in ['NonGlaucomatousDiscChange', 'GlaucomatousDiscChange']:
                m = generate_single_region_mask(
                    (self.image_size[0], self.image_size[1]),
                    (land_marks[1] * 0.5, land_marks[0] * 0.5),
                    (land_marks[3] * 0.5, land_marks[2] * 0.5),
                    'ID',
                    )  + generate_single_region_mask(
                    (self.image_size[0], self.image_size[1]),
                    (land_marks[1] * 0.5, land_marks[0] * 0.5),
                    (land_marks[3] * 0.5, land_marks[2] * 0.5),
                    'SD',
                    ) 
            elif self.mask_mode in ['MacularHole']:
                m = generate_single_region_mask(
                    (self.image_size[0], self.image_size[1]),
                    (land_marks[1] * 0.5, land_marks[0] * 0.5),
                    (land_marks[3] * 0.5, land_marks[2] * 0.5),
                'M')
            mask = {f'mask_{self.mask_mode}': m}
        else:
            fn_mask = fn.replace("resized_1024x1024", "mask_1024x1024") + "_{}.npy".format(self.mask_mode)  
            assert os.path.isfile(fn_mask) == True
            mask = {f'mask_{self.mask_mode}':np.load(fn_mask)}

        ret = {} 
        img = self.loader(fn)
        aug = self.tfs(image=img, **mask)
        img = aug['image']
        mask = aug[f'mask_{self.mask_mode}'].to(torch.uint8)

        # mask = self.get_mask()
        cond_image = img * (1.0 - mask) + mask * torch.randn_like(img) 

        ret["gt_image"] = img
        ret["cond_image"] = cond_image 
        ret["mask"] = mask
        ret["path"] = fn.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.df)
 

class Fundus_BaseDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        split,
        data_len=-1,
        image_size=[224, 224],
        loader=cv2_loader,
        ret_de_info=True,
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
        self.ToTensor = transforms.ToTensor() 
        self.loader = loader
        self.image_size = image_size 

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img_np = self.loader(path)
        img = self.tfs(image=img_np)["image"]   
        ret["gt_image"] = img 
        ret["path"] = path.rsplit("/")[-1].rsplit("\\")[-1] 
        return ret

    def __len__(self):
        return len(self.imgs)