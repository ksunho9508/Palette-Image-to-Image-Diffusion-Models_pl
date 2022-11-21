import random
import math
import torchvision.transforms.functional as F
import numpy as np
import cv2
import json
from PIL import Image

from skimage import measure


def center_mask(img, red_threshold=20):
    # img = imgs[0] # imgs shape : (1, h, w, 3)
    h, w, _ = np.shape(img)
    roi_check_len = h // 5

    # Find Connected Components with intensity above the threshold
    blobs_labels, n_blobs = measure.label(
        img[:, :, 0] > red_threshold, connectivity=1, return_num=True
    )

    # Find the Index of Connected Components of the Fundus Area (the central area)
    majority_vote = np.argmax(
        np.bincount(
            blobs_labels[
                h // 2 - roi_check_len // 2 : h // 2 + roi_check_len // 2,
                w // 2 - roi_check_len // 2 : w // 2 + roi_check_len // 2,
            ].flatten()
        )
    )
    mask = blobs_labels == majority_vote
    return mask  # np.expand_dims(mask, axis=0) # shape : (1, h, w)


import torchvision.transforms as transforms


def get_transform(resize_or_crop, loadSizeX, loadSizeY, fineSize):
    transform_list = []
    if resize_or_crop == "resize_and_crop":
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == "crop":
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == "scale":
        osize = [loadSizeX, loadSizeY]

    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)


# Must be edit this part to right size for each task!
# transform = get_transform('scale',loadSizeX =512, loadSizeY=512, fineSize=512)
transform = get_transform("scale", loadSizeX=224, loadSizeY=224, fineSize=224)


def DE_COLOR(img, brightness=0.3, contrast=0.4, saturation=0.4):
    """Randomly change the brightness, contrast and saturation of an image"""

    if brightness > 0:
        brightness_factor = random.uniform(
            max(0.0, 1.0 - brightness), 1.0 + brightness - 0.1
        )  # brightness factor
        img = F.adjust_brightness(img, brightness_factor)
    if contrast > 0:
        contrast_factor = random.uniform(
            max(0.0, 1.0 - contrast), 1.0 + contrast
        )  # contrast factor
        img = F.adjust_contrast(img, contrast_factor)
    if saturation > 0:
        saturation_factor = random.uniform(
            max(0.0, 1.0 - saturation), 1.0 + saturation
        )  # saturation factor
        img = F.adjust_saturation(img, saturation_factor)

    img = transform(img)
    img = img.numpy()

    color_params = {}
    color_params["brightness_factor"] = brightness_factor
    color_params["contrast_factor"] = contrast_factor
    color_params["saturation_factor"] = saturation_factor

    return img, color_params


def DE_HALO(img, h, w, brightness_factor, center=None, radius=None):
    """
    Defined to simulate a 'ringlike' halo noise in fundus image
    :param weight_r/weight_g/weight_b: Designed to simulate 3 kinds of halo noise shown in Kaggle dataset.
    :param center_a/center_b:          Position of each circle which is simulated the ringlike shape
    :param dia_a/dia_b:                Size of each circle which is simulated the ringlike noise
    :param weight_hal0:                Weight of added halo noise color
    :param sigma:                      Filter size for final Gaussian filter
    """

    weight_r = [251 / 255, 141 / 255, 177 / 255]
    weight_g = [249 / 255, 238 / 255, 195 / 255]
    weight_b = [246 / 255, 238 / 255, 147 / 255]
    # num
    if brightness_factor >= 0.2:
        num = random.randint(1, 2)
    else:
        num = random.randint(0, 2)
    w0_a = random.randint(w / 2 - int(w / 8), w / 2 + int(w / 8))
    h0_a = random.randint(h / 2 - int(h / 8), h / 2 + int(h / 8))
    center_a = [w0_a, h0_a]

    wei_dia_a = 0.75 + (1.0 - 0.75) * random.random()
    dia_a = min(h, w) * wei_dia_a
    Y_a, X_a = np.ogrid[:h, :w]
    dist_from_center_a = np.sqrt((X_a - center_a[0]) ** 2 + (Y_a - center_a[1]) ** 2)
    circle_a = dist_from_center_a <= (int(dia_a / 2))

    mask_a = np.zeros((h, w))
    mask_a[circle_a] = np.mean(img)  # np.multiply(A[0], (1 - t))

    center_b = center_a
    Y_b, X_b = np.ogrid[:h, :w]
    dist_from_center_b = np.sqrt((X_b - center_b[0]) ** 2 + (Y_b - center_b[1]) ** 2)

    dia_b_max = (
        2
        * int(
            np.sqrt(
                max(center_a[0], h - center_a[0]) * max(center_a[0], h - center_a[0])
                + max(center_a[1], h - center_a[1]) * max(center_a[1], w - center_a[1])
            )
        )
        / min(w, h)
    )
    wei_dia_b = 1.0 + (dia_b_max - 1.0) * random.random()

    if num == 0:
        # if halo tend to be a white one, set the circle with a larger radius.
        dia_b = min(h, w) * wei_dia_b + abs(
            max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) * 2 / 3
        )
    else:
        dia_b = min(h, w) * wei_dia_b + abs(
            max(center_b[0] - w / 2, center_b[1] - h / 2) + max(w, h) / 2
        )

    circle_b = dist_from_center_b <= (int(dia_b / 2))

    mask_b = np.zeros((h, w))
    mask_b[circle_b] = np.mean(img)

    weight_hal0 = [0, 1, 1.5, 2, 2.5]
    delta_circle = np.abs(mask_a - mask_b) * weight_hal0[1]
    dia = max(center_a[0], h - center_a[0], center_a[1], h - center_a[1]) * 2
    gauss_rad = int(np.abs(dia - dia_a))
    sigma = 2 / 3 * gauss_rad

    if (gauss_rad % 2) == 0:
        gauss_rad = gauss_rad + 1
    delta_circle = cv2.GaussianBlur(delta_circle, (gauss_rad, gauss_rad), sigma)

    delta_circle = np.array(
        [
            weight_r[num] * delta_circle,
            weight_g[num] * delta_circle,
            weight_b[num] * delta_circle,
        ]
    )
    img = img + delta_circle

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img, delta_circle


def DE_HOLE(img, h, w, region_mask, center=None, diameter=None):
    """

    :param diameter_circle:     The size of the simulated artifacts caused by non-uniform lighting
    :param center:              Position
    :param brightness_factor:   Weight utilized to adapt the value of generated non-uniform lighting artifacts.
    :param sigma:               Filter size for final Gaussian filter

    :return:
    """
    # if radius is None: # use the smallest distance between the center and image walls
    # diameter_circle = random.randint(int(0.3*w), int(0.5 * w))
    #  define the center based on the position of disc/cup
    diameter_circle = random.randint(int(0.4 * w), int(0.7 * w))

    center = [random.randint(w / 4, w * 3 / 4), random.randint(h * 3 / 8, h * 5 / 8)]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    circle = dist_from_center <= (int(diameter_circle / 2))

    mask = np.zeros((h, w))
    mask[circle] = 1

    num_valid = np.sum(region_mask)
    aver_color = np.sum(img) / (3 * num_valid)
    if aver_color > 0.25:
        brightness = random.uniform(-0.26, -0.262)
        brightness_factor = random.uniform(
            (brightness - 0.06 * aver_color), brightness - 0.05 * aver_color
        )
    else:
        brightness = 0
        brightness_factor = 0
    # print( (aver_color,brightness,brightness_factor))
    mask = mask * brightness_factor

    rad_w = random.randint(int(diameter_circle * 0.55), int(diameter_circle * 0.75))
    rad_h = random.randint(int(diameter_circle * 0.55), int(diameter_circle * 0.75))
    sigma = 2 / 3 * max(rad_h, rad_w) * 1.2

    if (rad_w % 2) == 0:
        rad_w = rad_w + 1
    if (rad_h % 2) == 0:
        rad_h = rad_h + 1

    mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
    mask = np.array([mask, mask, mask])
    img = img + mask
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img, mask


def DE_ILLUMINATION(img, region_mask, h=224, w=224, illumination_mask=None):

    img, color_params = DE_COLOR(img)  # (0.0, 1.0, (3, 128, 128))
    img, halo_mask = DE_HALO(img, h, w, color_params["brightness_factor"])  # [0 ~ 1)
    img, hole_mask = DE_HOLE(img, h, w, region_mask)  # [-0.3 ~ 0.3]

    return img, halo_mask.mean(axis=0), hole_mask.mean(axis=0)


def DE_SPOT(img, h, w, center=None, radius=None):
    """
    :param s_num:  The number of the generated artifacts spot on the fundus image
    :param radius: Define the size of each spot
    :param center: Position of each spot on the fundus image
    :param K:      Weight of original fundus image value
    :param beta:   Weight of generated artifacts(spots) mask value (The color is adapted based on the size(radius) of each spot)
    :param sigma:  Filter size for final Gaussian filter

    """
    spot_params = []
    s_num = random.randint(5, 10)
    mask0 = np.zeros((h, w))
    for i in range(s_num):
        # if radius is None: # use the smallest distance between the center and image walls
        # radius = min(center[0], center[1], w-center[0], h-center[1])
        radius = random.randint(math.ceil(0.01 * h), int(0.05 * h))

        # if center is None: # in the middle of the image
        center = [
            random.randint(radius + 1, w - radius - 1),
            random.randint(radius + 1, h - radius - 1),
        ]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        circle = dist_from_center <= (int(radius / 2))

        k = (14 / 25) + (1.0 - radius / 25)
        beta = 0.5 + (1.5 - 0.5) * (radius / 25)
        A = k * np.ones((3, 1))
        d = 0.3 * (radius / 25)
        t = math.exp(-beta * d)

        mask = np.zeros((h, w))
        mask[circle] = np.multiply(A[0], (1 - t))
        mask0 = mask0 + mask
        # mask0[mask0 != 0] = 1

        sigma = (5 + (20 - 0) * (radius / 25)) * 2
        rad_w = random.randint(int(sigma / 5), int(sigma / 4))
        rad_h = random.randint(int(sigma / 5), int(sigma / 4))
        if (rad_w % 2) == 0:
            rad_w = rad_w + 1
        if (rad_h % 2) == 0:
            rad_h = rad_h + 1

        mask = cv2.GaussianBlur(mask, (rad_w, rad_h), sigma)
        mask = np.array([mask, mask, mask])
        img = img + mask
        img = np.maximum(img, 0)
        img = np.minimum(img, 1)

    return img, mask0


def DE_BLUR(
    img,
):
    """

    :param sigma: Filter size for Gaussian filter

    """
    img = np.transpose(img, (1, 2, 0))
    sigma = 5 + (15 - 5) * random.random()

    img = cv2.GaussianBlur(img, (5, 5), sigma)
    img = np.transpose(img, (2, 0, 1))

    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    return img, sigma


from albumentations.core.transforms_interface import ImageOnlyTransform


# class WithDegradationInfo(BasicTransform):
#     """Transform applied to image only."""

#     @property
#     def targets(self):
#         return {
#             "image": self.apply,
#         }


class DE_process(ImageOnlyTransform):
    def __init__(self, prob_illumination=.5, prob_spot=.5, prob_blur=.5):  # , de_type="001", always_apply=False, p=1):
        super(DE_process, self).__init__(always_apply=True)
        self.de_prob = {
            "illumination": prob_illumination,
            "spot": prob_spot,
            "blur": prob_blur,
        }

    def apply(self, img, **params):
        mask = center_mask(img)
        h, w = mask.shape[0], mask.shape[1]
        img = Image.fromarray(img.astype(np.uint8))

        if random.random() < self.de_prob["illumination"]:
            img, halo_mask, hole_mask = DE_ILLUMINATION(img, mask, h, w)
        else:
            img = transform(img)
            img = img.numpy()
            halo_mask, hole_mask = np.zeros((h, w)), np.zeros((h, w))

        if random.random() < self.de_prob["spot"]:
            img, spot_mask = DE_SPOT(img, h, w)
        else:
            spot_mask = np.zeros((h, w))

        if random.random() < self.de_prob["blur"]:
            img, blurness = DE_BLUR(img)
        else:
            blurness = np.zeros((h, w))

        img = (np.transpose(img * mask, (1, 2, 0)) * 255).astype(np.uint8)
        self.degradation_info = {
            "blurness": blurness,  # [5, 15]
            "spot_mask": spot_mask,  # [0, 0.x],
            "halo_mask": halo_mask,  # [0, 0.3]
            "hole_mask": -hole_mask,  # [-0.3, 0],
        }
        return img


"""
from de_aug.degrad_process import DE_process
def train_aug_rgb_degrad():
    return Compose([
            OneOf([
                DE_process(de_type='001',p=1),#0.8),
                DE_process(de_type='010',p=1),#0.8),
                DE_process(de_type='100',p=1),#0.8),
                DE_process(de_type='011',p=1),#0.8),
                DE_process(de_type='101',p=1),#0.8),
                DE_process(de_type='110',p=1),#0.8),
                DE_process(de_type='111',p=1),#0.8),
            ],p=0.8),
            
            OneOf([
                RGBShift(r_shift_limit=20, g_shift_limit=10, b_shift_limit=10),
                RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2),
                RandomGamma(),
                ColorJitter(),
                HueSaturationValue()
            ], p=0.5),
            
            OneOf([
                GaussianBlur(),
                GaussNoise(var_limit=(0.01)),
                Sharpen(),
            ], p=0.2),

            OneOf([
                Downscale(), 
                Emboss(),
                ],p=0.2),
            
            OneOf([
                ISONoise(),
                RandomShadow(),
            ],p=0.2),

            OneOf([
                Posterize(),
                RandomToneCurve(),
            ],p=0.2),
            
            CLAHE(p=0.3),
            Equalize(p=0.1),
        ])  
    """
