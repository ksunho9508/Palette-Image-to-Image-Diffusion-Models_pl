import albumentations as A
from albumentations.pytorch import transforms

# from .aug_degradation import DE_process

#
target_list = [
    "Hemorrhage",
    "HardExudate",
    "CWP",
    "Drusen",
    "VascularAbnormality",
    "Membrane",
    "ChroioretinalAtrophy",
    "MyelinatedNerveFiber",
    "RNFLDefect",
    "GlaucomatousDiscChange",
    "NonGlaucomatousDiscChange",
    "MacularHole",
]

mask_targets = {f"mask_{t}": "mask" for t in target_list}
land_marks = {"optic_disc": "keypoints", "fovea": "keypoints"}


def aug_train(img_size):
    aug_train = A.Compose(
        [
            A.Resize(
                img_size,
                img_size,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                rotate_limit=180, border_mode=0, value=0, mask_value=0, p=0.9
            ),
            A.RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=10, p=0.3),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        brightness_by_max=True,
                        p=1,
                    ),
                    A.RandomGamma(gamma_limit=(50, 150), p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=10, p=1),
                    A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
                    A.Downscale(scale_max=0.5, p=1),
                ],
                p=0.3,
            ),
            A.Normalize(mean=0, std=1),
            transforms.ToTensorV2(),
        ],
        # keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"]),
        additional_targets={**mask_targets},  # , **land_marks},
    )
    return aug_train


def aug_val(img_size):
    aug_val = A.Compose(
        [
            A.Resize(
                img_size,
                img_size,
            ),
            A.Normalize(mean=0, std=1),
            transforms.ToTensorV2(),
        ],
        # keypoint_params=A.KeypointParams(format="xy", label_fields=["class_labels"]),
        additional_targets={**mask_targets},  # , **land_marks},
    )
    return aug_val
 