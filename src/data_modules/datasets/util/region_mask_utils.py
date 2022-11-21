import numpy as np
import os


def generate_single_region_mask(img_size, disc_center, macula_center, region):
    # pt format : (y,x)
    mask_val = 1
    if disc_center[1] < macula_center[1]:
        laterality = "L"
    else:
        laterality = "R"
    xor_val = False if laterality == "L" else True
    mask = np.zeros(img_size)
    d = dist(disc_center, macula_center)
    r_disc = 2.0 * d / 5
    r_macula = 2.0 * d / 3
    main_slope = slope(disc_center, macula_center)  # assume that main_slope is not inf
    normal_slope = -1.0 / main_slope if main_slope != 0 else 0
    x, y = np.ogrid[: mask.shape[0], : mask.shape[1]]

    # sup is above inf in the image
    inter_sup, inter_inf = intersection(disc_center, macula_center, r_disc, r_macula)
    macula_sup, macula_inf = circle_line_intersection(
        macula_center, r_macula, normal_slope
    )
    disc_sup, disc_inf = circle_line_intersection(disc_center, r_disc, normal_slope)
    macula_sup_ext = pt_on_the_slope(macula_sup, main_slope, laterality)
    macula_inf_ext = pt_on_the_slope(macula_inf, main_slope, laterality)

    if region == "M":  # Macular Area
        mask[
            ~out_circle((x, y), macula_center, r_macula)
            & (ccw((x, y), inter_sup, inter_inf) ^ xor_val)
        ] = mask_val
    elif region == "ID":  # Inferior Disc Area
        mask[
            ~out_circle((x, y), disc_center, r_disc)
            & (~ccw((x, y), inter_sup, inter_inf) ^ xor_val)
            & (~ccw((x, y), disc_center, macula_center) ^ xor_val)
        ] = mask_val
    elif region == "SD":  # Superior Disc Area
        mask[
            ~out_circle((x, y), disc_center, r_disc)
            & (~ccw((x, y), inter_sup, inter_inf) ^ xor_val)
            & (ccw((x, y), disc_center, macula_center) ^ xor_val)
        ] = mask_val
    elif region == "IT":  # Inferotemporal Area
        mask[
            (ccw((x, y), disc_sup, disc_inf) ^ xor_val)
            & (
                (~ccw((x, y), macula_inf, macula_inf_ext) ^ xor_val)
                | (
                    out_circle((x, y), macula_center, r_macula)
                    & out_circle((x, y), disc_center, r_disc)
                    & (~ccw((x, y), macula_sup, macula_inf) ^ xor_val)
                    & (ccw((x, y), macula_inf, macula_inf_ext) ^ xor_val)
                    & (~ccw((x, y), disc_center, macula_center) ^ xor_val)
                )
            )
        ] = mask_val
    elif region == "ST":  # Msuperotemporal Area
        mask[
            (ccw((x, y), disc_sup, disc_inf) ^ xor_val)
            & (
                (ccw((x, y), macula_sup, macula_sup_ext) ^ xor_val)
                | (
                    out_circle((x, y), macula_center, r_macula)
                    & out_circle((x, y), disc_center, r_disc)
                    & (~ccw((x, y), macula_sup, macula_inf) ^ xor_val)
                    & (~ccw((x, y), macula_sup, macula_sup_ext) ^ xor_val)
                    & (ccw((x, y), disc_center, macula_center) ^ xor_val)
                )
            )
        ] = mask_val
    elif region == "T":  # Temporal Area
        mask[
            (ccw((x, y), disc_sup, disc_inf) ^ xor_val)
            & (
                out_circle((x, y), macula_center, r_macula)
                & (ccw((x, y), macula_sup, macula_inf) ^ xor_val)
                & (~ccw((x, y), macula_sup, macula_sup_ext) ^ xor_val)
                & (ccw((x, y), macula_inf, macula_inf_ext) ^ xor_val)
            )
        ] = mask_val
    elif region == "IN":  # Inferonasal Area
        mask[
            out_circle((x, y), disc_center, r_disc)
            & (~ccw((x, y), disc_center, macula_center) ^ xor_val)
            & (~ccw((x, y), disc_sup, disc_inf) ^ xor_val)
        ] = mask_val
    elif region == "SN":  # Superonasal Area
        mask[
            out_circle((x, y), disc_center, r_disc)
            & (ccw((x, y), disc_center, macula_center) ^ xor_val)
            & (~ccw((x, y), disc_sup, disc_inf) ^ xor_val)
        ] = mask_val

    return mask


def out_circle(pt, c_pt, r):
    return dist(pt, c_pt) > r


def ccw(first_pt, second_pt, third_pt):
    # return if the path of first_pt->second_pt->third_pt bends in counter-clock-wise direction (following image coordinate)
    x1, y1, x2, y2, x3, y3 = (
        first_pt[1],
        first_pt[0],
        second_pt[1],
        second_pt[0],
        third_pt[1],
        third_pt[0],
    )
    return x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1 < 0


def intersection(pt1, pt2, r1, r2):
    a, b, c, d = pt1[1], pt1[0], pt2[1], pt2[0]
    r, s = r1, r2
    # code pasted from the web
    e = c - a
    f = d - b
    p = np.sqrt(e * e + f * f)
    k = (p * p + r * r - s * s) / (2 * p)
    x1 = a + (e * k) / p + (f / p) * np.sqrt(r * r - k * k)
    y1 = b + (f * k) / p - (e / p) * np.sqrt(r * r - k * k)
    x2 = a + (e * k) / p - (f / p) * np.sqrt(r * r - k * k)
    y2 = b + (f * k) / p + (e / p) * np.sqrt(r * r - k * k)

    if y1 < y2:
        return (y1, x1), (y2, x2)
    else:
        return (y2, x2), (y1, x1)


def circle_line_intersection(c_pt, r, slope):
    if slope != 0:
        if slope < 0:
            return (
                c_pt[0] + 1.0 * r / np.sqrt(1 + slope**2) * slope,
                c_pt[1] + 1.0 * r / np.sqrt(1 + slope**2),
            ), (
                c_pt[0] - 1.0 * r / np.sqrt(1 + slope**2) * slope,
                c_pt[1] - 1.0 * r / np.sqrt(1 + slope**2),
            )
        else:
            return (
                c_pt[0] - 1.0 * r / np.sqrt(1 + slope**2) * slope,
                c_pt[1] - 1.0 * r / np.sqrt(1 + slope**2),
            ), (
                c_pt[0] + 1.0 * r / np.sqrt(1 + slope**2) * slope,
                c_pt[1] + 1.0 * r / np.sqrt(1 + slope**2),
            )
    else:
        return (c_pt[0] - r, c_pt[1]), (c_pt[0] + r, c_pt[1])


def pt_on_the_slope(pt, slope, laterality):
    # point away from the optic disc
    if laterality == "L":
        return (pt[0] + slope * 100, pt[1] + 100)
    elif laterality == "R":
        return (pt[0] + slope * (-100), pt[1] - 100)
    else:
        return None


def dist(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def slope(pt1, pt2):
    if pt2[1] != pt1[1]:  # prevent infinite slope
        return 1.0 * (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
    else:
        return 1.0 * (pt2[0] - pt1[0])


def get_all_region_masks(img_size, disc_center, macula_center, regions):
    region_codes = ["None", "M", "T", "ST", "IT", "SN", "IN", "SD", "ID"]
    n_findings, n_region_codes = regions.shape
    mask = np.zeros((n_findings,) + img_size)
    for finding_index in range(n_findings):
        for region_code_index in range(1, n_region_codes):
            if regions[finding_index, region_code_index] == 1:
                mask[finding_index][
                    (mask[finding_index] == 1)
                    | (
                        generate_single_region_mask(
                            img_size,
                            disc_center,
                            macula_center,
                            region_codes[region_code_index],
                        )
                        == 1
                    )
                ] = 1
    return mask


def encoding_region(region_str):
    region_code = {
        "": 0,
        "M": 1,
        "T": 2,
        "ST": 3,
        "IT": 4,
        "SN": 5,
        "IN": 6,
        "SD": 7,
        "ID": 8,
    }
    n_regions = len(region_code) - 1
    codified_regions = np.zeros(n_regions + 1)
    for region in [chunk.strip() for chunk in region_str.split("|")]:
        codified_regions[region_code[region]] = 1
    return codified_regions


# def load_augmented_aux_loss(fname, lms, regions, normalize, augment, mask_shape):
#     # read image file
#     ori_img, img_aug, lms_aug = load_imgs_lms([fname], [lms], normalize, augment)
#     ori_img = ori_img[0, ...]
#     img_aug = img_aug[0, ...]
#     lms_aug = lms_aug[0, ...]
#     ratio = 1.0 * np.array(mask_shape) / np.array(ori_img.shape[:2])
#     region_mask = get_mask_of_regions_aux_loss(
#         mask_shape,
#         (lms_aug[0] * ratio[0], lms_aug[1] * ratio[1]),
#         (lms_aug[2] * ratio[0], lms_aug[3] * ratio[1]),
#         regions,
#     )
#     for finding_index in range(regions.shape[0]):
#         region_mask[
#             finding_index,
#             ori_img[:: int(1.0 / ratio[0]), :: int(1.0 / ratio[1]), 0] < 10,
#         ] = 0
#     assert (
#         len(img_aug.shape) == 3
#         and lms_aug.shape == (4,)
#         and len(region_mask.shape) == 3
#     )
#     return img_aug, lms_aug, region_mask

from PIL import Image


def plot_imgs(imgs, out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(imgs.shape[0]):
        Image.fromarray((imgs[i, ...]).astype(np.uint8)).save(
            os.path.join(out_dir, "imgs_{}.png".format(i + 1))
        )
        # Image.fromarray(imgs[i, ::32, ::32].astype(np.uint8)).save(
        #     os.path.join(out_dir, "imgs_discrete_{}.png".format(i + 1))
        # )
        # Image.fromarray(resized_img.astype(np.uint8)).save(
        #     os.path.join(out_dir, "imgs_resized_{}.png".format(i + 1))
        # )
