import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchstain
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, closing
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import cellpose

def plot_histogram_rgb(img, filepath):
    "IMG must be on rgb colorspace"

    colors = ("r", "g", "b")

    plt.figure()
    plt.title("Distribución de Intensidades RGB")
    plt.xlabel("Intensidad de Píxel")
    plt.ylabel("Frecuencia")

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)

    plt.xlim((0, 256))

    plt.savefig(filepath)
    plt.close()


def plot_img(img, filepath, title=''):
    "IMG must be on rgb colorspace"
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def compute_lab_statistics(img):
    """calculates and returns the mean and std deviation per img channel in LAB space
    img must be on RGB colorspace
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return np.mean(lab_img, axis=(0, 1)), np.std(lab_img, axis=(0, 1))


def equal_hist_l(img):
    """img must be on RGB colorspace
    returns HE img on RGB colorspace
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab_img[:, :, 0]

    hist, _ = np.histogram(l_channel.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf / float(cdf.max())
    lut = np.floor(255 * cdf_normalized)
    lab_img[:, :, 0] = lut[l_channel]
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB).astype(np.uint8)


def clahe_l(img):
    "img must be on RGB colorspace"
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = CLAHE.apply(lab_img[:, :, 0])
    new_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB).astype(np.uint8)
    return new_img


def reinhard_normalization(source, target):
    """source and target must be on RGB colorspace!!"""
    source_mean, source_std = compute_lab_statistics(source)
    target_mean, target_std = compute_lab_statistics(target)

    lab_source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)

    out = np.zeros_like(lab_source)

    for i in range(3):
        out[:, :, i] = (
            target_std[i] * (lab_source[:, :, i] - source_mean[i]) / source_std[i]
        ) + target_mean[i]

    out = np.clip(out, 0, 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def get_H_and_E_channels(img):
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    normalizer.fit(img)
    _, H, E = normalizer.normalize(img, stains=True)
    H_uint8 = np.clip(H, 0, 255).astype(np.uint8)  # type:ignore
    E_uint8 = np.clip(E, 0, 255).astype(np.uint8)  # type:ignore

    return H_uint8, E_uint8

def classic_segmentation(h_channel):
    if len(h_channel.shape) == 3:
        h_gray = cv2.cvtColor(h_channel, cv2.COLOR_RGB2GRAY)
    else:
        h_gray = h_channel
    thresh = threshold_otsu(h_gray)
    binary = h_gray < thresh

    kernel = disk(3)
    cleaned = closing(binary, kernel)
    cleaned = opening(cleaned, kernel)

    distance = ndi.distance_transform_edt(cleaned)

    local_max_coords = peak_local_max(distance, min_distance=7, labels=cleaned)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    markers, _ = ndi.label(local_max_mask)

    labels = watershed(-distance, markers, mask=cleaned)

    return labels

def deep_segmentation(img, model):
    masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])

    return masks

def prediction_resize(pred_mask, ground_truth_mask):
    return cv2.resize(pred_mask.astype(np.uint8), (ground_truth_mask.shape[1], ground_truth_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

def compute_metrics(pred_mask, gt_mask):

    pred_bin = pred_mask > 0
    gt_bin = gt_mask > 0
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    dice = (2. * intersection) / (pred_bin.sum() + gt_bin.sum())
    iou = intersection / (union)

    return dice, iou

if __name__ == "__main__":
    path = os.path.join("data", "images", "TCGA-A7-A13E-01Z-00-DX1.tif")
    os.makedirs("out", exist_ok=True)
    out = os.path.join("out", "histogram.pdf")
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # type:ignore
    print(compute_lab_statistics(img))
    plot_histogram_rgb(img, out)
