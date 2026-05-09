import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models


def select_roi_opencv(
    image: np.ndarray, window_name: str = "Selecciona ROI"
) -> Tuple[int, int, int, int]:
    """
    Seleccion interactiva con OpenCV.
    Devuelve (y0, y1, x0, x1).

    Uso:
    1) Arrastra para marcar ROI.
    2) Enter o Space para confirmar.
    3) c para cancelar.
    """
    img = image.copy()
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    x, y, w, h = cv2.selectROI(window_name, vis, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if w == 0 or h == 0:
        raise ValueError("ROI no seleccionada o cancelada.")

    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w), int(y + h)
    return y0, y1, x0, x1


def get_index(flat_idx, img_shape):
    row, col = np.unravel_index(flat_idx, img_shape)
    return row.flatten(), col.flatten()


def create_conv_filter(img, conv_thresh, coord_filename):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
    _, binary_img = cv2.threshold(gray_img, conv_thresh, 255, cv2.THRESH_BINARY)
    kern = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]
    )

    y0, y1, x0, x1 = load_coords(coord_filename, binary_img)
    filter_coords = y0, y1, x0, x1
    exclamation_mark = binary_img[y0:y1, x0:x1]
    return convolution(exclamation_mark, kern), filter_coords


def create_domino_conv_filters(img, conv_thresh, coord_filename):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
    _, binary_img = cv2.threshold(gray_img, conv_thresh, 255, cv2.THRESH_BINARY)
    kern = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]
    )

    y0, y1, x0, x1 = load_coords(coord_filename, binary_img)
    filter_coords = y0, y1, x0, x1
    exclamation_mark = binary_img[y0:y1, x0:x1]
    return convolution(exclamation_mark, kern), filter_coords


def load_template(filename, img=None):
    if img is not None:
        y0, y1, x0, x1 = select_roi_opencv(img)
        template = img[y0:y1, x0:x1]
        cv2.imwrite(filename, template)
        return template
    else:
        template = cv2.imread(filename)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # type: ignore

        if template is None:
            raise ValueError(f"Could not load {filename}")
        return template



def get_best_filtered_temp(filtered_template, kernels, filename=None):
    scores = np.sum(np.abs(filtered_template), axis=(1, 2))
    best_score_idx = np.argsort(scores)[::-1][:5]
    best_filters = filtered_template[best_score_idx, :, :]
    best_kernels = kernels[best_score_idx, :, :]

    if filename is not None:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i in range(len(best_filters)):
            axes[0, i].imshow(best_filters[i], cmap="gray")
            axes[0, i].set_title(f"Top {i + 1} Activación\n(Filtro VGG: {best_score_idx[i]})")
            axes[0, i].axis("off")
            axes[1, i].imshow(best_kernels[i], cmap="gray")
            axes[1, i].set_title("Kernel")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

    return best_filters, best_kernels

def map_collapse(activation_map):
    saliency_map = np.max(activation_map, axis=0)
    return saliency_map

def get_corners(cx, cy, filter_shape):
    "gets corner coordinates given the center of detections and size of filter"
    yf0, yf1, xf0, xf1 = filter_shape
    width = xf1 - xf0
    height = yf1 - yf0
    x0 = cx - (width // 2)
    y0 = cy - (height // 2)
    x1 = x0 + width
    y1 = y0 + height

    return y0.flatten(), y1.flatten(), x0.flatten(), x1.flatten()


def detect(img, template_coords, threshold=10):
    flattened_img = img.flatten()
    flat_idxs = np.where(flattened_img > threshold)
    row_index, col_index = get_index(flat_idxs, img.shape)
    scores = img[row_index, col_index]
    keep_indices = non_max_suppression(row_index, col_index, scores, template_coords)
    final_row_idxs = row_index[keep_indices]
    final_col_idxs = col_index[keep_indices]

    return get_corners(final_col_idxs, final_row_idxs, template_coords)

def non_max_suppression(row_idxs, col_idxs, scores, filter_shape):
    if len(scores) == 0:
        return []

    y0, y1, x0, x1 = filter_shape
    width = x1 - x0
    height = y1 - y0

    order = np.argsort(scores)[::-1]

    keep_indices = []

    while len(order) > 0:
        current_max_idx = order[0]
        keep_indices.append(current_max_idx)
        if len(order) == 1:
            break
        cx = col_idxs[current_max_idx]
        cy = row_idxs[current_max_idx]
        rest_idxs = order[1:]
        rest_cx = col_idxs[rest_idxs]
        rest_cy = row_idxs[rest_idxs]
        dx = np.abs(rest_cx - cx)
        dy = np.abs(rest_cy - cy)
        outside_neighborhood = (dx >= width) | (dy >= height)
        order = rest_idxs[outside_neighborhood]

    return keep_indices


def get_score(detections):
    y0_dets, y1_dets, x0_dets, x1_dets = detections
    house_score = np.sum(y0_dets < 281)
    p1_score = np.sum((y0_dets > 281) & (x0_dets < 269))
    p2_score = np.sum((y0_dets > 281) & (x0_dets > 269) & (x0_dets < 537))
    p3_score = np.sum((y0_dets > 281) & (x0_dets > 537))

    return p1_score, p2_score, p3_score, house_score


def draw_rectangles(img, detections):
    y0_dets, y1_dets, x0_dets, x1_dets = detections
    bounding_box_img = img
    for i in range(len(y0_dets)):
        x0 = x0_dets[i]
        y0 = y0_dets[i]
        x1 = x1_dets[i]
        y1 = y1_dets[i]

        bounding_box_img = cv2.rectangle(
            bounding_box_img, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=4
        )
    return bounding_box_img


def load_coords(filename, img):
    try:
        y0, y1, x0, x1 = np.load(filename)
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    except FileNotFoundError:
        y0, y1, x0, x1 = select_roi_opencv(img)
        coords = np.array([y0, y1, x0, x1])
        np.save(filename, coords)
    finally:
        return y0, y1, x0, x1  # type: ignore


def convolution(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """calculates the convolution of the image with the given filter.
    Adds zero padding to keep the same dimensions as the original image.
    """
    filter_height, filter_width = filter.shape
    img_height, img_width = img.shape

    padded_img = np.zeros(
        (img_height + filter_height - 1, img_width + filter_width - 1)
    )
    padded_img[
        (filter_height - 1) // 2 : (filter_height - 1) // 2 + img_height,
        (filter_width - 1) // 2 : (filter_width - 1) // 2 + img_width,
    ] = img

    out = np.zeros(img.shape)
    for i in range(img_height):
        for j in range(img_width):
            out[i, j] = np.sum(
                padded_img[i : i + filter_height, j : j + filter_width] * filter
            )
    return out


def get_filters_from_vgg():
    model = models.vgg16(weights="DEFAULT")
    return torch.mean(model.features[0].weight.data, dim=1).cpu().numpy()



def conv_2d(img, kern):
    n_filters = kern.shape[0]
    if img.ndim == 2:
        height, width = img.shape
        filtered_temp = np.zeros((n_filters, height, width))
        for i in range(n_filters):
            filtered_temp[i, :, :] = cv2.filter2D(img, -1, kern[i, :, :])

    elif img.ndim == 3:
        channels, height, width = img.shape
        filtered_temp = np.zeros((n_filters, height, width))
        if channels != n_filters:
            raise ValueError(f"Number of channels ({channels}) must match with number of filters ({n_filters})")
        for i in range(n_filters):
            filtered_temp[i, :, :] = cv2.filter2D(img[i,:,:], -1, kern[i, :, :])
    else:
        raise ValueError(f"Image dimensions not supported: {img.ndim}. Must be 2 or 3.")

    return filtered_temp


if __name__ == "__main__":
    img_name = "grid_symbols.png"
    path = os.path.join("media", img_name)
    img = cv2.imread(path)

    print(select_roi_opencv(img))
