import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    convolution,
    create_conv_filter,
    draw_rectangles,
    get_corners,
    get_index,
    non_max_suppression,
)


def main():
    img_name = "grid_symbols.png"
    plot_dir = "plots"
    conv_thresh = 200
    detection_thresh = 0.7e8
    noisy_name = "grid_symbols_blurred.png"
    coord_filename = "coords_basic.npy"

    path = os.path.join("media", img_name)
    noisy_path = os.path.join("media", noisy_name)
    img = cv2.imread(path)
    noisy_img = cv2.imread(noisy_path)
    gray_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)  # type: ignore
    _, binary_noisy = cv2.threshold(gray_noisy, conv_thresh, 255, cv2.THRESH_BINARY)

    filtered_kernel, filter_coords = create_conv_filter(
        img, conv_thresh, coord_filename
    )
    img_shape = binary_noisy.shape
    out = convolution(binary_noisy, filtered_kernel)

    flattened_img = out.flatten()
    flat_idxs = np.where(flattened_img > detection_thresh)
    row_idxs, col_idxs = get_index(flat_idxs, img_shape)
    print(f"{row_idxs = }, {col_idxs = }")
    scores = out[row_idxs, col_idxs]

    keep_indices = non_max_suppression(row_idxs, col_idxs, scores, filter_coords)

    final_row_idxs = row_idxs[keep_indices]
    final_col_idxs = col_idxs[keep_indices]

    print(f"Píxeles detectados originalmente: {len(scores)}")
    print(f"Detecciones finales tras NMS: {len(keep_indices)}")

    y0_dets, y1_dets, x0_dets, x1_dets = get_corners(
        final_col_idxs, final_row_idxs, filter_coords
    )
    detections = (y0_dets, y1_dets, x0_dets, x1_dets)

    # -----------------------------------------
    # 1. Filtered kernel
    # -----------------------------------------
    plt.figure(figsize=(6, 6))
    # Asumimos que el kernel es de un solo canal (escala de grises)
    plt.imshow(filtered_kernel, cmap="gray")
    plt.title("Filtro de Convolución")
    plt.axis("off")  # Oculta los ejes numéricos para que parezca una imagen pura
    plt.savefig(
        os.path.join(plot_dir, "conv_filter.pdf"), format="pdf", bbox_inches="tight"
    )
    plt.close()  # Cierra la figura para liberar memoria

    # -----------------------------------------
    # 2. Heatmap plot
    # -----------------------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(out, cmap="viridis")
    plt.colorbar()
    plt.title("Matriz de Activación")
    plt.savefig(
        os.path.join(plot_dir, f"activation_matrix_{noisy_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------------------
    # 3. Histogram plot
    # -----------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(flattened_img, label="Activaciones")
    plt.axhline(
        y=detection_thresh, label=f"Umbral de detección = {detection_thresh:.2e}", c="r"
    )
    plt.xlabel("Índice del pixel")
    plt.ylabel("Valor de activación")
    plt.title("Valor de los píxeles de la convolución")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(
        os.path.join(
            plot_dir, f"histogram_thresh_{detection_thresh:.2e}_{noisy_name}.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------------------
    # 4. Bounding box image
    # -----------------------------------------
    bb_img = draw_rectangles(noisy_img, detections)

    bb_img_rgb = cv2.cvtColor(bb_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(bb_img_rgb)
    plt.title("Cajas de Detección")
    plt.axis("off")  # Ocultamos los ejes
    plt.savefig(
        os.path.join(
            plot_dir, f"bounding_boxes_thresh_{detection_thresh:.2e}_{noisy_name}.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
