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
    get_score,
    non_max_suppression,
)


def main():
    img_name = "blackjack.png"
    plot_dir = os.path.join("plots", "blackjack")
    os.makedirs(plot_dir, exist_ok=True)
    conv_thresh = 200
    detection_thresh = 1.155e7
    coord_filename = os.path.join("templates","coords_blackjack.npy")

    path = os.path.join("media", img_name)
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
    _, binary_noisy = cv2.threshold(gray_img, conv_thresh, 255, cv2.THRESH_BINARY)

    filtered_kernel, filter_coords = create_conv_filter(
        img, conv_thresh, coord_filename
    )
    img_shape = binary_noisy.shape
    out = convolution(binary_noisy, filtered_kernel)

    flattened_img = out.flatten()
    flat_idxs = np.where(flattened_img > detection_thresh)
    row_idxs, col_idxs = get_index(flat_idxs, img_shape)
    scores = out[row_idxs, col_idxs]

    keep_indices = non_max_suppression(row_idxs, col_idxs, scores, filter_coords)

    final_row_idxs = row_idxs[keep_indices]
    final_col_idxs = col_idxs[keep_indices]

    print(f"Píxeles detectados originalmente: {len(scores)}")
    print(f"Detecciones finales tras NMS: {len(keep_indices)}")

    y0_dets, y1_dets, x0_dets, x1_dets = get_corners(col_idxs, row_idxs, filter_coords)
    detections = (y0_dets, y1_dets, x0_dets, x1_dets)
    p1_score, p2_score, p3_score, house_score = get_score(detections)

    print("RESULTADOS (Sin NMS)")
    print(f"Mano Jugador 1: {p1_score}")
    print(f"Mano Jugador 2: {p2_score}")
    print(f"Mano Jugador 3: {p3_score}")
    print(f"Mano de la casa: {house_score}")

    y0_dets, y1_dets, x0_dets, x1_dets = get_corners(
        final_col_idxs, final_row_idxs, filter_coords
    )
    detections = (y0_dets, y1_dets, x0_dets, x1_dets)

    p1_score, p2_score, p3_score, house_score = get_score(detections)

    print("RESULTADOS (NMS)")
    print(f"Mano Jugador 1: {p1_score}")
    print(f"Mano Jugador 2: {p2_score}")
    print(f"Mano Jugador 3: {p3_score}")
    print(f"Mano de la casa: {house_score}")

    # -----------------------------------------
    # 1. Filtered kernel
    # -----------------------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(filtered_kernel, cmap="gray")
    plt.title("Filtro de Convolución")
    plt.axis("off")
    plt.savefig(
        os.path.join(plot_dir, "conv_filter.pdf"), format="pdf", bbox_inches="tight"
    )
    plt.close()

    # -----------------------------------------
    # 2. Heatmap plot
    # -----------------------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(out, cmap="viridis")
    plt.colorbar()
    plt.title("Matriz de Activación")
    plt.savefig(
        os.path.join(plot_dir, f"activation_matrix_{img_name}.pdf"),
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
        y=detection_thresh, label=f"Umbral de detección = {detection_thresh:.3e}", c="r"
    )
    plt.xlabel("Índice del pixel")
    plt.ylabel("Valor de activación")
    plt.title("Valor de los píxeles de la convolución")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(
        os.path.join(
            plot_dir, f"histogram_thresh_{detection_thresh:.3e}_{img_name}.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    # -----------------------------------------
    # 4. Bounding box image
    # -----------------------------------------
    bb_img = draw_rectangles(img, detections)
    print(bb_img.shape)
    bb_img_rgb = cv2.cvtColor(bb_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(bb_img_rgb)
    plt.axhline(y=281, color="r")
    plt.vlines(x=269, ymin=281, ymax=562, color="r")
    plt.vlines(x=537, ymin=281, ymax=562, color="r")
    plt.title("Cajas de Detección")
    plt.axis("off")  # Ocultamos los ejes
    plt.savefig(
        os.path.join(
            plot_dir, f"bounding_boxes_thresh_{detection_thresh:.3e}_{img_name}.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
