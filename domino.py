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
    create_domino_conv_filters
)

def main():
    img_name = "tablero_domino.png"
    plot_dir = "plots"
    conv_thresh = 200

    coord_filename = "domino.npy"
    domino_path = os.path.join("media", "domino")
    board_img = cv2.imread(os.path.join("media", img_name))

    thresholds = {
        "1.png": 6.2e7,
        "2.png": 5.0e7,
        "3.png": 6e7,
        "4.png": 6.5e7,
        "5.png": 7.0e7,
        "6.png": 4.5e7
    }

    colors = {
        "1.png": (0, 0, 255),   # Rojo
        "2.png": (255, 0, 0),   # Azul
        "3.png": (0, 255, 0),   # Verde
        "4.png": (0, 255, 255), # Amarillo
        "5.png": (255, 0, 255), # Magenta
        "6.png": (255, 255, 0)  # Cian
    }

    final_bb_img = board_img.copy()

    gray_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)  # type: ignore
    _, binary_noisy = cv2.threshold(gray_img, conv_thresh, 255, cv2.THRESH_BINARY)
    img_shape = binary_noisy.shape

    os.makedirs(plot_dir, exist_ok=True)

    for filename in os.listdir(domino_path):
        filepath = os.path.join(domino_path, filename)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            filtered_kernel, filter_coords = create_domino_conv_filters(
                img, conv_thresh, coord_filename
            )
            out = convolution(binary_noisy, filtered_kernel)
            flattened_img = out.flatten()

            current_thresh = thresholds.get(filename, 4.0e7)

            flat_idxs = np.where(flattened_img > current_thresh)
            row_idxs, col_idxs = get_index(flat_idxs, img_shape)
            scores = out[row_idxs, col_idxs]

            keep_indices = non_max_suppression(row_idxs, col_idxs, scores, filter_coords)

            final_row_idxs = row_idxs[keep_indices]
            final_col_idxs = col_idxs[keep_indices]

            # print(f"--- Archivo: {filename} ---")
            # print(f"Umbral fijo utilizado: {current_thresh:.3e}")
            # print(f"Píxeles que superaron el umbral: {len(scores)}")
            # print(f"Detecciones finales tras NMS: {len(final_row_idxs)}\n")

            y0_dets, y1_dets, x0_dets, x1_dets = get_corners(final_col_idxs, final_row_idxs, filter_coords)

            color = colors.get(filename, (255, 255, 255))
            for y0, y1, x0, x1 in zip(y0_dets, y1_dets, x0_dets, x1_dets):
                cv2.rectangle(final_bb_img, (x0, y0), (x1, y1), color, 3)

            # -----------------------------------------
            # 1. Filtered kernel
            # -----------------------------------------
            plt.figure(figsize=(6, 6))
            plt.imshow(filtered_kernel, cmap="gray")
            plt.title("Filtro de Convolución")
            plt.axis("off")
            plt.savefig(os.path.join(plot_dir, f"conv_filter_{filename}.png"), bbox_inches="tight")
            plt.close()

            # -----------------------------------------
            # 2. Heatmap plot
            # -----------------------------------------
            plt.figure(figsize=(8, 6))
            plt.imshow(out, cmap="viridis")
            plt.colorbar()
            plt.title("Matriz de Activación")
            plt.savefig(os.path.join(plot_dir, f"activation_matrix_{filename}.png"), bbox_inches="tight")
            plt.close()

            # -----------------------------------------
            # 3. Histogram plot
            # -----------------------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(flattened_img, label="Activaciones")
            plt.axhline(
                y=current_thresh, label=f"Umbral fijo dictado = {current_thresh:.3e}", c="r"
            )
            plt.xlabel("Índice del pixel")
            plt.ylabel("Valor de activación")
            plt.title(f"Histograma de Convolución - {filename}")
            plt.legend(loc="best")
            plt.grid(True)
            plt.savefig(
                os.path.join(plot_dir, f"histogram_{filename}.png"),
                bbox_inches="tight",
            )
            plt.close()

    # -----------------------------------------
    # 4. Guardar Bounding Box image final
    # -----------------------------------------
    bb_img_rgb = cv2.cvtColor(final_bb_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 12))
    plt.imshow(bb_img_rgb)
    plt.title("Distribución de Fichas (Cajas Coloreadas)")
    plt.axis("off")
    plt.savefig(
        os.path.join(plot_dir, "tablero_final_detectado.png"),
        bbox_inches="tight",
    )
    plt.close()

    print("¡Detección completada! Revisa la carpeta 'plots' para ver 'tablero_final_detectado.png'")

if __name__ == "__main__":
    main()
