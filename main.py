import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import convolution, draw_rectangles, get_corners, get_index, load_coords


def main():
    img_name = "grid_symbols.png"
    plot_dir = "plots"
    conv_thresh = 200
    detection_thresh = 1.25e8

    path = os.path.join("media", img_name)
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
    img_shape = gray_img.shape
    _, binary_img = cv2.threshold(gray_img, conv_thresh, 255, cv2.THRESH_BINARY)
    kern = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]
    )

    y0, y1, x0, x1 = load_coords("coords.npy", binary_img)
    filter_shape = (y0, y1, x0, x1)
    print(f"{y0= }, {y1= }, {x0= }, {x1= }")
    exclamation_mark = binary_img[y0:y1, x0:x1]
    filtered_kernel = convolution(exclamation_mark, kern)
    out = convolution(binary_img, filtered_kernel)

    flattened_img = out.flatten()
    flat_idxs = np.where(flattened_img > detection_thresh)
    row_idxs, col_idxs = get_index(flat_idxs, img_shape)
    print(f"{row_idxs = }, {col_idxs = }")

    y0_dets, y1_dets, x0_dets, x1_dets = get_corners(col_idxs, row_idxs, filter_shape)
    detections = (y0_dets, y1_dets, x0_dets, x1_dets)




# -----------------------------------------
    # 1. Filtered kernel
    # -----------------------------------------
    plt.figure(figsize=(6, 6))
    # Asumimos que el kernel es de un solo canal (escala de grises)
    plt.imshow(filtered_kernel, cmap='gray')
    plt.title("Filtro de Convolución")
    plt.axis("off")  # Oculta los ejes numéricos para que parezca una imagen pura
    plt.savefig(os.path.join(plot_dir, "conv_filter.pdf"), format="pdf", bbox_inches="tight")
    plt.close() # Cierra la figura para liberar memoria

    # -----------------------------------------
    # 2. Heatmap plot
    # -----------------------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(out, cmap='viridis')
    plt.colorbar()
    plt.title("Matriz de Activación")
    plt.savefig(os.path.join(plot_dir, "activation_matrix.pdf"), format="pdf", bbox_inches="tight")
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
    plt.savefig(os.path.join(plot_dir, "histogram.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

    # -----------------------------------------
    # 4. Bounding box image
    # -----------------------------------------
    bb_img = draw_rectangles(img, detections)

    # ¡Importante! Convertir de BGR (OpenCV) a RGB (Matplotlib)
    bb_img_rgb = cv2.cvtColor(bb_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(bb_img_rgb)
    plt.title("Cajas de Detección")
    plt.axis("off") # Ocultamos los ejes
    plt.savefig(os.path.join(plot_dir, "bounding_boxes.pdf"), format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
