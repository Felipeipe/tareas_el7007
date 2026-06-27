import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import (
    clahe_l,
    compute_lab_statistics,
    equal_hist_l,
    plot_histogram_rgb,
    plot_img,
    reinhard_normalization,
)
from plotter import plot_image_comparison, plot_histogram_comparison

def run_analysis_pipeline():
    image_dir = os.path.join("data", "images")
    ref_path = os.path.join("data", "reference.tif")
    # out_dir = os.path.join("informe","img","out_analysis")
    out_dir = os.path.join("out_analysis")
    os.makedirs(out_dir, exist_ok=True)

    ref_img_bgr = cv2.imread(ref_path)
    if ref_img_bgr is None:
        print("Error: Could not load reference image.")
        return
    ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)

    img_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith('.tif')
    ]

    stats_collection = {"Original": [], "HE": [], "CLAHE": [], "Reinhard": []}

    for path in img_paths:
        filename = os.path.basename(path)
        img_name = os.path.splitext(filename)[0]

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) #type:ignore

        img_he = equal_hist_l(img)
        img_clahe = clahe_l(img)
        img_reinhard = reinhard_normalization(img, ref_img)

        images = [img, img_he, img_clahe, img_reinhard]
        titles = ["Original", "HE", "CLAHE", "Reinhard"]
        bottom_texts = []

        for name, t_img in zip(["Original", "HE", "CLAHE", "Reinhard"], images):
            mean_lab, std_lab = compute_lab_statistics(t_img)
            stats_collection[name].append(mean_lab)

            stats_text = (
                f"L: $\\mu={mean_lab[0]:.1f}$, $\\sigma={std_lab[0]:.1f}$\n"
                f"a: $\\mu={mean_lab[1]:.1f}$, $\\sigma={std_lab[1]:.1f}$\n"
                f"b: $\\mu={mean_lab[2]:.1f}$, $\\sigma={std_lab[2]:.1f}$"
            )
            bottom_texts.append(stats_text)

        plot_image_comparison(
            images, titles, bottom_texts,
            os.path.join(out_dir, f"{img_name}_comparison_img.pdf")
        )

        plot_histogram_comparison(
            images, titles,
            os.path.join(out_dir, f"{img_name}_comparison_hist.pdf")
        )

    print("--- Análisis de Variabilidad (Desviación Estándar de las Medias LAB) ---")
    for method, means_list in stats_collection.items():
        means_array = np.array(means_list)
        std_of_means = np.std(means_array, axis=0)
        print(
            f"{method}: L_std={std_of_means[0]:.2f}, a_std={std_of_means[1]:.2f}, b_std={std_of_means[2]:.2f}"
        )

def classic_seg_pipeline():
    pass
if __name__ == "__main__":
    run_analysis_pipeline()
    classic_seg_pipeline()
