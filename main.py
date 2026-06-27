import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cellpose.models import CellposeModel

from plotter import plot_histogram_comparison, plot_image_comparison
from utils import (
    clahe_l,
    classic_segmentation,
    compute_lab_statistics,
    compute_metrics,
    deep_segmentation,
    equal_hist_l,
    get_H_and_E_channels,
    plot_histogram_rgb,
    plot_img,
    prediction_resize,
    reinhard_normalization,
)


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
        os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")
    ]

    stats_collection = {"Original": [], "HE": [], "CLAHE": [], "Reinhard": []}

    for path in img_paths:
        filename = os.path.basename(path)
        img_name = os.path.splitext(filename)[0]

        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # type:ignore

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
            images,
            titles,
            bottom_texts,
            os.path.join(out_dir, f"{img_name}_comparison_img.pdf"),
        )

        plot_histogram_comparison(
            images, titles, os.path.join(out_dir, f"{img_name}_comparison_hist.pdf")
        )

    print("--- Análisis de Variabilidad (Desviación Estándar de las Medias LAB) ---")
    for method, means_list in stats_collection.items():
        means_array = np.array(means_list)
        std_of_means = np.std(means_array, axis=0)
        print(
            f"{method}: L_std={std_of_means[0]:.2f}, a_std={std_of_means[1]:.2f}, b_std={std_of_means[2]:.2f}"
        )


def evaluate_pipelines(preproc=None):
    image_dir = os.path.join("data", "images")
    mask_dir = os.path.join("data", "masks")
    ref_path = os.path.join("data", "reference.tif")

    out_dir = "out_eval"
    os.makedirs(out_dir, exist_ok=True)

    ref_img_bgr = cv2.imread(ref_path)
    ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)  # type:ignore

    cellpose_model = CellposeModel(gpu=True)

    img_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    pipelines = ["Classic", "DeepLearning"]
    preprocs = ["Original", "HE", "CLAHE", "Reinhard"]

    results = {
        p: {prep: {"dice": [], "iou": []} for prep in preprocs} for p in pipelines
    }

    for img_file in img_files:
        print(f"Evaluando: {img_file}")
        img_name = os.path.splitext(img_file)[0]

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file.replace(".tif", ".png"))

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        imgs_prep = {
            "Original": img,
            "HE": equal_hist_l(img),
            "CLAHE": clahe_l(img),
            "Reinhard": reinhard_normalization(img, ref_img),
        }

        for prep_name, prep_img in imgs_prep.items():
            print(f"\tPreproc: {prep_name}...")
            h_channel, _ = get_H_and_E_channels(prep_img)
            pred_classic = classic_segmentation(h_channel)

            pred_classic_res = prediction_resize(pred_classic, gt_mask)
            d_classic, i_classic = compute_metrics(pred_classic_res, gt_mask)

            results["Classic"][prep_name]["dice"].append(d_classic)
            results["Classic"][prep_name]["iou"].append(i_classic)

            mask_classic_plot = (pred_classic_res > 0).astype(np.uint8) * 255

            masks_dl, _, _ = cellpose_model.eval(
                prep_img, diameter=None, channels=[0, 0]
            )

            pred_dl_res = prediction_resize(masks_dl, gt_mask)
            d_dl, i_dl = compute_metrics(pred_dl_res, gt_mask)

            results["DeepLearning"][prep_name]["dice"].append(d_dl)
            results["DeepLearning"][prep_name]["iou"].append(i_dl)

            mask_dl_plot = (pred_dl_res > 0).astype(np.uint8) * 255

            images_to_plot = [prep_img, gt_mask, mask_classic_plot, mask_dl_plot]

            titles = [
                f"{prep_name}",
                "Ground Truth",
                "Classic",
                "Cellpose",
            ]

            bottom_texts = ["", "", f"Dice: {d_classic:.2f}  \nIoU: {i_classic:.2f}", f"Dice: {d_dl:.2f}  \nIoU: {i_dl:.2f}"]

            plot_filepath = os.path.join(out_dir, f"{img_name}_{prep_name}_eval.pdf")
            plot_image_comparison(images_to_plot, titles, bottom_texts, plot_filepath)

    print("\n" + "=" * 50)
    print("RESULTADOS DE EVALUACIÓN (Media ± Std)")
    print("=" * 50)

    for p in pipelines:
        print(f"\n--- Pipeline: {p} ---")
        for prep in preprocs:
            dice_arr = np.array(results[p][prep]["dice"])
            iou_arr = np.array(results[p][prep]["iou"])

            print(
                f"Preprocesamiento: {prep:<10} | "
                f"Dice: {dice_arr.mean():.4f} ± {dice_arr.std():.4f} | "
                f"IoU: {iou_arr.mean():.4f} ± {iou_arr.std():.4f}"
            )


if __name__ == "__main__":
    evaluate_pipelines()
