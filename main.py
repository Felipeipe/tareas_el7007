import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    PCC,
    PSNR,
    apply_watermark,
    entropy_map,
    kband,
    load_img,
    luma_fft,
    noise_img,
    plot_img,
    ring_mask,
)

DOG_IMG_PATH = os.path.join("media", "perro.jpg")
OUT_PART_1 = os.path.join("out", "pure_spread_spectrum")
OUT_PART_2 = os.path.join("out", "spatial_adaptation")
ALPHA = [0.1, 0.5, 2.0, 5.0]

os.makedirs(OUT_PART_1, exist_ok=True)
os.makedirs(OUT_PART_2, exist_ok=True)


def main1():
    img = load_img(DOG_IMG_PATH)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    fft_img, fft_phase = luma_fft(img)

    X_sym = noise_img(fft_img)
    mask = ring_mask(fft_img)
    fake_keys = [noise_img(fft_img) for _ in range(3)]

    psnrs = []
    all_true_corrs = []
    all_fake_corrs = [[] for _ in range(3)]

    for alpha in ALPHA:
        Y_mark = apply_watermark(fft_img, fft_phase, X_sym, mask, alpha)
        watermark_img = img.copy()
        watermark_img[:, :, 0] = Y_mark
        watermark_img = cv2.cvtColor(watermark_img, cv2.COLOR_YCrCb2BGR)
        current_psnr = PSNR(bgr_img, watermark_img)
        plot_img(
            os.path.join(OUT_PART_1, f"perro_{alpha}.pdf"),
            watermark_img,
            title=f"Peak Signal-to-Noise Ratio = {current_psnr:.3f}dB",
        )
        psnrs.append(current_psnr)
        all_true_corrs.append(PCC(watermark_img, mask, X_sym))
        for j, fk in enumerate(fake_keys):
            all_fake_corrs[j].append(PCC(watermark_img, mask, fk))

    x = np.arange(len(ALPHA))
    width = 0.18
    plt.figure(figsize=(10, 6))

    bars_true = plt.bar(
        x - 1.5 * width,
        all_true_corrs,
        width,
        label="Clave Original",
        color="#2ca02c",
        edgecolor="black",
    )
    bars_f1 = plt.bar(
        x - 0.5 * width,
        all_fake_corrs[0],
        width,
        label="Clave Falsa 1",
        color="#ff7f0e",
        edgecolor="black",
    )
    bars_f2 = plt.bar(
        x + 0.5 * width,
        all_fake_corrs[1],
        width,
        label="Clave Falsa 2",
        color="#1f77b4",
        edgecolor="black",
    )
    bars_f3 = plt.bar(
        x + 1.5 * width,
        all_fake_corrs[2],
        width,
        label="Clave Falsa 3",
        color="#d62728",
        edgecolor="black",
    )

    for bars in [bars_true, bars_f1, bars_f2, bars_f3]:
        for bar in bars:
            height = bar.get_height()

            va = "bottom" if height > 0 else "top"
            y_offset = height + (0.002 if height > 0 else -0.002)

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y_offset,
                f"{height:.4f}",
                ha="center",
                va=va,
                fontsize=9,
                rotation=90,
            )

    plt.title(
        "Correlación entre distintas llaves, con distintos valores de alpha",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Coeficiente de Correlación (Pearson)", fontsize=12)
    plt.xlabel("Alpha", fontsize=12)

    x_labels = [f"Alpha = {a}\n(PSNR: {p:.1f} dB)" for a, p in zip(ALPHA, psnrs)]
    plt.xticks(x, x_labels, fontsize=11)

    plt.axhline(0, color="black", linewidth=1)

    plt.margins(y=0.2)

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=11)

    chart_path = os.path.join(OUT_PART_1, "corr_alpha.pdf")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()


def main2():
    img = load_img(DOG_IMG_PATH)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    Y = img[:, :, 0]
    X_sym = noise_img(Y)
    mask = ring_mask(Y)
    K_band = kband(X_sym, mask)
    E = entropy_map(img)
    psnrs = []
    for alpha in ALPHA:
        Y_mark = Y + alpha * E * K_band
        marked_img = img.copy()
        marked_img[:, :, 0] = Y_mark
        bgr_img = cv2.cvtColor(marked_img, cv2.COLOR_YCrCb2BGR)
        peak_snr = PSNR(img, marked_img)
        psnrs.append(peak_snr)

        plot_img(
            os.path.join(OUT_PART_2, f"perro_{alpha}.pdf"),
            bgr_img,
            title=f"Peak Signal-to-Noise Ratio = {peak_snr:.3f}dB",
        )


if __name__ == "__main__":
    main1()
    main2()
