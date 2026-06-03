import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    PCC,
    PSNR,
    apply_watermark,
    blind_detection,
    entropy_map,
    informed_detection,
    kband,
    load_img,
    luma_fft,
    noise_img,
    plot_detections,
    plot_img,
    ring_mask,
)

DOG_IMG_PATH = os.path.join("media", "perro.jpg")
OUT_PART_1 = os.path.join("out", "pure_spread_spectrum")
OUT_PART_2 = os.path.join("out", "spatial_adaptation")
ALPHA = [0.1, 0.5, 2.0, 5.0]
ALPHA2 = [0.5, 20.0]

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

        watermark_img_ycrcb = img.copy()
        watermark_img_ycrcb[:, :, 0] = Y_mark

        all_true_corrs.append(PCC(watermark_img_ycrcb, mask, X_sym))
        for j, fk in enumerate(fake_keys):
            all_fake_corrs[j].append(PCC(watermark_img_ycrcb, mask, fk))

        watermark_img_bgr = cv2.cvtColor(watermark_img_ycrcb, cv2.COLOR_YCrCb2BGR)
        current_psnr = PSNR(bgr_img, watermark_img_bgr)
        psnrs.append(current_psnr)

        plot_img(
            os.path.join(OUT_PART_1, f"perro_{alpha}.pdf"),
            watermark_img_bgr,
            title=f"Peak Signal-to-Noise Ratio = {current_psnr:.3f}dB",
        )

    plot_detections(
        ALPHA,
        psnrs,
        all_true_corrs,
        all_fake_corrs,
        os.path.join(OUT_PART_1, "corr_alpha.pdf"),
    )


def main2():
    img = load_img(DOG_IMG_PATH)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    N_FALSE_KEYS = 3
    Y = img[:, :, 0]
    mask = ring_mask(Y)
    K_band = kband(mask)
    E = entropy_map(img)
    plot_img(
        os.path.join(OUT_PART_2, "entropy_map.pdf"),
        E,
        title="Mapa de entropía de perro",
    )
    psnrs = []
    all_true_corrs = []
    all_true_corrs_blind = []

    all_fake_corrs = [[] for _ in range(N_FALSE_KEYS)]
    all_fake_corrs_blind = [[] for _ in range(N_FALSE_KEYS)]

    for alpha in ALPHA2:
        rng = np.random.default_rng()
        true_key = E * K_band
        false_keys = [rng.standard_normal(true_key.shape) for i in range(N_FALSE_KEYS)]
        false_k_bands = [kband(mask) for i in range(N_FALSE_KEYS)]

        Y_mark = np.clip(Y + alpha * true_key, 0, 255)
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

        true_corr, false_corrs = informed_detection(marked_img, true_key, false_keys)
        true_corr_blind, false_corrs_blind = blind_detection(
            marked_img, K_band, false_k_bands
        )

        all_true_corrs.append(true_corr)
        all_true_corrs_blind.append(true_corr_blind)

        for j in range(N_FALSE_KEYS):
            all_fake_corrs[j].append(false_corrs[j])
            all_fake_corrs_blind[j].append(false_corrs_blind[j])

    plot_detections(
        ALPHA2,
        psnrs,
        all_true_corrs,
        all_fake_corrs,
        os.path.join(OUT_PART_2, "corr_alpha_informed.pdf"),
    )
    plot_detections(
        ALPHA2,
        psnrs,
        all_true_corrs_blind,
        all_fake_corrs_blind,
        os.path.join(OUT_PART_2, "corr_alpha_blind.pdf"),
    )


if __name__ == "__main__":
    main1()
    main2()
