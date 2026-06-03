import os

import cv2
import numpy as np
import pywt

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
OUT_PART_3 = os.path.join("out", "wavelet_marking")
ALPHA1 = [0.1, 0.5, 2.0, 5.0]
ALPHA2 = [0.5, 20.0]
ALPHA3 = [4.9]

os.makedirs(OUT_PART_1, exist_ok=True)
os.makedirs(OUT_PART_2, exist_ok=True)
os.makedirs(OUT_PART_3, exist_ok=True)


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

    for alpha in ALPHA1:
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
        ALPHA1,
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
            os.path.join(OUT_PART_2, f"perro_{alpha}.pdf"),  # yo
            bgr_img,
            title=f"Peak Signal-to-Noise Ratio = {peak_snr:.3f}dB",
        )

        true_corr, false_corrs = informed_detection(marked_img, true_key, false_keys)
        true_corr_blind, false_corrs_blind, false_key = blind_detection(
            marked_img, K_band, false_k_bands
        )

        all_true_corrs.append(true_corr)
        all_true_corrs_blind.append(true_corr_blind)
        rmse = np.sqrt((true_key - false_key) ** 2).mean()
        diff = np.abs(true_key - false_key)
        print(f"RMSE for {alpha = }: {rmse}")
        plot_img(
            os.path.join(OUT_PART_2, f"diff_{alpha}.pdf"),
            diff,
            title="Diferencia entre llaves en la detección informada y la detección ciega",
        )
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


def main3():
    N_FALSE_KEYS = 3
    SEED = 10

    img = load_img(DOG_IMG_PATH).astype(np.float64)

    rng = np.random.default_rng(SEED)
    Y = img[:, :, 0]
    LL, (LH, HL, HH) = pywt.dwt2(Y, "haar")

    X_key = rng.standard_normal(LL.shape)
    fake_keys = [rng.standard_normal(LL.shape) for _ in range(N_FALSE_KEYS)]
    psnrs = []
    all_true_corrs = []
    all_fake_corrs = [[] for _ in range(N_FALSE_KEYS)]

    for alpha in ALPHA3:
        HL_mark = HL + alpha * X_key
        LH_mark = LH + alpha * X_key
        coeffs_mark = (LL, (LH_mark, HL_mark, HH))
        Y_mark = pywt.idwt2(coeffs_mark, "haar")

        marked_img = img.copy()
        marked_img[:, :, 0] = Y_mark

        safe_marked_img = np.clip(marked_img, 0, 255)
        current_psnr = PSNR(img, safe_marked_img)
        psnrs.append(current_psnr)

        bgr_marked = cv2.cvtColor(safe_marked_img.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
        plot_img(
            os.path.join(OUT_PART_3, f"marked_img_alpha_{alpha}.pdf"),
            bgr_marked,
            title=f"Marcado Wavelet (Alpha={alpha}, PSNR={current_psnr:.2f}dB)",
        )

        Y_ext = marked_img[:, :, 0]

        LL_ext, (LH_ext, HL_ext, HH_ext) = pywt.dwt2(Y_ext, "haar")


        corr_true_HL = np.corrcoef(HL_ext.flatten(), X_key.flatten())[0, 1]
        corr_true_LH = np.corrcoef(LH_ext.flatten(), X_key.flatten())[0,1]
        all_true_corrs.append((corr_true_HL + corr_true_LH) / 2.0)

        for j in range(N_FALSE_KEYS):
            corr_fake_HL = np.corrcoef(HL_ext.flatten(), fake_keys[j].flatten())[0,1]
            corr_fake_LH = np.corrcoef(LH_ext.flatten(), fake_keys[j].flatten())[0,1]
            all_fake_corrs[j].append((corr_fake_HL + corr_fake_LH) / 2.0)

    plot_detections(
        ALPHA3,
        psnrs,
        all_true_corrs,
        all_fake_corrs,
        os.path.join(OUT_PART_3, "corr_alpha_dwt.pdf"),
    )


if __name__ == "__main__":
    main1()
    main2()
    main3()
