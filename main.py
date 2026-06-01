import os

import cv2
import numpy as np

from utils import PCC, PSNR, apply_watermark, load_img, luma_fft, noise_img, ring_mask

DOG_IMG_PATH = os.path.join("media", "perro.jpg")
OUT_PART_1 = os.path.join("out","pure_spread_spectrum")
ALPHA = [0.5, 2.0, 5.0]


def main():
    img = load_img(DOG_IMG_PATH)
    fft_img, fft_phase = luma_fft(img)
    X_sym = noise_img(fft_img)
    mask = ring_mask(fft_img)
    corr = np.zeros(len(ALPHA))
    peak_snr = np.zeros(len(ALPHA))
    for i, alpha in enumerate(ALPHA):
        Y_mark = apply_watermark(fft_img, fft_phase, X_sym, mask, alpha)
        watermark_img = img
        watermark_img[:, :, 0] = Y_mark
        watermark_img = cv2.cvtColor(watermark_img, cv2.COLOR_YCrCb2BGR)
        cv2.imshow(f"PSNR = {PSNR(img, watermark_img)}", watermark_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Pearson's Correlation coeficient {PCC(watermark_img, mask, X_sym):.2f}")


if __name__ == "__main__":
    main()
