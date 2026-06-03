import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk


def load_img(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # type:ignore


def plot_img(filepath, img, title=""):
    "NOTE: img must be in BGR colorspace"
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        img_disp = img
        cmap = "gray"
    plt.figure(figsize=(8, 8))
    plt.imshow(img_disp, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def luma_fft(img):
    """Receives an image in YCrCb and returns the fft of the luminance channel."""
    luminance = img[:, :, 0]
    f = np.fft.fft2(luminance)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    phase = np.angle(fshift)
    return mag, phase


def noise_img(fft_img):
    rng = np.random.default_rng()
    X = rng.standard_normal(fft_img.shape)
    X_rot = np.rot90(X, k=2, axes=(0, 1))
    return (X + X_rot) / 2


def ring_mask(fft_img, inner_radius=40, outer_radius=100):
    h, w = fft_img.shape
    mask = np.zeros(fft_img.shape)
    cv2.circle(
        mask, center=(w // 2, h // 2), radius=outer_radius, color=1, thickness=-1
    )
    cv2.circle(
        mask, center=(w // 2, h // 2), radius=inner_radius, color=0, thickness=-1
    )
    return mask


def kband(mask):
    rng = np.random.default_rng()
    X = rng.standard_normal(mask.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(X) * mask))


def informed_detection(marked_img, true_key, false_keys):
    "marked_img must be on YCrCb colorspace"
    Y_mark = marked_img[:, :, 0]
    true_corr = np.corrcoef(Y_mark.flatten(), true_key.flatten())[0, 1]
    false_corrs = []
    for fk in false_keys:
        false_corrs.append(np.corrcoef(Y_mark.flatten(), fk.flatten())[0, 1])
    return true_corr, false_corrs


def blind_detection(marked_img, true_k_band, false_k_bands):
    "marked_img must be on ycrcb colorspace"
    E = entropy_map(marked_img)
    true_key = E * true_k_band
    false_keys = [E * fkband for fkband in false_k_bands]
    true_corr, false_corrs = informed_detection(marked_img, true_key, false_keys)
    return true_corr, false_corrs


def entropy_map(img):
    """
    NOTE: img must be on YCrCb colorspace
    """
    Y = img[:, :, 0]
    E_raw = entropy(Y, disk(4))
    E_min = E_raw.min()
    E_max = E_raw.max()
    if E_max - E_min == 0:
        return np.zeros_like(E_raw, dtype=np.float64)
    E_norm = (E_raw - E_min) / (E_max - E_min)

    return E_norm


def apply_watermark(mag, phase, X_sym, mask, alpha=1.0):
    mag_mod = np.maximum(0, mag * (1 + alpha * X_sym * mask))
    f_mod_shifted = mag_mod * np.exp(1j * phase)
    f_mod = np.fft.ifftshift(f_mod_shifted)

    return np.clip(np.real(np.fft.ifft2(f_mod)), 0, 255)


def PSNR(img, marked_img):
    "NOTE, both images are expected to be on the same colorspace (ie, BGR, RGB...)"
    img_f = img.astype(np.float64)
    marked_f = marked_img.astype(np.float64)
    mse = np.mean((img_f - marked_f) ** 2)
    return 10 * np.log10(255**2 / mse)


def PCC(marked_img, mask, X_sym):
    img_ycrcb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2YCrCb)
    mag_fft, phase_fft = luma_fft(img_ycrcb)
    mag_masked = mag_fft * mask
    return np.corrcoef(mag_masked.flatten(), X_sym.flatten())[0, 1]


def plot_detections(ALPHA, psnrs, true_corrs, fake_corrs, output_path):
    x = np.arange(len(ALPHA))
    width = 0.18
    plt.figure(figsize=(10, 6))

    bars_true = plt.bar(
        x - 1.5 * width,
        true_corrs,
        width,
        label="Clave Original",
        color="#2ca02c",
        edgecolor="black",
    )
    bars_f1 = plt.bar(
        x - 0.5 * width,
        fake_corrs[0],
        width,
        label="Clave Falsa 1",
        color="#ff7f0e",
        edgecolor="black",
    )
    bars_f2 = plt.bar(
        x + 0.5 * width,
        fake_corrs[1],
        width,
        label="Clave Falsa 2",
        color="#1f77b4",
        edgecolor="black",
    )
    bars_f3 = plt.bar(
        x + 1.5 * width,
        fake_corrs[2],
        width,
        label="Clave Falsa 3",
        color="#d62728",
        edgecolor="black",
    )

    for bars in [bars_true, bars_f1, bars_f2, bars_f3]:
        plt.bar_label(bars, fmt="%.4f", padding=4, fontsize=9, rotation=90)

    plt.title(
        "Correlación entre distintas llaves, con distintos valores de alpha",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Coeficiente de Correlación (Pearson)", fontsize=12)
    plt.xlabel("Alpha", fontsize=12)

    x_labels = [f"Alpha = {a}\n(PSNR: {p:.1f} dB)" for a, p in zip(ALPHA, psnrs)]
    plt.xticks(x, x_labels, fontsize=11)

    # Zorder asegura que la línea del cero no tape los bordes de las barras
    plt.axhline(0, color="black", linewidth=1, zorder=0)

    plt.margins(y=0.25)  # Aumentado ligeramente para acomodar el texto rotado
    plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    plt.legend(loc="best", fontsize=11)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
