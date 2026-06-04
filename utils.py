import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
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
    calculated_key = E * true_k_band
    false_keys = [E * fkband for fkband in false_k_bands]
    true_corr, false_corrs = informed_detection(marked_img, calculated_key, false_keys)
    return true_corr, false_corrs, calculated_key


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


def mark_pure_spread_spectrum(img_path, alpha):
    img = load_img(img_path)
    fft_img, fft_phase = luma_fft(img)
    X_sym = noise_img(fft_img)
    mask = ring_mask(fft_img)
    Y_mark = apply_watermark(fft_img, fft_phase, X_sym, mask, alpha)

    watermark_img_ycrcb = img.copy()
    watermark_img_ycrcb[:, :, 0] = Y_mark

    watermark_img_bgr = cv2.cvtColor(watermark_img_ycrcb, cv2.COLOR_YCrCb2BGR)
    return watermark_img_bgr, X_sym, mask


def mark_spatial_adaptation(img_path, alpha):
    img = load_img(img_path)
    Y = img[:, :, 0]
    mask = ring_mask(Y)
    K_band = kband(mask)
    E = entropy_map(img)

    true_key = E * K_band
    Y_mark = np.clip(Y + alpha * true_key, 0, 255)
    marked_img = img.copy()
    marked_img[:, :, 0] = Y_mark

    marked_img_bgr = cv2.cvtColor(marked_img, cv2.COLOR_YCrCb2BGR)
    return marked_img_bgr, true_key, K_band, mask


def mark_wavelet(img_path, alpha):
    img = load_img(img_path).astype(np.float64)
    rng = np.random.default_rng()
    Y = img[:, :, 0]

    LL, (LH, HL, HH) = pywt.dwt2(Y, "haar")
    X_key = rng.standard_normal(LL.shape)

    HL_mark = HL + alpha * X_key
    LH_mark = LH + alpha * X_key
    coeffs_mark = (LL, (LH_mark, HL_mark, HH))
    Y_mark = pywt.idwt2(coeffs_mark, "haar")

    marked_img = img.copy()
    marked_img[:, :, 0] = Y_mark

    safe_marked_img = np.clip(marked_img, 0, 255).astype(np.uint8)

    return safe_marked_img, X_key


def plot_detection_peak(true_corr, fake_corrs, method_name, output_path):
    fake_corrs = np.array(fake_corrs)

    mu = np.mean(fake_corrs)
    sigma = np.std(fake_corrs)
    T = mu + 3 * sigma

    plt.figure(figsize=(10, 5))

    indices_falsas = np.arange(len(fake_corrs))
    plt.scatter(
        indices_falsas,
        fake_corrs,
        color="#7f7f7f",
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Claves Falsas (100)",
        zorder=3,
    )

    index_true = len(fake_corrs)
    plt.scatter(
        index_true,
        true_corr,
        color="#d62728",
        marker="*",
        s=200,
        edgecolors="black",
        label=f"Clave Real ({true_corr:.4f})",
        zorder=4,
    )

    plt.axhline(
        y=T,
        color="darkred",
        linestyle="--",
        linewidth=1.5,
        label=rf"Umbral T = $\mu + 3\sigma$ ({T:.4f})",
        zorder=2,
    )

    plt.axhline(y=0, color="black", linewidth=0.8, linestyle=":", alpha=0.5, zorder=1)

    plt.title(
        f"Análisis de Umbral y Pico de Detección\nMétodo: {method_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Índice del Experimento / Clave", fontsize=11)
    plt.ylabel("Coeficiente de Correlación (Pearson)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.legend(loc="best", fontsize=10)

    plt.margins(x=0.03)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    return T


def PSNR(img, marked_img):
    "NOTE, both images are expected to be on the same colorspace (ie, BGR, RGB...)"
    img_f = img.astype(np.float64)
    marked_f = marked_img.astype(np.float64)
    mse = np.mean((img_f - marked_f) ** 2)
    return 10 * np.log10(255**2 / mse)


def PCC_spread_spectrum(marked_img, mask, X_sym):
    img_ycrcb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2YCrCb)
    mag_fft, phase_fft = luma_fft(img_ycrcb)
    mag_masked = mag_fft * mask
    return np.corrcoef(mag_masked.flatten(), X_sym.flatten())[0, 1]


def PCC(marked_img, X_sym):
    img_ycrcb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2YCrCb)
    Y = img_ycrcb[:, :, 0]
    return np.corrcoef(Y.flatten(), X_sym.flatten())[0, 1]


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


def thresh_pss(alpha, dog_img_path, plot_path):

    marked_bgr_pss, X_sym_true, mask_pss = mark_pure_spread_spectrum(
        dog_img_path, alpha
    )
    marked_ycrcb_pss = cv2.cvtColor(marked_bgr_pss, cv2.COLOR_BGR2YCrCb)

    true_corr_pss = PCC_spread_spectrum(marked_ycrcb_pss, mask_pss, X_sym_true)

    fake_corrs_pss = []
    img_original = load_img(dog_img_path)
    fft_img, _ = luma_fft(img_original)

    for i in range(100):
        X_sym_fake = noise_img(fft_img)
        corr_fake = PCC_spread_spectrum(marked_ycrcb_pss, mask_pss, X_sym_fake)
        fake_corrs_pss.append(corr_fake)

    T_pss = plot_detection_peak(
        true_corr=true_corr_pss,
        fake_corrs=fake_corrs_pss,
        method_name="Pure Spread Spectrum",
        output_path=os.path.join(plot_path, "detection_peak_pss.pdf"),
    )
    return T_pss


def thresh_spatial_adaptation(alpha, dog_img_path, plot_path):
    marked_img_bgr, true_key, K_band, mask = mark_spatial_adaptation(
        dog_img_path, alpha
    )

    marked_ycrcb = cv2.cvtColor(marked_img_bgr, cv2.COLOR_BGR2YCrCb)
    rng = np.random.default_rng()
    false_keys = [rng.standard_normal(mask.shape) for i in range(100)]
    true_corr, false_corrs = informed_detection(marked_ycrcb, true_key, false_keys)
    T_spatial_adaptation = plot_detection_peak(
        true_corr,
        false_corrs,
        method_name="Adaptación Espacial",
        output_path=os.path.join(plot_path, "detection_peak_spatial_adaptation.pdf"),
    )
    return T_spatial_adaptation


def thresh_dwt(alpha, dog_img_path, plot_path):
    marked_img, X_key = mark_wavelet(dog_img_path, alpha)

    Y_ext = marked_img[:, :, 0].astype(np.float64)
    LL_ext, (LH_ext, HL_ext, HH_ext) = pywt.dwt2(Y_ext, "haar")

    rng = np.random.default_rng()
    fake_keys = [rng.standard_normal(LL_ext.shape) for _ in range(100)]

    corr_true_HL = np.corrcoef(HL_ext.flatten(), X_key.flatten())[0, 1]
    corr_true_LH = np.corrcoef(LH_ext.flatten(), X_key.flatten())[0, 1]
    true_corr = (corr_true_HL + corr_true_LH) / 2.0

    false_corrs = []
    for j in range(100):
        corr_fake_HL = np.corrcoef(HL_ext.flatten(), fake_keys[j].flatten())[0, 1]
        corr_fake_LH = np.corrcoef(LH_ext.flatten(), fake_keys[j].flatten())[0, 1]
        false_corrs.append((corr_fake_HL + corr_fake_LH) / 2.0)

    threshold = plot_detection_peak(
        true_corr,
        false_corrs,
        method_name="Dominio Wavelet",
        output_path=os.path.join(plot_path, "detection_peak_wavelet.pdf"),
    )

    return threshold


def attack_gaussian_blur(img_bgr):
    return cv2.GaussianBlur(img_bgr, (15, 15), 5)


def attack_noise(img_bgr):
    noise = np.random.normal(0, 20, img_bgr.shape)
    attacked = img_bgr.astype(np.float64) + noise
    return np.clip(attacked, 0, 255).astype(np.uint8)


def attack_cropping(img_bgr):
    attacked = img_bgr.copy()
    h = attacked.shape[0]
    attacked[h // 2 :, :] = 0
    return attacked


def attack_jpeg(img_bgr):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 15]
    _, encimg = cv2.imencode(".jpg", img_bgr, encode_param)
    attacked = cv2.imdecode(encimg, 1)
    return attacked


def test_attacks(img_bgr_marked, method_name, threshold, original_corr, img_path, detect_func):
    print(f"\n{'=' * 50}")
    print(f"Evaluando ataques para: {method_name}")
    print(f"Umbral de supervivencia (T): {threshold:.4f}")
    print(f"Correlación original: {original_corr:.4f}")
    print(f"{'-' * 50}")

    attacks = {
        "Gaussian Blur": attack_gaussian_blur,
        "Ruido Aleatorio": attack_noise,
        "Cropping": attack_cropping,
        "Compresión JPEG": attack_jpeg,
    }

    for attack_name, attack_func in attacks.items():
        # 1. Aplicamos el ataque
        attacked_img = attack_func(img_bgr_marked)

        # 2. Calculamos la nueva correlación dinámicamente para ESTE ataque en específico
        new_corr = detect_func(attacked_img)

        # 3. Verificamos supervivencia y graficamos
        survives = "SÍ" if new_corr > threshold else "NO"

        plot_img(
            os.path.join(img_path, f"{method_name}_vs_{attack_name.replace(' ', '_')}.pdf"),
            attacked_img,
            title=f"Ataque: {attack_name: <18} | Nueva Corr: {new_corr:.4f} | Sobrevive: {survives}"
        )
        print(f"Ataque: {attack_name: <18} | Nueva Corr: {new_corr:.4f} | Sobrevive: {survives}")
