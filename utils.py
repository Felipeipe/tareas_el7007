import cv2
import numpy as np


def load_img(filepath):
    img = cv2.imread(filepath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # type:ignore


def plot_img(filepath, img, title=""):

    pass


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
    cv2.circle(mask, center=(w // 2, h // 2), radius=outer_radius, color=1, thickness=-1)
    cv2.circle(mask, center=(w // 2, h // 2), radius=inner_radius, color=0, thickness=-1)
    return mask


def apply_watermark(mag, phase, X_sym, mask, alpha=1.0):
    mag_mod = np.maximum(0, mag * (1 + alpha * X_sym * mask))
    f_mod_shifted = mag_mod * np.exp(1j * phase)
    f_mod = np.fft.ifftshift(f_mod_shifted)

    return np.clip(np.real(np.fft.ifft2(f_mod)),0, 255)

def PSNR(img, marked_img):
    mse = np.mean((img-marked_img)**2)
    return 10*np.log10(255**2/mse)

def PCC(marked_img, mask, X_sym):
    img_ycrcb = cv2.cvtColor(marked_img, cv2.COLOR_BGR2YCrCb)
    mag_fft, phase_fft = luma_fft(img_ycrcb)
    mag_masked = mag_fft * mask
    return np.corrcoef(mag_masked.flatten(), X_sym.flatten())[0, 1]
