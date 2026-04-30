from typing import Tuple
import cv2
import numpy as np
import os

def select_roi_opencv(image: np.ndarray, window_name: str = 'Selecciona ROI') -> Tuple[int, int, int, int]:
    """
    Seleccion interactiva con OpenCV.
    Devuelve (y0, y1, x0, x1).

    Uso:
    1) Arrastra para marcar ROI.
    2) Enter o Space para confirmar.
    3) c para cancelar.
    """
    img = image.copy()
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    x, y, w, h = cv2.selectROI(window_name, vis, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    if w == 0 or h == 0:
        raise ValueError('ROI no seleccionada o cancelada.')

    x0, y0 = int(x), int(y)
    x1, y1 = int(x + w), int(y + h)
    return y0, y1, x0, x1

def convolution(img: np.ndarray, filter: np.ndarray):
    """ calculates the convolution of the image with the given filter.
    Adds zero padding to keep the same dimensions as the original image.
    """
    filter_height, filter_width = filter.shape
    img_height, img_width = img.shape

    padded_img = np.zeros((img_height + filter_height - 1, img_width + filter_width - 1))
    padded_img[(filter_height-1)//2 : (filter_height-1)//2 + img_height,
               (filter_width-1)//2 : (filter_width-1)//2 + img_width] = img

    out = np.zeros(img.shape)
    for i in range(img_height):
        for j in range(img_width):
            value = 0.0
            for fi in range(filter_height):
                for fj in range(filter_width):
                    value += padded_img[i + fi, j + fj] * filter[fi, fj]
            out[i, j] = value
    return out

if __name__ == '__main__':
    img_name = 'grid_symbols.png'
    path = os.path.join('media', img_name)
    img = cv2.imread(path)

    print(select_roi_opencv(img))
