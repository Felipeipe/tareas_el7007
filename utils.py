from typing import Tuple
import cv2
import numpy as np

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