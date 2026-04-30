from utils import select_roi_opencv, convolution
import numpy as np
import cv2
import os

def main():
    img_name = 'grid_symbols.png'
    thresh = 200
    path = os.path.join('media', img_name)
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    y0, y1, x0, x1 = select_roi_opencv(binary_img)
    filter = binary_img[y0:y1, x0:x1]

    out = convolution(binary_img, filter)
    cv2.imshow("img", out)
    cv2.waitKey(0)
    pass

if __name__=='__main__':
    main()
