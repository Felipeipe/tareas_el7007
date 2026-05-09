import os

import cv2
import torchvision.models as models
from cv2 import filter2D

from utils import conv_2d, get_filters_from_vgg, load_template


def main():
    filename = "galaga.mp4"
    processed = "processed.mp4"
    filepath = os.path.join("media", filename)
    template = load_template("template.png")
    kern = get_filters_from_vgg()
    filtered_temp = conv_2d(template, kern)

    print(filtered_temp.shape)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(processed, fourcc, framerate, (width, height), isColor=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray)
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
