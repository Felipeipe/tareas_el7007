import os

import cv2

from utils import (
    conv_2d,
    detect,
    draw_rectangles,
    get_best_filtered_temp,
    get_filters_from_vgg,
    load_template,
    map_collapse,
)


def main():
    thresh = 4.5e6
    filename = "galaga.mp4"
    processed = "processed.mp4"
    filepath = os.path.join("media", filename)
    template = load_template(os.path.join("templates","template.png"))
    y1, x1 = template.shape
    template_coords = 0, y1, 0, x1
    plot_dir = os.path.join("plots", "tracking")
    os.makedirs(plot_dir, exist_ok=True)

    kern = get_filters_from_vgg()
    filtered_temp = conv_2d(template, kern)
    best_filtered, best_kern = get_best_filtered_temp(
        filtered_temp, kern, filename=os.path.join(plot_dir, "best_filters.pdf")
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(processed, fourcc, framerate, (width, height))
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        char_map = conv_2d(gray, best_kern)
        activation_map = conv_2d(char_map, best_filtered)
        saliency_map = map_collapse(activation_map)
        bbox_coordinates = detect(saliency_map, template_coords, threshold=int(thresh))
        if len(bbox_coordinates[0]) == 0:
            print(f"Frame {frame_number}: Sin detecciones.")
        new_img = draw_rectangles(frame, bbox_coordinates)
        texto = f"Frame: {frame_number}"
        posicion = (30, 50)
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala = 1.0
        color = (0, 255, 255)
        grosor = 2
        cv2.putText(new_img, texto, posicion, fuente, escala, color, grosor, cv2.LINE_AA)
        out.write(new_img)
        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
