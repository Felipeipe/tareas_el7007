import matplotlib.pyplot as plt
import cv2

def plot_image_comparison(images, titles, bottom_texts, filepath):
    "Plots 4 images side by side"
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    for ax, img, title, text in zip(axes, images, titles, bottom_texts):
        ax.imshow(img)
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.axis("off")

        if text:
            ax.text(
                0.5,
                -0.05,
                text,
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=15,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()


def plot_histogram_comparison(images, titles, filepath):
    "Plots 4 histograms side by side"
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    colors = ("r", "g", "b")

    for ax, img, title in zip(axes, images, titles):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Intensidad de Píxel")
        ax.set_ylabel("Frecuencia")

        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)

        ax.set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
