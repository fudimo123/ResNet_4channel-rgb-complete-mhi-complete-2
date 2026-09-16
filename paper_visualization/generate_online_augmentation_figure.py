import os
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt


SOURCE_DIR = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub20\EP15_03f"
OUTPUT_PATH = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\online_augmentation_demo_en.png"


def load_reference_image():
    image_files = sorted(
        [
            os.path.join(SOURCE_DIR, name)
            for name in os.listdir(SOURCE_DIR)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    if not image_files:
        raise FileNotFoundError(f"No image files found in {SOURCE_DIR}")
    return Image.open(image_files[len(image_files) // 2]).convert("RGB")


def rotate_image(image, angle):
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(235, 235, 235))


def adjust_brightness(image, factor):
    return ImageEnhance.Brightness(image).enhance(factor)


def adjust_contrast(image, factor):
    return ImageEnhance.Contrast(image).enhance(factor)


def translate_image(image, tx_ratio, ty_ratio):
    width, height = image.size
    tx = int(width * tx_ratio)
    ty = int(height * ty_ratio)
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, 0, tx, 0, 1, ty),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(235, 235, 235),
    )


def main():
    base_image = load_reference_image()
    images = [
        base_image,
        ImageOps.mirror(base_image),
        rotate_image(base_image, 4),
        rotate_image(base_image, 8),
        rotate_image(base_image, -4),
        rotate_image(base_image, -8),
        adjust_brightness(base_image, 0.7),
        adjust_brightness(base_image, 1.3),
        adjust_contrast(base_image, 0.7),
        adjust_contrast(base_image, 1.3),
        translate_image(base_image, 0.05, 0.0),
        translate_image(base_image, -0.05, 0.0),
        translate_image(base_image, 0.0, 0.05),
        translate_image(base_image, 0.0, -0.05),
        translate_image(base_image, 0.05, 0.05),
    ]
    labels = [
        "Original",
        "Horizontal Flip",
        "Rotation +4°",
        "Rotation +8°",
        "Rotation -4°",
        "Rotation -8°",
        "Brightness 0.7x",
        "Brightness 1.3x",
        "Contrast 0.7x",
        "Contrast 1.3x",
        "Shift Right 5%",
        "Shift Left 5%",
        "Shift Down 5%",
        "Shift Up 5%",
        "Shift Lower Right 5%",
    ]

    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 5, figsize=(15, 8.6), facecolor="#e6e6e6")
    plt.subplots_adjust(wspace=0.04, hspace=0.34, left=0.03, right=0.97, top=0.97, bottom=0.06)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx])
        ax.axis("off")
        ax.text(
            0.5,
            -0.08,
            labels[idx],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=20,
            fontfamily="Times New Roman",
            fontweight="normal",
        )

    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved augmentation figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
