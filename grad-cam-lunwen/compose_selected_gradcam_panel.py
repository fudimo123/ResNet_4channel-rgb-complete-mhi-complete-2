from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent

PANELS = [
    ("(a) Negative Sample", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub01_EP19_05f_negative" / "dual_branch_gradcampp_panel.png"),
    ("(b) Surprise Sample", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub02_EP13_04_surprise" / "dual_branch_gradcampp_panel.png"),
    ("(c) Positive Sample", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub09_EP02_01f_positive" / "dual_branch_gradcampp_panel.png"),
]

OUTPUT_PATH = BASE_DIR / "gradcam_selected_for_paper.png"

BACKGROUND = "white"
TEXT_COLOR = "black"
TARGET_WIDTH = 920
CROP_TOP_PIXELS = 58
MASK_TOP_PIXELS = 14
TOP_MARGIN = 24
BOTTOM_MARGIN = 28
ROW_GAP = 24
LABEL_GAP = 10


def load_font(size: int):
    candidates = [
        "times.ttf",
        "Times New Roman.ttf",
        "timesbd.ttf",
        "Times.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


LABEL_FONT = load_font(30)


def text_size(text: str, font):
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def resize_keep_ratio(image: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def crop_panel_header(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_top = min(CROP_TOP_PIXELS, max(0, height - 1))
    cropped = image.crop((0, crop_top, width, height))
    if MASK_TOP_PIXELS > 0:
        draw = ImageDraw.Draw(cropped)
        draw.rectangle((0, 0, cropped.width, min(MASK_TOP_PIXELS, cropped.height)), fill="white")
    return cropped


def main() -> None:
    processed = []
    max_width = 0
    total_height = TOP_MARGIN + BOTTOM_MARGIN

    for label, path in PANELS:
        image = Image.open(path).convert("RGB")
        image = crop_panel_header(image)
        image = resize_keep_ratio(image, TARGET_WIDTH)
        label_w, label_h = text_size(label, LABEL_FONT)
        processed.append((label, image, label_w, label_h))
        max_width = max(max_width, image.width)
        total_height += image.height + LABEL_GAP + label_h

    total_height += ROW_GAP * (len(processed) - 1)
    canvas_width = max_width + 80

    canvas = Image.new("RGB", (canvas_width, total_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    y = TOP_MARGIN
    for idx, (label, image, label_w, label_h) in enumerate(processed):
        x = (canvas_width - image.width) // 2
        canvas.paste(image, (x, y))
        label_x = (canvas_width - label_w) // 2
        label_y = y + image.height + LABEL_GAP
        draw.text((label_x, label_y), label, fill=TEXT_COLOR, font=LABEL_FONT)
        y = label_y + label_h
        if idx != len(processed) - 1:
            y += ROW_GAP

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved selected Grad-CAM panel to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
