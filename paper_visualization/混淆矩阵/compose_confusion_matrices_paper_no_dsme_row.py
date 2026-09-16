from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent

PANELS = [
    ("(a) CASMEII", BASE_DIR / "Confusion_Matrix_CASME2_New.png"),
    ("(b) SAMM", BASE_DIR / "Confusion_Matrix_SAMM_Clean.png"),
    ("(c) SMIC", BASE_DIR / "Confusion_Matrix_SMIC_Clean.png"),
    ("(d) 3DB-combined", BASE_DIR / "Confusion_Matrix_CrossDB.png"),
]

OUTPUT_PATH = BASE_DIR / "Confusion_Matrix_Combined_Paper_No_DSME_Row.png"

BG_COLOR = "white"
TEXT_COLOR = "black"

TARGET_PANEL_WIDTH = 330
TOP_GAP = 20
COL_GAP = 24
BOTTOM_MARGIN = 20
LABEL_GAP = 10


def load_font(size: int):
    font_candidates = [
        "times.ttf",
        "Times New Roman.ttf",
        "timesbd.ttf",
        "Times.ttf",
    ]
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


LABEL_FONT = load_font(25)


def crop_panel(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    return image.crop((0, 0, width, int(height * 0.985)))


def resize_panel(image: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def text_size(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def main():
    processed = []
    max_height = 0
    max_label_height = 0

    for label, path in PANELS:
        panel = resize_panel(crop_panel(path), TARGET_PANEL_WIDTH)
        label_w, label_h = text_size(label, LABEL_FONT)
        processed.append((label, panel, label_w, label_h))
        max_height = max(max_height, panel.height)
        max_label_height = max(max_label_height, label_h)

    total_width = sum(panel.width for _, panel, _, _ in processed) + COL_GAP * (len(processed) - 1)
    canvas_width = total_width + 40
    canvas_height = TOP_GAP + max_height + LABEL_GAP + max_label_height + BOTTOM_MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    x = (canvas_width - total_width) // 2
    for label, panel, label_w, label_h in processed:
        panel_x = x
        panel_y = TOP_GAP
        canvas.paste(panel, (panel_x, panel_y))

        label_x = panel_x + (panel.width - label_w) // 2
        label_y = panel_y + panel.height + LABEL_GAP
        draw.text((label_x, label_y), label, fill=TEXT_COLOR, font=LABEL_FONT)
        x += panel.width + COL_GAP

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved 1x4 combined confusion matrix figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
