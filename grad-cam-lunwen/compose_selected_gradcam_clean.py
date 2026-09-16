from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent

PANELS = [
    ("(a) Negative", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub01_EP19_05f_negative" / "dual_branch_gradcampp_panel.png"),
    ("(b) Surprise", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub02_EP13_04_surprise" / "dual_branch_gradcampp_panel.png"),
    ("(c) Positive", BASE_DIR / "outputs_batch5_rgb_cbam_dyn_layer3_gradcampp" / "sub09_EP02_01f_positive" / "dual_branch_gradcampp_panel.png"),
]

OUTPUT_PATH = BASE_DIR / "gradcam_selected_for_paper_clean_horizontal.png"

# Coordinates measured from the exported Grad-CAM++ panel.
RGB_INPUT_BOX = (30, 194, 1045, 1208)
RGB_OVERLAY_BOX = (2567, 194, 3582, 1208)
DUAL_OVERLAY_BOX = (2567, 1321, 3582, 2336)

ROW_LABELS = [
    "RGB Input",
    "RGB Overlay",
    "Dual-Branch Overlay",
]

BACKGROUND = "white"
TEXT_COLOR = "black"
ROW_LABEL_AREA = 260
TOP_MARGIN = 36
BOTTOM_MARGIN = 36
LEFT_MARGIN = 28
RIGHT_MARGIN = 28
COLUMN_GAP = 34
ROW_GAP = 24
HEADER_GAP = 14
ROW_LABEL_GAP = 14
TILE_TARGET_WIDTH = 350


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


HEADER_FONT = load_font(30)
ROW_FONT = load_font(28)


def text_size(text: str, font):
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def crop_tile(image: Image.Image, box):
    return image.crop(box)


def resize_keep_ratio(image: Image.Image, target_width: int) -> Image.Image:
    scale = target_width / image.width
    target_height = int(round(image.height * scale))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def center_text_x(area_x: int, area_width: int, text_width: int) -> int:
    return area_x + (area_width - text_width) // 2


def build_column_tiles(panel_path: Path):
    panel = Image.open(panel_path).convert("RGB")
    tiles = [
        resize_keep_ratio(crop_tile(panel, RGB_INPUT_BOX), TILE_TARGET_WIDTH),
        resize_keep_ratio(crop_tile(panel, RGB_OVERLAY_BOX), TILE_TARGET_WIDTH),
        resize_keep_ratio(crop_tile(panel, DUAL_OVERLAY_BOX), TILE_TARGET_WIDTH),
    ]
    return tiles


def main() -> None:
    columns = []
    header_h_max = 0

    for header, panel_path in PANELS:
        tiles = build_column_tiles(panel_path)
        header_w, header_h = text_size(header, HEADER_FONT)
        header_h_max = max(header_h_max, header_h)
        columns.append((header, header_w, header_h, tiles))

    tile_width = columns[0][3][0].width
    tile_height = columns[0][3][0].height

    content_width = ROW_LABEL_AREA + ROW_LABEL_GAP + (tile_width * len(columns)) + (COLUMN_GAP * (len(columns) - 1))
    canvas_width = LEFT_MARGIN + content_width + RIGHT_MARGIN
    content_height = header_h_max + HEADER_GAP + (tile_height * len(ROW_LABELS)) + (ROW_GAP * (len(ROW_LABELS) - 1))
    canvas_height = TOP_MARGIN + content_height + BOTTOM_MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    grid_x0 = LEFT_MARGIN + ROW_LABEL_AREA + ROW_LABEL_GAP
    grid_y0 = TOP_MARGIN + header_h_max + HEADER_GAP

    for col_idx, (header, header_w, _, tiles) in enumerate(columns):
        x = grid_x0 + col_idx * (tile_width + COLUMN_GAP)
        header_x = center_text_x(x, tile_width, header_w)
        draw.text((header_x, TOP_MARGIN), header, fill=TEXT_COLOR, font=HEADER_FONT)

        for row_idx, tile in enumerate(tiles):
            y = grid_y0 + row_idx * (tile_height + ROW_GAP)
            canvas.paste(tile, (x, y))

    for row_idx, label in enumerate(ROW_LABELS):
        label_w, label_h = text_size(label, ROW_FONT)
        area_y = grid_y0 + row_idx * (tile_height + ROW_GAP)
        label_x = LEFT_MARGIN + max(0, ROW_LABEL_AREA - label_w)
        label_y = area_y + (tile_height - label_h) // 2
        draw.text((label_x, label_y), label, fill=TEXT_COLOR, font=ROW_FONT)

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved clean horizontal Grad-CAM figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
