from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "图像处理前后对比.png"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SAMPLES = [
    {
        "row_label": "Samples in CASME II",
        "raw_dir": PROJECT_ROOT / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected" / "sub02" / "EP13_04",
        "aligned_dir": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "casme2_aligned_rgb" / "sub02" / "EP13_04",
        "dynamic_path": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "dynamic_data" / "sub02" / "s2_EP13_04.jpg",
        "base_dynamic_from_dir": True,
        "decal_dir": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "casme2_static_decalcomanie" / "L" / "sub02" / "EP13_04",
        "evm_dir": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "casme2_aligned_rgb_evm_alpha6" / "sub02" / "EP13_04",
        "evm_dynamic_path": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "casme2_dynamic_data_evm_alpha6" / "sub02" / "s2_EP13_04.jpg",
    },
    {
        "row_label": "Samples in SAMM",
        "raw_dir": PROJECT_ROOT / "data" / "SAMM" / "006" / "006_1_2",
        "aligned_dir": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "samm_aligned_rgb" / "006" / "006_1_2",
        "dynamic_path": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "samm_dynamic_data_clean" / "006" / "s006_006_1_2.jpg",
        "decal_dir": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "samm_static_decalcomanie" / "L" / "006" / "006_1_2",
        "evm_dir": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "samm_aligned_rgb_evm_alpha6" / "006" / "006_1_2",
        "evm_dynamic_path": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "samm_dynamic_data_evm_alpha6" / "006" / "s006_006_1_2.jpg",
    },
    {
        "row_label": "Samples in SMIC",
        "raw_dir": PROJECT_ROOT / "data" / "SMIC2" / "s1" / "micro" / "surprise" / "s1_sur_01",
        "aligned_dir": PROJECT_ROOT / "smic_fusion_train_clean copy" / "smic_aligned_rgb" / "s1" / "s1_sur_01",
        "dynamic_path": PROJECT_ROOT / "smic_fusion_train_clean copy" / "smic_dynamic_data_clean" / "s1" / "s1_s1_sur_01.jpg",
        "decal_dir": PROJECT_ROOT / "smic_fusion_train_clean copy" / "smic_static_decalcomanie" / "L" / "s1" / "s1_sur_01",
        "evm_dir": PROJECT_ROOT / "smic_fusion_train_clean copy" / "smic_aligned_rgb_evm_alpha6" / "s1" / "s1_sur_01",
        "evm_dynamic_path": PROJECT_ROOT / "smic_fusion_train_clean copy" / "smic_dynamic_data_evm_alpha6" / "s1" / "s1_s1_sur_01.jpg",
    },
]

COLUMN_LABELS = [
    "Raw Frame",
    "MTCNN Aligned Face",
    "Base Dynamic Image",
    "Decalcomanie Face",
    "Dynamic Image from Decal",
    "EVM Face (a=6)",
    "Dynamic Image after EVM",
]

BACKGROUND = "white"
TEXT_COLOR = "black"
BORDER_COLOR = "#555555"
ROW_LABEL_WIDTH = 260
CELL_SIZE = 210
COL_GAP = 16
ROW_GAP = 16
LEFT_MARGIN = 24
RIGHT_MARGIN = 24
TOP_MARGIN = 24
BOTTOM_MARGIN = 100
LABEL_GAP = 12


def load_font(size: int):
    for name in ["times.ttf", "Times New Roman.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


ROW_FONT = load_font(28)
COL_FONT = load_font(23)


def text_size(text: str, font):
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def list_image_files(directory: Path):
    files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No image files found in {directory}")
    return files


def select_middle_frame(directory: Path) -> Image.Image:
    files = list_image_files(directory)
    return Image.open(files[len(files) // 2]).convert("RGB")


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_dynamic_image_from_dir(sequence_dir: Path) -> Image.Image:
    frame_paths = list_image_files(sequence_dir)
    frames = [np.array(Image.open(path).convert("RGB"), dtype=np.float32) for path in frame_paths]
    num_frames = len(frames)

    coefficients = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])

    coefficients /= np.sum(np.abs(coefficients))
    dynamic_image = np.zeros_like(frames[0], dtype=np.float32)
    for coeff, frame in zip(coefficients, frames):
        dynamic_image += coeff * frame

    dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
    dynamic_image = np.uint8(dynamic_image)
    return Image.fromarray(dynamic_image)


def fit_to_square(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size - 1, size - 1), outline=BORDER_COLOR, width=1)
    return canvas


def wrap_text(text: str, font, max_width: int):
    words = text.split()
    lines = []
    current = []
    for word in words:
        trial = " ".join(current + [word])
        if text_size(trial, font)[0] <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def build_row_images(sample):
    if sample.get("base_dynamic_from_dir"):
        base_dynamic_image = generate_dynamic_image_from_dir(sample["aligned_dir"])
    else:
        base_dynamic_image = load_image(sample["dynamic_path"])

    return [
        fit_to_square(select_middle_frame(sample["raw_dir"]), CELL_SIZE),
        fit_to_square(select_middle_frame(sample["aligned_dir"]), CELL_SIZE),
        fit_to_square(base_dynamic_image, CELL_SIZE),
        fit_to_square(select_middle_frame(sample["decal_dir"]), CELL_SIZE),
        fit_to_square(generate_dynamic_image_from_dir(sample["decal_dir"]), CELL_SIZE),
        fit_to_square(select_middle_frame(sample["evm_dir"]), CELL_SIZE),
        fit_to_square(load_image(sample["evm_dynamic_path"]), CELL_SIZE),
    ]


def main():
    rows = [build_row_images(sample) for sample in SAMPLES]

    grid_width = len(COLUMN_LABELS) * CELL_SIZE + (len(COLUMN_LABELS) - 1) * COL_GAP
    grid_height = len(SAMPLES) * CELL_SIZE + (len(SAMPLES) - 1) * ROW_GAP
    canvas_width = LEFT_MARGIN + ROW_LABEL_WIDTH + grid_width + RIGHT_MARGIN
    canvas_height = TOP_MARGIN + grid_height + LABEL_GAP + BOTTOM_MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    grid_x0 = LEFT_MARGIN + ROW_LABEL_WIDTH
    grid_y0 = TOP_MARGIN

    for row_idx, (sample, row_images) in enumerate(zip(SAMPLES, rows)):
        y = grid_y0 + row_idx * (CELL_SIZE + ROW_GAP)

        label = sample["row_label"]
        label_w, label_h = text_size(label, ROW_FONT)
        label_x = LEFT_MARGIN + max(0, ROW_LABEL_WIDTH - label_w - 14)
        label_y = y + (CELL_SIZE - label_h) // 2
        draw.text((label_x, label_y), label, fill=TEXT_COLOR, font=ROW_FONT)

        for col_idx, image in enumerate(row_images):
            x = grid_x0 + col_idx * (CELL_SIZE + COL_GAP)
            canvas.paste(image, (x, y))

    col_label_top = grid_y0 + grid_height + LABEL_GAP
    max_label_width = CELL_SIZE - 8
    for col_idx, label in enumerate(COLUMN_LABELS):
        x = grid_x0 + col_idx * (CELL_SIZE + COL_GAP)
        lines = wrap_text(label, COL_FONT, max_label_width)
        line_heights = [text_size(line, COL_FONT)[1] for line in lines]
        total_h = sum(line_heights) + max(0, len(lines) - 1) * 2
        yy = col_label_top + ((BOTTOM_MARGIN - total_h) // 2) - 4
        for line, h in zip(lines, line_heights):
            w, _ = text_size(line, COL_FONT)
            xx = x + (CELL_SIZE - w) // 2
            draw.text((xx, yy), line, fill=TEXT_COLOR, font=COL_FONT)
            yy += h + 2

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
