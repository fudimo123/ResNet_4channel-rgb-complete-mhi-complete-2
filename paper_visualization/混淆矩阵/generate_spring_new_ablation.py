import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
CLASS_NAMES = ["Positive", "Negative", "Surprise"]

METRIC_SOURCES = {
    "casme2": PROJECT_ROOT / "casme2_fusion_train_clean copy" / "fusion_result_casme2_se_reprofix" / "final_metrics.txt",
    "samm": PROJECT_ROOT / "SAMM_fusion_train_clean copy" / "fusion_result_samm_se_reprofix_evm_classic_butter_o1" / "final_metrics.txt",
    "smic": PROJECT_ROOT / "smic_fusion_train_clean copy" / "fusion_result_smic_se_reprofix_evm_alpha10_decal_lr_possur" / "final_metrics.txt",
}

INTERMEDIATE_PANELS = {
    "casme2": BASE_DIR / "spring_new_ablation_casme2.png",
    "samm": BASE_DIR / "spring_new_ablation_samm.png",
    "smic": BASE_DIR / "spring_new_ablation_smic.png",
}

PANELS = [
    ("(a) CASMEII", INTERMEDIATE_PANELS["casme2"]),
    ("(b) SAMM", INTERMEDIATE_PANELS["samm"]),
    ("(c) SMIC", INTERMEDIATE_PANELS["smic"]),
    ("(d) DSME", BASE_DIR / "Confusion_Matrix_DSME_New.png"),
    ("(e) 3DB-combined", BASE_DIR / "Confusion_Matrix_CrossDB.png"),
]

OUTPUT_PATH = BASE_DIR / "spring-new-ablation.png"
TOP_ROW = PANELS[:3]
BOTTOM_ROW = PANELS[3:]

BG_COLOR = "white"
TEXT_COLOR = "black"

TARGET_PANEL_WIDTH = 430
TOP_GAP = 20
ROW_GAP = 48
COL_GAP = 34
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


LABEL_FONT = load_font(27)


def parse_mean_confusion_matrix(metrics_path: Path) -> np.ndarray:
    text = metrics_path.read_text(encoding="utf-8")
    match = re.search(r"Mean:\s*\n(\[\[.*?\]\])\s*\n\s*Std Dev:", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find mean confusion matrix in {metrics_path}")

    matrix_block = match.group(1)
    rows = []
    for raw_line in matrix_block.strip().splitlines():
        line = raw_line.strip().strip("[]")
        if not line:
            continue
        rows.append([float(value) for value in line.split()])

    matrix = np.array(rows, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix in {metrics_path}, got {matrix.shape}")
    return matrix


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def generate_panel_image(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    heatmap = sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 38, "weight": "bold"},
        cbar=True,
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
    )

    display_title = r"CASME $\mathrm{II}$" if title == "CASMEII" else title
    ax.set_title(display_title, fontsize=40, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=30, fontweight="bold", labelpad=14)
    ax.set_ylabel("True Label", fontsize=30, fontweight="bold", labelpad=14)
    ax.tick_params(axis="x", labelsize=28, rotation=0)
    ax.tick_params(axis="y", labelsize=28, rotation=0)

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=24)

    fig.tight_layout()
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close(fig)


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


def layout_row(draw: ImageDraw.ImageDraw, canvas: Image.Image, panels, y_top: int, canvas_width: int):
    processed = []
    max_height = 0
    label_block_height = 0

    for label, path in panels:
        panel = resize_panel(crop_panel(path), TARGET_PANEL_WIDTH)
        label_w, label_h = text_size(label, LABEL_FONT)
        processed.append((label, panel, label_w, label_h))
        max_height = max(max_height, panel.height)
        label_block_height = max(label_block_height, label_h)

    total_width = sum(panel.width for _, panel, _, _ in processed) + COL_GAP * (len(processed) - 1)
    x = (canvas_width - total_width) // 2

    for label, panel, label_w, label_h in processed:
        panel_x = x
        panel_y = y_top
        canvas.paste(panel, (panel_x, panel_y))

        label_x = panel_x + (panel.width - label_w) // 2
        label_y = panel_y + panel.height + LABEL_GAP
        draw.text((label_x, label_y), label, fill=TEXT_COLOR, font=LABEL_FONT)
        x += panel.width + COL_GAP

    return y_top + max_height + LABEL_GAP + label_block_height


def compose_combined_figure() -> None:
    top_processed = [resize_panel(crop_panel(path), TARGET_PANEL_WIDTH) for _, path in TOP_ROW]
    bottom_processed = [resize_panel(crop_panel(path), TARGET_PANEL_WIDTH) for _, path in BOTTOM_ROW]

    canvas_width = TARGET_PANEL_WIDTH * 3 + COL_GAP * 2 + 40
    top_height = max(img.height for img in top_processed)
    bottom_height = max(img.height for img in bottom_processed)
    label_height = max(text_size(label, LABEL_FONT)[1] for label, _ in PANELS)

    canvas_height = (
        TOP_GAP
        + top_height
        + LABEL_GAP
        + label_height
        + ROW_GAP
        + bottom_height
        + LABEL_GAP
        + label_height
        + BOTTOM_MARGIN
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    y_after_top = layout_row(draw, canvas, TOP_ROW, TOP_GAP, canvas_width)
    bottom_y = y_after_top + ROW_GAP
    layout_row(draw, canvas, BOTTOM_ROW, bottom_y, canvas_width)

    canvas.save(OUTPUT_PATH, dpi=(300, 300))


def main() -> None:
    sns.set_style("white")

    generate_panel_image(
        normalize_rows(parse_mean_confusion_matrix(METRIC_SOURCES["casme2"])),
        "CASMEII",
        INTERMEDIATE_PANELS["casme2"],
    )
    generate_panel_image(
        normalize_rows(parse_mean_confusion_matrix(METRIC_SOURCES["samm"])),
        "SAMM",
        INTERMEDIATE_PANELS["samm"],
    )
    generate_panel_image(
        normalize_rows(parse_mean_confusion_matrix(METRIC_SOURCES["smic"])),
        "SMIC",
        INTERMEDIATE_PANELS["smic"],
    )
    compose_combined_figure()
    print(f"Saved combined confusion matrix figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
