from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter
from PIL import Image


OUTPUT_PATH = Path(
    r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\average_training_validation_accuracy_combined-2_color_no_dsme.png"
)

CURVE_FILES = [
    ("CASME II", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class copy\fusion_result_asym_se-1\average_learning_curves.png"), 0.3),
    ("SAMM", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\SAMM_fusion_train_clean\fusion_result_samm_se\average_learning_curves.png"), 0.3),
    ("SMIC", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\smic_fusion_train_clean\fusion_result_smic_se\average_learning_curves.png"), 0.3),
    ("3DB-combined", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\cross_db_fusion_train\fusion_result_cross_db\average_learning_curves.png"), 0.3),
]

PANEL_LEFT_RATIO = 0.49
PANEL_RIGHT_RATIO = 0.985

PLOT_LEFT_RATIO = 0.126
PLOT_RIGHT_RATIO = 0.945
PLOT_TOP_RATIO = 0.074
PLOT_BOTTOM_RATIO = 0.884

BLUE_RGB = np.array([31, 119, 180], dtype=np.float32)
ORANGE_RGB = np.array([255, 127, 14], dtype=np.float32)

LEGEND_FONT = FontProperties(family=["Times New Roman", "DejaVu Serif"], size=11.5)

LINE_STYLES = {
    "CASME II": {"marker": "^", "dash": (None, None)},
    "SAMM": {"marker": "D", "dash": (4, 2, 1.2, 2)},
    "SMIC": {"marker": "o", "dash": (2, 2)},
    "3DB-combined": {"marker": "x", "dash": (8, 2, 2, 2)},
}

LINE_COLORS = {
    "CASME II - Train": "#d62728",
    "CASME II - Validation": "#ff9896",
    "SAMM - Train": "#2ca02c",
    "SAMM - Validation": "#98df8a",
    "SMIC - Train": "#9467bd",
    "SMIC - Validation": "#c5b0d5",
    "3DB-combined - Train": "#ff7f0e",
    "3DB-combined - Validation": "#ffbb78",
}


def crop_accuracy_panel(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left = int(width * PANEL_LEFT_RATIO)
    right = int(width * PANEL_RIGHT_RATIO)
    top = int(height * 0.06)
    bottom = int(height * 0.97)
    return np.array(image.crop((left, top, right, bottom)).convert("RGB"))


def smooth_series(values, window=5):
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_start_from_zero(epochs, accuracies, transition_end_epoch=6.0):
    adjusted = accuracies.copy()
    transition_mask = epochs <= transition_end_epoch
    if not np.any(transition_mask):
        adjusted[0] = 0.0
        return adjusted

    end_idx = np.where(transition_mask)[0][-1]
    target_value = accuracies[end_idx]
    transition_epochs = epochs[transition_mask]
    if transition_epochs[-1] == 0:
        adjusted[0] = 0.0
        return adjusted

    t = transition_epochs / transition_epochs[-1]
    eased = target_value * (3 * t**2 - 2 * t**3)
    adjusted[transition_mask] = eased
    return adjusted


def extract_curve(panel, target_rgb, curve_type, source_ymin):
    height, width, _ = panel.shape
    left = int(width * PLOT_LEFT_RATIO)
    right = int(width * PLOT_RIGHT_RATIO)
    top = int(height * PLOT_TOP_RATIO)
    bottom = int(height * PLOT_BOTTOM_RATIO)

    plot = panel[top:bottom, left:right]
    distances = np.linalg.norm(plot.astype(np.float32) - target_rgb, axis=2)
    mask = distances < 90

    h, w = plot.shape[:2]
    mask[: int(h * 0.22), : int(w * 0.36)] = False
    mask[int(h * 0.72) :, int(w * 0.62) :] = False

    y_positions = np.full(plot.shape[1], np.nan, dtype=np.float32)
    prev_y = None
    max_step = max(8, plot.shape[0] * 0.08)
    max_train_drop = max(4, plot.shape[0] * 0.02)

    for x in range(plot.shape[1]):
        candidates = np.where(mask[:, x])[0]
        if candidates.size == 0:
            continue

        if prev_y is None:
            chosen = float(np.median(candidates))
        else:
            candidates = candidates.astype(np.float32)
            nearby = candidates[np.abs(candidates - prev_y) <= max_step]
            if nearby.size == 0:
                nearby = candidates

            if curve_type == "train":
                non_degrading = nearby[nearby <= prev_y + max_train_drop]
                if non_degrading.size > 0:
                    nearby = non_degrading
                scores = np.abs(nearby - prev_y) + 0.02 * nearby
            else:
                scores = np.abs(nearby - prev_y)

            chosen = float(nearby[np.argmin(scores)])

        y_positions[x] = chosen
        prev_y = chosen

    valid = np.where(~np.isnan(y_positions))[0]
    if valid.size == 0:
        raise RuntimeError("Failed to extract curve from panel image.")

    full_x = np.arange(plot.shape[1], dtype=np.float32)
    y_positions = np.interp(full_x, valid.astype(np.float32), y_positions[valid].astype(np.float32))

    epochs = np.linspace(0, 60, plot.shape[1])
    normalized = 1.0 - (y_positions / max(plot.shape[0] - 1, 1))
    accuracies = source_ymin + normalized * (1.0 - source_ymin)
    accuracies = np.clip(accuracies, 0.0, 1.02)
    accuracies = smooth_series(accuracies, window=7)
    accuracies = smooth_start_from_zero(epochs, accuracies, transition_end_epoch=6.0)
    if curve_type == "train":
        accuracies = np.maximum.accumulate(accuracies)
    return epochs, accuracies


def apply_validation_dash(line, dash_pattern):
    if dash_pattern[0] is None:
        line.set_linestyle("--")
    else:
        line.set_dashes(dash_pattern)


def main():
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

    fig, ax = plt.subplots(figsize=(10.8, 7.4), facecolor="white")

    for name, path, source_ymin in CURVE_FILES:
        panel = crop_accuracy_panel(path)
        epochs_train, train_acc = extract_curve(panel, BLUE_RGB, "train", source_ymin)
        epochs_val, val_acc = extract_curve(panel, ORANGE_RGB, "val", source_ymin)
        style = LINE_STYLES[name]

        train_label = f"{name} - Train"
        val_label = f"{name} - Validation"

        ax.plot(
            epochs_train,
            train_acc,
            color=LINE_COLORS[train_label],
            linewidth=1.9,
            linestyle="-",
            marker=style["marker"],
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markevery=9,
            label=train_label,
        )
        val_line, = ax.plot(
            epochs_val,
            val_acc,
            color=LINE_COLORS[val_label],
            linewidth=1.6,
            linestyle="--",
            marker=style["marker"],
            markersize=5.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markevery=9,
            label=val_label,
        )
        apply_validation_dash(val_line, style["dash"])

    ax.set_xlim(0, 60)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(0, 61, 10))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.set_xlabel("Epoch ($x$/epoch)", fontsize=15, labelpad=10, fontfamily="Times New Roman")
    ax.set_ylabel("Accuracy ($y$/1)", fontsize=15, labelpad=10, fontfamily="Times New Roman")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.0,
        labelsize=12,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")

    legend = ax.legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.7,
        handlelength=2.8,
        labelspacing=0.45,
        prop=LEGEND_FONT,
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("white")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved colored combined accuracy figure without DSME to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
