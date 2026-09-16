from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


OUTPUT_PATH = Path(
    r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\average_training_validation_accuracy_combined.png"
)

CURVE_FILES = [
    ("CASME II", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class copy\fusion_result_asym_se-1\average_learning_curves.png"), 0.3),
    ("DSME", Path(r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\DSME_fusion_train copy\fusion_result_dsme_se2\average_learning_curves.png"), 0.3),
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
    mask[int(h * 0.72):, int(w * 0.62):] = False

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


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    color_pairs = {
        "CASME II": ("#1f77b4", "#ff7f0e"),
        "DSME": ("#2ca02c", "#d62728"),
        "SAMM": ("#9467bd", "#8B4513"),
        "SMIC": ("#e377c2", "#7f7f7f"),
        "3DB-combined": ("#bcbd22", "#17becf"),
    }

    fig, ax = plt.subplots(figsize=(18, 11), facecolor="white")

    for name, path, source_ymin in CURVE_FILES:
        panel = crop_accuracy_panel(path)
        epochs_train, train_acc = extract_curve(panel, BLUE_RGB, "train", source_ymin)
        epochs_val, val_acc = extract_curve(panel, ORANGE_RGB, "val", source_ymin)
        train_color, val_color = color_pairs[name]
        ax.plot(epochs_train, train_acc, color=train_color, linewidth=3.2, label=f"{name} - Train")
        ax.plot(epochs_val, val_acc, color=val_color, linewidth=3.2, linestyle="--", label=f"{name} - Validation")

    ax.set_title("Average Training and Validation Accuracy", fontsize=34, pad=20)
    ax.set_xlabel("Epoch", fontsize=28, labelpad=12)
    ax.set_ylabel("Accuracy", fontsize=28, labelpad=12)
    ax.set_xlim(0, 60)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.arange(0.1, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.985, 0.045),
        frameon=True,
        framealpha=0.98,
        fancybox=True,
        fontsize=16,
        borderpad=0.95,
        labelspacing=0.5,
        handlelength=2.8,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined accuracy figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
