import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["font.family"] = "DejaVu Sans"

OUTPUT_DIR = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization\混淆矩阵"
CLASS_NAMES = ["Positive", "Negative", "Surprise"]

MATRICES = {
    "Confusion_Matrix_CASME2_New.png": {
        "title": "CASMEⅡ",
        "matrix": np.array([
            [0.88, 0.10, 0.02],
            [0.03, 0.96, 0.01],
            [0.01, 0.15, 0.84],
        ]),
    },
    "Confusion_Matrix_CrossDB.png": {
        "title": "3DB-combined",
        "matrix": np.array([
            [0.71, 0.23, 0.06],
            [0.10, 0.85, 0.06],
            [0.10, 0.20, 0.69],
        ]),
    },
    "Confusion_Matrix_DSME_New.png": {
        "title": "DSME",
        "matrix": np.array([
            [0.88, 0.12, 0.00],
            [0.03, 0.92, 0.04],
            [0.04, 0.25, 0.71],
        ]),
    },
    "Confusion_Matrix_SAMM_Clean.png": {
        "title": "SAMM",
        "matrix": np.array([
            [0.59, 0.40, 0.01],
            [0.01, 0.97, 0.01],
            [0.16, 0.20, 0.64],
        ]),
    },
    "Confusion_Matrix_SMIC_Clean.png": {
        "title": "SMIC",
        "matrix": np.array([
            [0.72, 0.20, 0.08],
            [0.18, 0.71, 0.11],
            [0.06, 0.23, 0.71],
        ]),
    },
}


def generate_confusion_matrix(filename: str, title: str, matrix: np.ndarray) -> None:
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

    display_title = r"CASME $\mathrm{II}$" if title == "CASMEⅡ" else title
    ax.set_title(display_title, fontsize=40, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=30, fontweight="bold", labelpad=14)
    ax.set_ylabel("True Label", fontsize=30, fontweight="bold", labelpad=14)
    ax.tick_params(axis="x", labelsize=28, rotation=0)
    ax.tick_params(axis="y", labelsize=28, rotation=0)

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=24)

    fig.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(save_path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_style("white")

    for filename, config in MATRICES.items():
        generate_confusion_matrix(filename, config["title"], config["matrix"])


if __name__ == "__main__":
    main()
