from pathlib import Path


VARIANTS = {
    "alpha6": {
        "rgb_dir": "samm_aligned_rgb_evm_alpha6",
        "dyn_dir": "samm_dynamic_data_evm_alpha6",
    },
    "alpha10": {
        "rgb_dir": "samm_aligned_rgb_evm_alpha10",
        "dyn_dir": "samm_dynamic_data_evm_alpha10",
    },
    "alpha12": {
        "rgb_dir": "samm_aligned_rgb_evm_alpha12",
        "dyn_dir": "samm_dynamic_data_evm_alpha12",
    },
    "fh25": {
        "rgb_dir": "samm_aligned_rgb_evm_fh25",
        "dyn_dir": "samm_dynamic_data_evm_fh25",
    },
    "fh35": {
        "rgb_dir": "samm_aligned_rgb_evm_fh35",
        "dyn_dir": "samm_dynamic_data_evm_fh35",
    },
    "fh40": {
        "rgb_dir": "samm_aligned_rgb_evm_fh40",
        "dyn_dir": "samm_dynamic_data_evm_fh40",
    },
    "pyr2": {
        "rgb_dir": "samm_aligned_rgb_evm_pyr2",
        "dyn_dir": "samm_dynamic_data_evm_pyr2",
    },
    "alpha10_fh35": {
        "rgb_dir": "samm_aligned_rgb_evm_alpha10_fh35",
        "dyn_dir": "samm_dynamic_data_evm_alpha10_fh35",
    },
    "classic_butter_o1": {
        "rgb_dir": "samm_aligned_rgb_evm_classic_butter_o1",
        "dyn_dir": "samm_dynamic_data_evm_classic_butter_o1",
    },
    "classic_butter_o2": {
        "rgb_dir": "samm_aligned_rgb_evm_classic_butter_o2",
        "dyn_dir": "samm_dynamic_data_evm_classic_butter_o2",
    },
    "classic_fft_ideal": {
        "rgb_dir": "samm_aligned_rgb_evm_classic_fft_ideal",
        "dyn_dir": "samm_dynamic_data_evm_classic_fft_ideal",
    },
}


TEMPLATE = """import os
import samm_fusion_train as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, '{result_dir}')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, '{rgb_dir}')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, '{dyn_dir}')

TRANSFER_WEIGHTS_PATH = None


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print("Starting SAMM Fusion Training with SE-Fusion (Adaptive Weighting, reprofix + {variant})...")
    print("Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print("Strategy: {variant} RGB/Dynamic inputs + Train Random Frame RGB + Val/Test Middle Frame RGB + Label Smoothing (0.1)")
    print(f"RGB dir: {{ft.ALIGNED_RGB_DIR}}")
    print(f"Dynamic dir: {{ft.DYNAMIC_IMAGE_DIR}}")
    print(f"Results will be saved to: {{ft.RESULT_DIR}}")
    print(f"Loading transfer weights from: {{TRANSFER_WEIGHTS_PATH}}")

    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: create_fusion_model_se(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        transfer_weights_path=TRANSFER_WEIGHTS_PATH,
    )

    print("Monkey patching FocalLoss with FocalLossLabelSmoothing (smoothing=0.1)...")
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)
    ft.main()


if __name__ == '__main__':
    main()
"""


def main():
    script_dir = Path(__file__).resolve().parent
    for variant, cfg in VARIANTS.items():
        path = script_dir / f"run_samm_fusion_train_se_reprofix_evm_{variant}.py"
        content = TEMPLATE.format(
            variant=variant,
            result_dir=f"fusion_result_samm_se_reprofix_evm_{variant}",
            rgb_dir=cfg["rgb_dir"],
            dyn_dir=cfg["dyn_dir"],
        )
        path.write_text(content, encoding="utf-8")
        print(f"Generated {path.name}")


if __name__ == "__main__":
    main()
