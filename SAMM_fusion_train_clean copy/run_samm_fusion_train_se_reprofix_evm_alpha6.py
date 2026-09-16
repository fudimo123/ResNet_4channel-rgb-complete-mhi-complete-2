import os
import samm_fusion_train as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_samm_se_reprofix_evm_alpha6')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'samm_aligned_rgb_evm_alpha6')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'samm_dynamic_data_evm_alpha6')

TRANSFER_WEIGHTS_PATH = None


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print("Starting SAMM Fusion Training with SE-Fusion (Adaptive Weighting, reprofix + alpha6)...")
    print("Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print("Strategy: alpha6 RGB/Dynamic inputs + Train Random Frame RGB + Val/Test Middle Frame RGB + Label Smoothing (0.1)")
    print(f"RGB dir: {ft.ALIGNED_RGB_DIR}")
    print(f"Dynamic dir: {ft.DYNAMIC_IMAGE_DIR}")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

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
