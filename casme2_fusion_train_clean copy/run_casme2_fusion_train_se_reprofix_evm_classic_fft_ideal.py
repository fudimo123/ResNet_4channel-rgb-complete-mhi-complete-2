import os
import fusion_train as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_casme2_se_reprofix_evm_classic_fft_ideal')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'casme2_aligned_rgb_evm_classic_fft_ideal')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'casme2_dynamic_data_evm_classic_fft_ideal')

TRANSFER_WEIGHTS_PATH = None


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print("Starting CASME2 Fusion Training with SE-Fusion (reprofix + classic_fft_ideal)...")
    print("Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print("Strategy: classic_fft_ideal RGB/Dynamic inputs + Train Random Frame RGB + Val/Test Middle Frame RGB + Label Smoothing (0.1)")
    print(f"RGB dir: {ft.ALIGNED_RGB_DIR}")
    print(f"Dynamic dir: {ft.DYNAMIC_IMAGE_DIR}")
    print(f"Results will be saved to: {ft.RESULT_DIR}")

    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: create_fusion_model_se(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        transfer_weights_path=TRANSFER_WEIGHTS_PATH,
    )
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)
    ft.main()


if __name__ == '__main__':
    main()
