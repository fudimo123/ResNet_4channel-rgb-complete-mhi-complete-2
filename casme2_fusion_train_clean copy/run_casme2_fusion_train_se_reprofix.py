import os

import fusion_train as ft
from focal_loss_ls import FocalLossLabelSmoothing
from fusion_model_se import create_fusion_model_se


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_casme2_se_reprofix')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'casme2_aligned_rgb')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'dynamic_data')

TRANSFER_WEIGHTS_PATH = None


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print("Starting CASME2 Fusion Training with SE-Fusion (reprofix baseline)...")
    print("Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print("Strategy: offline aligned RGB + original dynamic image + Label Smoothing (0.1)")
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
