import os
import samm_fusion_train_decal_ratio as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_samm_se_reprofix_decal_ratio_ps0_sr1')

TRANSFER_WEIGHTS_PATH = None

ft.USE_STATIC_AUG = True
ft.STATIC_AUG_DIR = os.path.join(SCRIPT_DIR, 'samm_static_decalcomanie')
ft.STATIC_AUG_VARIANTS = ('L', 'R')
ft.STATIC_AUG_EMOTIONS = ('positive', 'surprise')
ft.STATIC_AUG_EMOTION_RATIOS = {
    'positive': 0.0,
    'surprise': 1.0,
}
ft.STATIC_AUG_DEFAULT_RATIO = 0.0


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print("Starting SAMM Fusion Training with SE-Fusion (Adaptive Weighting, reprofix + static ratio decalcomanie)...")
    print("Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print("Strategy: Train Random Frame RGB + static L/R ratio control (positive=0.0, surprise=1.0) + Val/Test Middle Frame RGB")
    print(f"Static augmentation dir: {ft.STATIC_AUG_DIR}")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_se(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )

    print("Monkey patching FocalLoss with FocalLossLabelSmoothing (smoothing=0.1)...")
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)

    ft.main()


if __name__ == '__main__':
    main()
