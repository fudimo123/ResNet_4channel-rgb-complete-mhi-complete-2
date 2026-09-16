import os

import fusion_train as ft
from casme2_static_augmentation_ratio import append_augmented_static_samples
from focal_loss_ls import FocalLossLabelSmoothing
from fusion_model_se import create_fusion_model_se


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_casme2_se_reprofix_decal_ratio_ps05_sr1')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'casme2_aligned_rgb')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'dynamic_data')
ft.USE_STATIC_AUG = True
ft.STATIC_AUG_DIR = os.path.join(SCRIPT_DIR, 'casme2_static_decalcomanie')
ft.STATIC_AUG_VARIANTS = ('L', 'R')
ft.STATIC_AUG_EMOTIONS = ('positive', 'surprise')
ft.STATIC_AUG_KWARGS = {
    'emotion_variant_ratios': {'positive': 0.5, 'surprise': 1.0},
    'default_ratio': 0.0,
}
ft.APPEND_STATIC_AUG_SAMPLES = append_augmented_static_samples

TRANSFER_WEIGHTS_PATH = None


def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

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
