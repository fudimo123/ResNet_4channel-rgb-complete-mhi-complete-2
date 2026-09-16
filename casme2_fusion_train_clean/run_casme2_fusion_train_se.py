import os
import fusion_train as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_casme2_se')

# Override data paths for cleaned data
ft.ANNOTATION_FILE = os.path.join(SCRIPT_DIR, 'casme2_aligned_rgb', 'casme2_clean_labels.csv')
ft.RAW_VIDEO_DIR = os.path.join(SCRIPT_DIR, 'casme2_aligned_rgb')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'casme2_dynamic_data_clean')

# The cleaned labels use the simplified emotion names
ft.EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}

TRANSFER_WEIGHTS_PATH = None

def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print(f"Starting CASME2 Fusion Training with SE-Fusion (Adaptive Weighting)...")
    print(f"Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print(f"Strategy: Random Frame RGB + Label Smoothing (0.1)")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

    # Override the default model factory in fusion_train
    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_se(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )

    # Monkey patch the FocalLoss class in fusion_train module
    print("Monkey patching FocalLoss with FocalLossLabelSmoothing (smoothing=0.1)...")
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)

    # Run the main training loop
    ft.main()

if __name__ == '__main__':
    main()
