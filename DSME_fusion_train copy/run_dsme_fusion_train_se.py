import os
import fusion_train as ft
from fusion_model_se import create_fusion_model_se
# from focal_loss_ls import FocalLossLabelSmoothing # Removed Label Smoothing

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_dsme_se')
TRANSFER_WEIGHTS_PATH = None

def main():
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print(f"Starting DSME Fusion Training with SE-Fusion (Adaptive Weighting)...")
    print(f"Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print(f"Strategy: Random Frame RGB (Standard Focal Loss, No Label Smoothing)")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    
    # Override model factory
    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_se(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )
    
    # We do NOT monkey patch FocalLoss here, so it uses the standard FocalLoss from fusion_train (which imports from focal_loss.py)
    # ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)

    ft.main()

if __name__ == '__main__':
    main()
