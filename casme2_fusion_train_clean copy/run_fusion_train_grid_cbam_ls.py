import os
import fusion_train as ft
from fusion_model_grid_cbam import create_fusion_model_with_grid_cbam
from focal_loss_ls import FocalLossLabelSmoothing

# --- Configuration ---
# You can change the result directory name here
ft.RESULT_DIR = 'fusion_result_asym_grid_cbam_ls'

# Path to the pre-trained weights (from JAFFE transfer learning)
# Use relative path to support both local and cloud environments
# Assuming directory structure:
# ProjectRoot/
#   casme2_fusion_train2_3class copy/ (Current Dir)
#   casme_rgb_train/jaffe/...
TRANSFER_WEIGHTS_PATH = os.path.join('..', 'casme_rgb_train', 'jaffe', 'resnet18_jaffe_final_model_for_transfer.pth')

def main():
    print(f"Starting Fusion Training with Asymmetric Architecture (Grid(2x2)+CBAM for RGB, SPP for Dynamic) and Transfer Learning...")
    print(f"Strategy: Random Frame RGB + Label Smoothing (0.1)")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

    # Override the default model factory in fusion_train
    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_with_grid_cbam(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )
    
    # Monkey patch the FocalLoss class in fusion_train module
    # This forces fusion_train to use our new FocalLoss with Label Smoothing
    print("Monkey patching FocalLoss with FocalLossLabelSmoothing (smoothing=0.1)...")
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)

    # Run the main training loop
    ft.main()

if __name__ == '__main__':
    main()
