import os
import samm_fusion_train as ft
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing

# --- Configuration ---
# You can change the result directory name here
# Note: ft.RESULT_DIR was originally set to os.path.join(SCRIPT_DIR, 'fusion_result') in samm_fusion_train.py
# We override it to a specific directory for this experiment.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_samm_se_reprofix_evm')
ft.ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'samm_aligned_rgb_evm')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'samm_dynamic_data_evm')

# Path to the pre-trained weights
# As per user instruction and reference architecture, we switch back to ImageNet weights (None)
TRANSFER_WEIGHTS_PATH = None

def main():
    # Ensure result directory exists
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)

    print(f"Starting SAMM Fusion Training with SE-Fusion (Adaptive Weighting, reprofix + EVM-only)...")
    print(f"Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print(f"Strategy: EVM RGB/Dynamic inputs + Train Random Frame RGB + Val/Test Middle Frame RGB + Label Smoothing (0.1)")
    print(f"EVM RGB dir: {ft.ALIGNED_RGB_DIR}")
    print(f"EVM Dynamic dir: {ft.DYNAMIC_IMAGE_DIR}")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

    # Override the default model factory in samm_fusion_train
    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_se(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )
    
    # Monkey patch the FocalLoss class in samm_fusion_train module
    print("Monkey patching FocalLoss with FocalLossLabelSmoothing (smoothing=0.1)...")
    ft.FocalLoss = lambda gamma: FocalLossLabelSmoothing(gamma=gamma, smoothing=0.1)

    # Run the main training loop
    ft.main()

if __name__ == '__main__':
    main()
