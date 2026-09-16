import os
import fusion_train as ft
from fusion_model_grid_cbam import create_fusion_model_with_grid_cbam
import fusion_dataset_apex

# --- Configuration ---
# You can change the result directory name here
ft.RESULT_DIR = 'fusion_result_asym_grid_cbam_apex'

# Path to the pre-trained weights (from JAFFE transfer learning)
# Adjust this path to point to your actual .pth file
TRANSFER_WEIGHTS_PATH = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\jaffe_train\jaffe_result\result_jaffe_resnet18\best_model.pth'

def main():
    print(f"Starting Fusion Training with Asymmetric Architecture (Grid(2x2)+CBAM for RGB, SPP for Dynamic) and Transfer Learning...")
    print(f"Using Apex Frame for RGB channel.")
    print(f"Results will be saved to: {ft.RESULT_DIR}")
    print(f"Loading transfer weights from: {TRANSFER_WEIGHTS_PATH}")

    # Override the default model factory in fusion_train
    # We use a lambda to inject the transfer_weights_path argument
    ft.create_fusion_model = lambda num_classes, dropout_p, pretrained: \
        create_fusion_model_with_grid_cbam(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH
        )
    
    # Monkey patch the Dataset class in fusion_train module
    # This forces fusion_train to use our new Dataset class that loads Apex frames
    print("Monkey patching CASME2FusionDataset with CASME2FusionDatasetApex...")
    ft.CASME2FusionDataset = fusion_dataset_apex.CASME2FusionDatasetApex

    # Run the main training loop
    ft.main()

if __name__ == '__main__':
    main()
