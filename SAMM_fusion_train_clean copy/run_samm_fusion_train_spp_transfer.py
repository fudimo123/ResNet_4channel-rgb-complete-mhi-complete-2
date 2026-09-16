import os
try:
    from . import samm_fusion_train as ft
    from .fusion_model_spp import create_fusion_model_with_transfer
except ImportError:
    import samm_fusion_train as ft
    from fusion_model_spp import create_fusion_model_with_transfer

# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Location of JAFFE transfer weights
DEFAULT_TRANSFER_WEIGHTS = os.path.join(
    REPO_ROOT, 'casme_rgb_train', 'jaffe', 'resnet18_jaffe_final_model_for_transfer.pth'
)

TRANSFER_WEIGHTS_PATH = os.environ.get('TRANSFER_WEIGHTS_PATH', DEFAULT_TRANSFER_WEIGHTS)

# Override the model factory
def _factory(num_classes=3, dropout_p=0.5, pretrained=True):
    return create_fusion_model_with_transfer(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        transfer_weights_path=TRANSFER_WEIGHTS_PATH
    )

ft.create_fusion_model = _factory

# Result directory
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_spp_transfer')

if __name__ == '__main__':
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)
    
    print(f"Starting SAMM Training with SPP + Transfer Learning...")
    print(f"Transfer weights: {TRANSFER_WEIGHTS_PATH}")
    
    ft.main()
