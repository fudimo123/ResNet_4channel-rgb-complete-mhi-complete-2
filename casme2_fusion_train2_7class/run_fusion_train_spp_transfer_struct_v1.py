import os
try:
    from . import fusion_train_struct_v1 as ft
    from .fusion_model_spp_struct_v1 import create_fusion_model_with_transfer
except ImportError:
    import fusion_train_struct_v1 as ft
    from fusion_model_spp_struct_v1 import create_fusion_model_with_transfer

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

DEFAULT_TRANSFER_WEIGHTS = os.path.join(
    REPO_ROOT, 'casme_rgb_train', 'jaffe', 'resnet18_jaffe_final_model_for_transfer.pth'
)
TRANSFER_WEIGHTS_PATH = os.environ.get('TRANSFER_WEIGHTS_PATH', DEFAULT_TRANSFER_WEIGHTS)

def _factory(num_classes=7, dropout_p=0.5, pretrained=True):
    return create_fusion_model_with_transfer(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        transfer_weights_path=TRANSFER_WEIGHTS_PATH
    )

ft.create_fusion_model = _factory
ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result_spp_transfer_7class_struct_v1')
ft.DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'dynamic_data')
ft.ANNOTATION_FILE = os.path.join(REPO_ROOT, 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx')
ft.RAW_VIDEO_DIR = os.path.join(REPO_ROOT, 'data', 'CASME2_RAW_selected', 'CASME2_RAW_selected')

if __name__ == '__main__':
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)
    ft.main()

