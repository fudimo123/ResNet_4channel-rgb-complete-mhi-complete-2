import os
try:
    from . import fusion_train as ft
    from .fusion_model_spp import create_fusion_model_with_transfer
except ImportError:
    import fusion_train as ft
    from fusion_model_spp import create_fusion_model_with_transfer

# Resolve paths relative to this script (no hard-coded absolute paths)
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Default relative location of JAFFE transfer weights within the repo
DEFAULT_TRANSFER_WEIGHTS = os.path.join(
    REPO_ROOT, 'casme_rgb_train', 'jaffe', 'resnet18_jaffe_final_model_for_transfer.pth'
)

# Allow overriding via environment variable TRANSFER_WEIGHTS_PATH
TRANSFER_WEIGHTS_PATH = os.environ.get('TRANSFER_WEIGHTS_PATH', DEFAULT_TRANSFER_WEIGHTS)

def _factory_with_levels(levels):
    def _factory(num_classes=3, dropout_p=0.5, pretrained=True):
        return create_fusion_model_with_transfer(
            num_classes=num_classes,
            dropout_p=dropout_p,
            pretrained=pretrained,
            transfer_weights_path=TRANSFER_WEIGHTS_PATH,
            spp_levels=levels
        )
    return _factory

ft.RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result1')

if __name__ == '__main__':
    if not os.path.exists(ft.RESULT_DIR):
        os.makedirs(ft.RESULT_DIR)
    final_metrics_path = os.path.join(ft.RESULT_DIR, 'final_metrics.txt')
    if os.path.exists(final_metrics_path):
        os.remove(final_metrics_path)

    os.environ['USE_EMA'] = '0'
    os.environ['TTA_VIEWS'] = '1'
    os.environ['FREEZE_EPOCHS'] = '0'
    os.environ['LR_BACKBONE_MULT'] = '1.0'
    os.environ['STAGE_NAME'] = 'stage1_gating_dropout'
    ft.create_fusion_model = _factory_with_levels((1, 2, 4))
    ft.main()

    os.environ['STAGE_NAME'] = 'stage2_spp_levels'
    ft.create_fusion_model = _factory_with_levels((1, 2, 3, 6))
    ft.main()

    os.environ['STAGE_NAME'] = 'stage3_freeze_layerwise_lr'
    os.environ['FREEZE_EPOCHS'] = '8'
    os.environ['LR_BACKBONE_MULT'] = '0.1'
    ft.create_fusion_model = _factory_with_levels((1, 2, 3, 6))
    ft.main()

    os.environ['STAGE_NAME'] = 'stage4_ema_tta'
    os.environ['USE_EMA'] = '1'
    os.environ['TTA_VIEWS'] = '2'
    ft.create_fusion_model = _factory_with_levels((1, 2, 3, 6))
    ft.main()
