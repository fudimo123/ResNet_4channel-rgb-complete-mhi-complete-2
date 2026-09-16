import os
import fusion_train as ft
from fusion_model_spp import create_fusion_model as create_spp_fusion_model

ft.create_fusion_model = create_spp_fusion_model
ft.RESULT_DIR = 'fusion_result1'

if __name__ == '__main__':
    os.environ['STAGE_NAME'] = 'single_run'
    os.environ['USE_EMA'] = '1'
    os.environ['TTA_VIEWS'] = '2'
    os.environ['FREEZE_EPOCHS'] = '8'
    os.environ['LR_BACKBONE_MULT'] = '0.1'
    ft.main()
