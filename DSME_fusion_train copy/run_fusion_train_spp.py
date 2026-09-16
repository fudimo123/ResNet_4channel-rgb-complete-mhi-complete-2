import os
import fusion_train as ft
from fusion_model_spp import create_fusion_model as create_spp_fusion_model

# Override the model factory to use SPP version
ft.create_fusion_model = create_spp_fusion_model

# Set a distinct result directory to avoid overwriting existing results
ft.RESULT_DIR = 'fusion_result2'

if __name__ == '__main__':
    ft.main()