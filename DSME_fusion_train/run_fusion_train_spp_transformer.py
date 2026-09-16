# This runner reuses fusion_train.py's pipeline but swaps in the SPP-Token Transformer fusion model
import fusion_train as ft
from fusion_model_spp_transformer import create_fusion_model as create_model_spp_transformer

# Override factory and result dir
ft.create_fusion_model = create_model_spp_transformer
ft.RESULT_DIR = 'fusion_result_spp_transformer'

if __name__ == '__main__':
    ft.main()