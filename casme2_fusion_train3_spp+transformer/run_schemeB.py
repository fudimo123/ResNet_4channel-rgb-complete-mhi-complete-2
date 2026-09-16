import os
import sys

# Ensure module search path includes current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import Scheme B factory
from fusion_model_schemeB import create_fusion_model_schemeB

# Import original training script as a module
import fusion_train_optimized as ft

# Override the model factory to use Scheme B
ft.create_fusion_model = create_fusion_model_schemeB

# Set a separate result directory to avoid overwriting Scheme A results
ft.RESULT_DIR = 'fusion_result_schemeB'

if __name__ == '__main__':
    ft.main()