import os
import shutil

def setup_smic_workspace():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_smic_dir = os.path.join(base_dir, 'smic_fusion_train2')
    source_casme_dir = os.path.join(base_dir, 'casme2_fusion_train2_3class copy')
    target_dir = os.path.join(base_dir, 'smic_fusion_train_se')
    
    print(f"Setting up workspace at: {target_dir}")
    
    # Create target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print("Created target directory.")
    
    # 1. Copy base python files from SMIC source (to keep SMIC-specific parsers)
    # files_to_copy_from_smic = [
    #     'smic_data_parser.py', 
    #     'fusion_dataset.py', # Will check if this needs update
    #     'metrics.py',
    #     'early_stopping.py',
    #     'fusion_train.py'    # Will modify this later
    # ]
    
    # Actually, we should copy everything py from SMIC first, then overwrite with new models
    for item in os.listdir(source_smic_dir):
        s = os.path.join(source_smic_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.isfile(s) and item.endswith('.py'):
            shutil.copy2(s, d)
            print(f"Copied {item} from SMIC source.")
            
    # 2. Copy new model files from CASME2 source
    files_to_copy_from_casme = [
        'fusion_model_se.py',
        'fusion_model_grid_cbam.py',
        'focal_loss_ls.py'
    ]
    
    for item in files_to_copy_from_casme:
        s = os.path.join(source_casme_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.exists(s):
            shutil.copy2(s, d)
            print(f"Copied {item} from CASME2 source (New Models).")
        else:
            print(f"WARNING: {item} not found in CASME2 source!")

    # 3. Copy Dynamic Data (Recursively)
    # This might take a moment, but SMIC is small
    smic_data_src = os.path.join(source_smic_dir, 'smic_dynamic_data2')
    smic_data_dst = os.path.join(target_dir, 'smic_dynamic_data2')
    
    if os.path.exists(smic_data_src):
        if not os.path.exists(smic_data_dst):
            print("Copying dynamic data folder...")
            shutil.copytree(smic_data_src, smic_data_dst)
            print("Dynamic data copied.")
        else:
            print("Dynamic data folder already exists, skipping copy.")
    else:
        print("WARNING: Source dynamic data not found!")

    print("Workspace setup complete.")

if __name__ == "__main__":
    setup_smic_workspace()
