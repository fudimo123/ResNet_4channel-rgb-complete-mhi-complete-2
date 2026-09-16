import os
import shutil
import glob

# Source Directories
CASME2_SRC = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class copy\dynamic_data'
SAMM_SRC = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\SAMM_fusion_train_3class copy\dynamic_data'
SMIC_SRC = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\smic_fusion_train_se\smic_dynamic_data2'

# Target Directory
TARGET_DIR = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\cross_db_fusion_train\merged_dynamic_data'

def merge_datasets():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    print(f"Target Directory: {TARGET_DIR}")
    
    # 1. Process CASME II
    print("\nProcessing CASME II...")
    count = 0
    # CASME2 structure: dynamic_data/subXX/filename.jpg
    for root, dirs, files in os.walk(CASME2_SRC):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                src_path = os.path.join(root, file)
                # New name: casme2_{filename}
                # Assuming filename is unique enough or contains subject info
                # CASME2 filenames are like: sub01_EP02_01f.jpg or similar
                # To be safe, we prefix with dataset name
                new_name = f"casme2_{file}"
                dst_path = os.path.join(TARGET_DIR, new_name)
                shutil.copy2(src_path, dst_path)
                count += 1
    print(f"Copied {count} files from CASME II.")

    # 2. Process SAMM
    print("\nProcessing SAMM...")
    count = 0
    # SAMM structure: dynamic_data/subXX/filename.jpg
    for root, dirs, files in os.walk(SAMM_SRC):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                src_path = os.path.join(root, file)
                new_name = f"samm_{file}"
                dst_path = os.path.join(TARGET_DIR, new_name)
                shutil.copy2(src_path, dst_path)
                count += 1
    print(f"Copied {count} files from SAMM.")

    # 3. Process SMIC
    print("\nProcessing SMIC...")
    count = 0
    # SMIC structure: smic_dynamic_data2/s1/filename.jpg
    for root, dirs, files in os.walk(SMIC_SRC):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                src_path = os.path.join(root, file)
                new_name = f"smic_{file}"
                dst_path = os.path.join(TARGET_DIR, new_name)
                shutil.copy2(src_path, dst_path)
                count += 1
    print(f"Copied {count} files from SMIC.")
    
    total = len(os.listdir(TARGET_DIR))
    print(f"\nTotal files in merged directory: {total}")

if __name__ == "__main__":
    merge_datasets()
