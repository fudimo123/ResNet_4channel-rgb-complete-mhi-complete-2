import os
import torch
import sys

def check_environment():
    print("=== Environment & Path Check Tool ===")
    
    # 1. Define the relative path we intend to use
    relative_path = os.path.join('..', 'casme_rgb_train', 'jaffe', 'resnet18_jaffe_final_model_for_transfer.pth')
    
    # 2. Resolve to absolute path for clarity
    current_dir = os.getcwd()
    abs_path = os.path.abspath(relative_path)
    
    print(f"Current Directory: {current_dir}")
    print(f"Target Relative Path: {relative_path}")
    print(f"Resolved Absolute Path: {abs_path}")
    
    # 3. Check if file exists
    if os.path.exists(abs_path):
        print("\n[PASS] File exists on disk.")
    else:
        print("\n[FAIL] File NOT FOUND at the resolved path!")
        print("Suggestion: Check if 'casme_rgb_train' folder is uploaded and is a sibling of the current folder.")
        # List sibling directories for debugging
        parent_dir = os.path.dirname(current_dir)
        print(f"\nContents of parent directory ({parent_dir}):")
        try:
            for item in os.listdir(parent_dir):
                print(f" - {item}")
        except Exception as e:
            print(f" (Cannot list directory: {e})")
        return

    # 4. Try loading the weights with torch
    print("\nAttempting to load weights with torch.load...")
    try:
        checkpoint = torch.load(abs_path, map_location='cpu')
        print("[PASS] torch.load successful.")
        
        # 5. Check content structure
        if isinstance(checkpoint, dict):
            print(f"[INFO] Checkpoint is a dict with keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                keys = list(checkpoint['state_dict'].keys())
                print(f"[INFO] First 5 layer keys in state_dict: {keys[:5]}")
            else:
                # Maybe it's a direct state_dict
                keys = list(checkpoint.keys())
                print(f"[INFO] First 5 keys (assuming state_dict): {keys[:5]}")
        else:
            print("[INFO] Checkpoint is not a dict (maybe raw state_dict).")
            
        print("\n=== CONGRATULATIONS: Environment Check Passed! ===")
        print("You can now safely run the training scripts.")
        
    except Exception as e:
        print(f"\n[FAIL] Error loading weights: {e}")
        print("The file exists but might be corrupted or incompatible.")

if __name__ == "__main__":
    check_environment()
