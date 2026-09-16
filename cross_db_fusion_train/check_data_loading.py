import os
import random
from combined_data_parser import CombinedDataParser
from fusion_dataset import CrossDBFusionDataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

def test_data_loading():
    print("=== Testing Cross-Database Data Loading ===")
    
    # 1. Initialize Parser
    print("\n1. Initializing CombinedDataParser...")
    try:
        parser = CombinedDataParser()
        samples = parser.get_samples()
        print(f"Total samples found: {len(samples)}")
        
        # Check counts per dataset
        casme2_count = sum(1 for s in samples if s['dataset'] == 'casme2')
        samm_count = sum(1 for s in samples if s['dataset'] == 'samm')
        smic_count = sum(1 for s in samples if s['dataset'] == 'smic')
        print(f"  - CASME II: {casme2_count}")
        print(f"  - SAMM: {samm_count}")
        print(f"  - SMIC: {smic_count}")
        
    except Exception as e:
        print(f"ERROR initializing parser: {e}")
        return

    # 2. Initialize Dataset
    print("\n2. Initializing CrossDBFusionDataset...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = CrossDBFusionDataset(samples, rgb_transform=transform, dynamic_transform=transform)
    
    # 3. Random Sampling Test
    print("\n3. Testing Random Samples Loading (RGB + Dynamic)...")
    
    # Pick random samples from each dataset to test
    datasets_to_test = ['casme2', 'samm', 'smic']
    
    for ds_name in datasets_to_test:
        print(f"\n--- Testing {ds_name.upper()} Sample ---")
        ds_samples = [s for s in samples if s['dataset'] == ds_name]
        if not ds_samples:
            print(f"No samples for {ds_name}!")
            continue
            
        # Try up to 3 samples
        for i in range(3):
            sample = random.choice(ds_samples)
            print(f"Sample {i+1}: Subject={sample['subject']}, Path={os.path.basename(sample['raw_video_path'])}")
            
            # Find index in main dataset
            idx = samples.index(sample)
            
            try:
                rgb, dyn, label = dataset[idx]
                
                # Check if data is valid (not dummy zeros)
                is_rgb_dummy = torch.all(rgb == 0)
                is_dyn_dummy = torch.all(dyn == 0)
                
                if is_rgb_dummy:
                    print(f"  [FAIL] RGB Image Load Failed (Dummy Tensor Returned)")
                else:
                    print(f"  [PASS] RGB Image Loaded: {rgb.shape}")
                    
                if is_dyn_dummy:
                    print(f"  [FAIL] Dynamic Image Load Failed (Dummy Tensor Returned)")
                    # Try to debug why
                    # Call internal method to see path
                    # dataset._get_dynamic_image(sample) -> we can't easily hook into this without modifying class
                    # But we can inspect the sample properties that might affect it
                    print(f"    Expected Dynamic Sequence: {os.path.basename(sample['raw_video_path'])}")
                else:
                    print(f"  [PASS] Dynamic Image Loaded: {dyn.shape}")
                    
            except Exception as e:
                print(f"  [CRITICAL ERROR] Exception during loading: {e}")

if __name__ == "__main__":
    test_data_loading()
