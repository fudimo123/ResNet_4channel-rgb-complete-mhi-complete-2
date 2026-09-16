import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class CrossDBFusionDataset(Dataset):
    """
    Dataset class for cross-database fusion training.
    Loads both RGB and Dynamic images from CASME II, SAMM, and SMIC.
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir # Kept for signature compatibility but not used

    def __len__(self):
        return len(self.samples)

    def _get_rgb_image(self, sample):
        # We now use 'aligned_path' because the data is already cleaned and cropped
        sequence_path = sample['aligned_path']
        dataset = sample['dataset']

        try:
            # All our clean data uses .jpg
            frame_files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.jpg')])
            if not frame_files:
                 return None
            
            # Since data is pre-sampled (e.g. 16 or 32 frames), we just pick a random one
            selected_frame_file = random.choice(frame_files)

            rgb_image_path = os.path.join(sequence_path, selected_frame_file)   
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            return rgb_image

        except Exception as e:
            print(f"Error loading RGB for {dataset} sample {sequence_path}: {e}")
            return None

    def _get_dynamic_image(self, sample):
        dataset = sample['dataset']
        # Original subject ID (e.g., '01', '006', 's1')
        orig_sub = str(sample['original_subject'])
        sequence_name = sample['video_name'] # From parser compatibility

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dynamic_image_path = None

        if dataset == 'casme2':
            dynamic_dir = os.path.join(base_dir, 'casme2_fusion_train_clean', 'casme2_dynamic_data_clean')
            dynamic_filename = f"s{orig_sub}_{sequence_name}.jpg"
            dynamic_image_path = os.path.join(dynamic_dir, orig_sub, dynamic_filename)

        elif dataset == 'samm':
            dynamic_dir = os.path.join(base_dir, 'SAMM_fusion_train_clean', 'samm_dynamic_data_clean')
            dynamic_filename = f"s{orig_sub}_{sequence_name}.jpg"
            dynamic_image_path = os.path.join(dynamic_dir, orig_sub, dynamic_filename)

        elif dataset == 'smic':
            dynamic_dir = os.path.join(base_dir, 'smic_fusion_train_clean', 'smic_dynamic_data_clean')
            # SMIC orig_sub is already 's1', sequence is like 's1_ne_01'
            # In SMIC preprocess, we saved it as f"{subject}_{seq_name}.jpg", so "s1_s1_ne_01.jpg"
            dynamic_filename = f"{orig_sub}_{sequence_name}.jpg"
            dynamic_image_path = os.path.join(dynamic_dir, orig_sub, dynamic_filename)

        if dynamic_image_path and os.path.exists(dynamic_image_path):
            try:
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')       
                return dynamic_image
            except Exception as e:
                print(f"Error opening dynamic image {dynamic_image_path}: {e}")     
                return None
        else:
            print(f"Dynamic image not found: {dynamic_image_path}")
            return None

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']

        # --- Load Pre-aligned RGB Image ---
        rgb_image = self._get_rgb_image(sample)
        if rgb_image is None:
            # Return dummy if failed
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1       

        # --- Load Pre-generated Dynamic Image ---
        dynamic_image = self._get_dynamic_image(sample)
        if dynamic_image is None:
             # Return dummy
             return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1      

        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)

        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)

        return rgb_image, dynamic_image, label
