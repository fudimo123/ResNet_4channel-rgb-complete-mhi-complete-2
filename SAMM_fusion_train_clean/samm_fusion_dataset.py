import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class SAMMFusionDataset(Dataset):
    """
    Dataset class for SAMM late fusion training that loads pre-aligned RGB and Dynamic images.
    MTCNN is removed as data is already cropped.
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './samm_dynamic_data_clean'
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        aligned_path = sample['aligned_path']
        subject = sample['subject'] # String, e.g. '006'
        video_name = sample['video_name'] # e.g. '006_1_2'
        image_files = sample['image_files']
        
        if not image_files:
            print(f"No image files found for {aligned_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load RGB Image ---
        selected_frame_file = random.choice(image_files)
        rgb_image_path = os.path.join(aligned_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading RGB: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
            
        # --- Load Dynamic Image ---
        # Convention: dynamic_image_dir / 006 / s006_006_1_2.jpg
        dynamic_filename = f"s{subject}_{video_name}.jpg"
        dynamic_image_path = os.path.join(self.dynamic_image_dir, subject, dynamic_filename)
        
        try:
            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
            else:
                print(f"Dynamic image not found: {dynamic_image_path}")
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        except Exception as e:
            print(f"Error loading Dynamic: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
            
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)
            
        return rgb_image, dynamic_image, label
