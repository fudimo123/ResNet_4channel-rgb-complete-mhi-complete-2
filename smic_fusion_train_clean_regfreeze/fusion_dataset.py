import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class SMICFusionDataset(Dataset):
    """
    Dataset class for late fusion training that loads pre-aligned RGB and Dynamic images
    for SMIC dataset. MTCNN is removed as data is already cropped.
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './smic_dynamic_data_clean'
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        aligned_path = sample['aligned_path']
        subject = sample['subject'] # already formatted like 's1'
        sequence = sample['sequence']
        image_files = sample['image_files']
        
        if not image_files:
            print(f"No image files found for {aligned_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load RGB Image ---
        # Randomly select a frame from the pre-aligned sequence
        selected_frame_file = random.choice(image_files)
        rgb_image_path = os.path.join(aligned_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"RGB image file not found: {rgb_image_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load Dynamic Image ---
        try:
            # Generate dynamic image filename based on subject and sequence
            dynamic_filename = f"{subject}_{sequence}.jpg"
            dynamic_image_path = os.path.join(self.dynamic_image_dir, subject, dynamic_filename)
            
            # Load dynamic image
            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
            else:
                print(f"Dynamic image not found: {dynamic_image_path}")
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading dynamic image for {sequence}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)
        
        return rgb_image, dynamic_image, label
