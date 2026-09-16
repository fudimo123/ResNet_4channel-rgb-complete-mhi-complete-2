import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

class CASME2FusionDataset(Dataset):
    """
    Dataset class for late fusion training that loads both preprocessed RGB and Dynamic images
    """

    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        subject = sample['subject']
        sequence_name = sample['video_name']
        aligned_path = sample['aligned_path']
        image_files = sample['image_files']
        num_frames = sample['num_frames']

        if num_frames == 0:
            print(f"No frames found for {sequence_name}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        # --- Load RGB Image ---
        # Randomly select a frame from the preprocessed sequence (typically the apex region is best, but random is okay for now)
        selected_frame_file = random.choice(image_files)
        rgb_image_path = os.path.join(aligned_path, selected_frame_file)

        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"RGB image file not found: {rgb_image_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        # --- Load Dynamic Image ---
        try:
            dynamic_filename = f"s{subject}_{sequence_name}.jpg"
            dynamic_image_path = os.path.join(self.dynamic_image_dir, subject, dynamic_filename)

            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
            else:
                print(f"Dynamic image not found: {dynamic_image_path}")
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        except Exception as e:
            print(f"Error loading dynamic image for {sequence_name}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)

        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)

        return rgb_image, dynamic_image, label
