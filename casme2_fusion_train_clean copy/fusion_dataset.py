import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class CASME2FusionDataset(Dataset):
    """
    Dataset class for CASME2 late fusion training that loads offline aligned RGB
    sequences and precomputed dynamic images.
    """

    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None, frame_selection='random'):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './dynamic_data'
        self.frame_selection = frame_selection

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        aligned_path = sample['aligned_path']
        subject = sample['subject']
        subject_num = sample['subject_num']
        sequence = sample['sequence']
        image_files = sample['image_files']

        if not image_files:
            print(f"No image files found for {aligned_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        if self.frame_selection == 'middle':
            selected_frame_file = image_files[len(image_files) // 2]
        else:
            selected_frame_file = random.choice(image_files)

        rgb_image_path = os.path.join(aligned_path, selected_frame_file)
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading RGB image {rgb_image_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        dynamic_filename = f"s{subject_num}_{sequence}.jpg"
        dynamic_image_path = os.path.join(self.dynamic_image_dir, subject, dynamic_filename)
        try:
            if not os.path.exists(dynamic_image_path):
                print(f"Dynamic image not found: {dynamic_image_path}")
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
            dynamic_image = Image.open(dynamic_image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading dynamic image {dynamic_image_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1

        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)

        return rgb_image, dynamic_image, label
