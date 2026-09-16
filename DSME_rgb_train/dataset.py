import os
import random
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import torch

class DSMERGBDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.mtcnn = MTCNN(keep_all=False, device='cpu')

    def __len__(self):
        return len(self.samples)

    def _extract_face_roi(self, image):
        try:
            boxes, _ = self.mtcnn.detect(image)
            if boxes is not None:
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                margin = 20
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.width, x2 + margin)
                y2 = min(image.height, y2 + margin)
                
                face_image = image.crop((x1, y1, x2, y2))
                return face_image
            else:
                return image
        except Exception as e:
            # print(f"MTCNN Error: {e}")
            return image

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            print(f"Directory not found: {sequence_path}")
            return torch.zeros(3, 224, 224), -1

        # DSME_pic structure: 001/fear/1/img1.jpg
        # Just pick a random frame from the folder
        valid_frames = [f for f in frame_files if f.lower().endswith(('.jpg', '.png'))]
        
        if not valid_frames:
            print(f"No valid frames found for {sequence_path}")
            return torch.zeros(3, 224, 224), -1
        
        selected_frame_file = random.choice(valid_frames)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            # Apply MTCNN cropping
            rgb_image = self._extract_face_roi(rgb_image)
        except Exception as e:
            print(f"Error loading RGB: {e}")
            return torch.zeros(3, 224, 224), -1
            
        if self.transform:
            rgb_image = self.transform(rgb_image)
            
        return rgb_image, label
