import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class SMICFusionDataset(Dataset):
    """
    Dataset class for late fusion training that loads both RGB and Dynamic images for SMIC dataset
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './dynamic_data'
        
        # Initialize MTCNN for face detection (for RGB images)
        self.mtcnn = MTCNN(keep_all=False, device='cpu')
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_face_roi(self, image):
        """
        Extract face ROI using MTCNN, similar to RGB training
        """
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Add some margin around the face
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)
            
            # Crop the face region
            face_image = image.crop((x1, y1, x2, y2))
            return face_image
        else:
            # If no face detected, return the original image
            return image
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        subject = sample['subject']
        sequence = sample['sequence']
        image_files = sample['image_files']
        
        if not image_files:
            print(f"No image files found for {sequence_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load RGB Image ---
        # Randomly select a frame from the sequence
        selected_frame_file = random.choice(image_files)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_image = self._extract_face_roi(rgb_image)
        except FileNotFoundError:
            print(f"RGB image file not found: {rgb_image_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load Dynamic Image ---
        try:
            # Generate dynamic image filename based on subject and sequence
            dynamic_filename = f"s{subject}_{sequence}.jpg"
            subject_dir = f"s{subject}"
            dynamic_image_path = os.path.join(self.dynamic_image_dir, subject_dir, dynamic_filename)
            
            # Load dynamic image
            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
                dynamic_image = self._extract_face_roi(dynamic_image)
            else:
                print(f"Dynamic image not found: {dynamic_image_path}")
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading dynamic image for {sequence_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)
        
        return rgb_image, dynamic_image, label