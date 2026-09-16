import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class CASME2FusionDataset(Dataset):
    """
    Dataset class for late fusion training that loads both RGB and Dynamic images
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './dynamic_data'
        
        # Initialize MTCNN for face detection (for RGB images)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=device)
    
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
        onset = sample['onset']
        offset = sample['offset']
        subject = sample.get('subject', 'unknown')
        
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            print(f"Directory not found: {sequence_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Find valid frames within onset-offset range
        valid_frames = []
        for f in frame_files:
            if f.startswith('img') and f.endswith('.jpg'):
                try:
                    frame_num = int(f[3:-4])
                    if onset <= frame_num <= offset:
                        valid_frames.append((frame_num, f))
                except ValueError:
                    continue
        valid_frames.sort(key=lambda x: x[0])
        
        if not valid_frames:
            print(f"No valid frames found for {sequence_path} between {onset} and {offset}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # --- Load RGB Image ---
        mid_idx = len(valid_frames) // 2
        selected_frame_file = valid_frames[mid_idx][1]
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
            subject = sample['subject']
            sequence_name = sample.get('sequence', os.path.basename(sequence_path))
            subject_dir = f"sub{subject:03d}"
            # primary (canonical emotion)
            dynamic_filename = f"s{subject}_{sequence_name}.jpg"
            dynamic_image_path = os.path.join(self.dynamic_image_dir, subject_dir, dynamic_filename)
            # fallback (raw emotion from dataset folder)
            if not os.path.exists(dynamic_image_path) and 'raw_emotion' in sample and 'segment' in sample:
                alt_sequence = f"{sample['raw_emotion']}_{sample['segment']}"
                alt_filename = f"s{subject}_{alt_sequence}.jpg"
                alt_path = os.path.join(self.dynamic_image_dir, subject_dir, alt_filename)
                if os.path.exists(alt_path):
                    dynamic_image_path = alt_path
            
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