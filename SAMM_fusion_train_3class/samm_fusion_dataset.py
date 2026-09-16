import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN

class SAMMFusionDataset(Dataset):
    """
    Dataset class for SAMM late fusion training.
    """
    
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './dynamic_data'
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=False, device='cpu')
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_face_roi(self, image):
        # ... (Same as CASME2) ...
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
            print(f"MTCNN Error: {e}")
            return image
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']
        subject = sample['subject'] # String, e.g. '006'
        video_name = sample['video_name'] # e.g. '006_1_2'
        
        # --- Load RGB Image ---
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            print(f"Directory not found: {sequence_path}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        # Filter frames by onset/offset
        # SAMM frames are typically 006_0180.jpg etc.
        # Format: {Subject}_{FrameNum}.jpg
        valid_frames = []
        for f in frame_files:
            if f.lower().endswith('.jpg'):
                # Extract frame number
                # Filename: 006_0180.jpg -> 0180
                try:
                    # Assuming format Subject_Frame.jpg
                    parts = f.split('_')
                    if len(parts) >= 2:
                        frame_str = parts[-1].split('.')[0]
                        frame_num = int(frame_str)
                        if onset <= frame_num <= offset:
                            valid_frames.append(f)
                except ValueError:
                    continue
        
        if not valid_frames:
            # Fallback: just take all frames if parsing fails (shouldn't happen if assumption holds)
            # Or print error
            # print(f"No valid frames found for {sequence_path} ({onset}-{offset})")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        
        selected_frame_file = random.choice(valid_frames)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_image = self._extract_face_roi(rgb_image)
        except Exception as e:
            print(f"Error loading RGB: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
            
        # --- Load Dynamic Image ---
        # Convention: dynamic_image_dir / sub006 / s006_006_1_2.jpg
        subject_dir = f"sub{subject}" # subject is '006' -> 'sub006'
        dynamic_filename = f"s{subject}_{video_name}.jpg"
        dynamic_image_path = os.path.join(self.dynamic_image_dir, subject_dir, dynamic_filename)
        
        try:
            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
                dynamic_image = self._extract_face_roi(dynamic_image)
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
