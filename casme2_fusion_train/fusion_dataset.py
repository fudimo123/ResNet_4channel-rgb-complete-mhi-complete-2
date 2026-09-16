import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import sys
sys.path.append('../casme2_mhi_train')
from mhi_generator import MHIGenerator


class CASME2FusionDataset(Dataset):
    """
    Dataset class for late fusion training that loads both RGB and MHI images
    """
    
    def __init__(self, samples, rgb_transform=None, mhi_transform=None, save_mhi=False, mhi_save_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.mhi_transform = mhi_transform
        self.mhi_generator = MHIGenerator(duration=40)
        self.save_mhi = save_mhi
        self.mhi_save_dir = mhi_save_dir
        
        # Initialize MTCNN for face detection (for RGB images)
        self.mtcnn = MTCNN(keep_all=False, device='cpu')
        
        # Create save directory if needed
        if self.save_mhi and self.mhi_save_dir:
            os.makedirs(self.mhi_save_dir, exist_ok=True)
    
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
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # Find valid frames within onset-offset range
        valid_frames = []
        for f in frame_files:
            if f.startswith('img') and f.endswith('.jpg'):
                try:
                    frame_num = int(f[3:-4])
                    if onset <= frame_num <= offset:
                        valid_frames.append(f)
                except ValueError:
                    continue
        
        if not valid_frames:
            print(f"No valid frames found for {sequence_path} between {onset} and {offset}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # --- Load RGB Image ---
        selected_frame_file = random.choice(valid_frames)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_image = self._extract_face_roi(rgb_image)
        except FileNotFoundError:
            print(f"RGB image file not found: {rgb_image_path}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # --- Generate MHI Image ---
        try:
            mhi_image = self.mhi_generator.generate_mhi_from_sequence(sequence_path, onset, offset)
            
            # Save MHI image if requested
            if self.save_mhi and self.mhi_save_dir:
                # Create subject directory
                subject_dir = os.path.join(self.mhi_save_dir, f"sub{subject:02d}" if isinstance(subject, int) else str(subject))
                os.makedirs(subject_dir, exist_ok=True)
                
                # Generate filename based on sequence info
                sequence_name = os.path.basename(sequence_path)
                mhi_filename = f"{sequence_name}_onset{onset}_offset{offset}_mhi.png"
                mhi_path = os.path.join(subject_dir, mhi_filename)
                
                # Save MHI image
                mhi_image.save(mhi_path)
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error generating MHI for {sequence_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.mhi_transform:
            mhi_image = self.mhi_transform(mhi_image)
        
        return rgb_image, mhi_image, label