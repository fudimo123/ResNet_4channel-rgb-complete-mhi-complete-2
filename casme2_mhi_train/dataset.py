import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from mhi_generator import MHIGenerator

class CASME2Dataset(Dataset):
    def __init__(self, samples, transform=None, save_mhi=False, mhi_save_dir=None):
        self.samples = samples
        self.transform = transform
        self.mhi_generator = MHIGenerator(duration=40)  # Updated duration
        self.save_mhi = save_mhi
        self.mhi_save_dir = mhi_save_dir
        
        # Create save directory if needed
        if self.save_mhi and self.mhi_save_dir:
            os.makedirs(self.mhi_save_dir, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']
        subject = sample.get('subject', 'unknown')

        try:
            # Generate MHI from the video sequence
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
            return torch.zeros(1, 224, 224), -1

        if self.transform:
            mhi_image = self.transform(mhi_image)

        return mhi_image, label