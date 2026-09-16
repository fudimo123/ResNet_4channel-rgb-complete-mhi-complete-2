import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN

class CASME2Dataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.mtcnn = MTCNN(keep_all=False, post_process=False, device='cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']

        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            print(f"Directory not found: {sequence_path}")
            return torch.zeros(3, 224, 224), -1

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
            return torch.zeros(3, 224, 224), -1

        selected_frame_file = random.choice(valid_frames)
        image_path = os.path.join(sequence_path, selected_frame_file)

        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            return torch.zeros(3, 224, 224), -1

        # --- ROI Extraction using MTCNN ---
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            box = boxes[0]
            # Add some padding to the bounding box
            x1, y1, x2, y2 = [max(0, int(c)) for c in box]
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.width, x2 + padding)
            y2 = min(image.height, y2 + padding)
            image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        return image, label