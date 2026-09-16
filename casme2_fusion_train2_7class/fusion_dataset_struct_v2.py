import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN


class CASME2FusionDataset(Dataset):
    def __init__(self, samples, rgb_transform=None, dynamic_transform=None, dynamic_image_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.dynamic_transform = dynamic_transform
        self.dynamic_image_dir = dynamic_image_dir or './dynamic_data'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(keep_all=False, device=device)

    def __len__(self):
        return len(self.samples)

    def _extract_face_roi(self, image):
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
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
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
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        selected_frame_file = random.choice(valid_frames)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_image = self._extract_face_roi(rgb_image)
        except FileNotFoundError:
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        try:
            sequence_name = os.path.basename(sequence_path)
            dynamic_filename = f"s{subject}_{sequence_name}.jpg"
            subject_dir = f"sub{subject:02d}"
            dynamic_image_path = os.path.join(self.dynamic_image_dir, subject_dir, dynamic_filename)
            if os.path.exists(dynamic_image_path):
                dynamic_image = Image.open(dynamic_image_path).convert('RGB')
                dynamic_image = self._extract_face_roi(dynamic_image)
            else:
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        except (FileNotFoundError, ValueError):
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), -1
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        if self.dynamic_transform:
            dynamic_image = self.dynamic_transform(dynamic_image)
        return rgb_image, dynamic_image, label

