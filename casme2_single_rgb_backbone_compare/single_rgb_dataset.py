import os
import random

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torch.utils.data import Dataset


class CASME2SingleRGBDataset(Dataset):
    """
    Single-channel RGB dataset for CASME2.

    The dataset mirrors the fusion pipeline's frame selection policy:
    - train: random frame within onset-offset
    - val/test: middle frame within onset-offset
    """

    def __init__(self, samples, transform=None, frame_selection="random", face_device=None, processed_rgb_root=None):
        self.samples = samples
        self.transform = transform
        self.frame_selection = frame_selection
        self.processed_rgb_root = processed_rgb_root
        self.mtcnn = None
        if processed_rgb_root is None:
            if face_device is None:
                face_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mtcnn = MTCNN(keep_all=False, post_process=False, device=face_device)

    def __len__(self):
        return len(self.samples)

    def _extract_face_roi(self, image):
        if self.mtcnn is None:
            return image
        boxes, _ = self.mtcnn.detect(image)
        if boxes is None:
            return image

        box = boxes[0]
        x1, y1, x2, y2 = [int(v) for v in box]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)
        return image.crop((x1, y1, x2, y2))

    def _collect_valid_frames(self, sequence_path, onset, offset):
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            return []

        valid_frames = []
        for file_name in frame_files:
            if not (file_name.startswith("img") and file_name.endswith(".jpg")):
                continue
            try:
                frame_num = int(file_name[3:-4])
            except ValueError:
                continue
            if onset <= frame_num <= offset:
                valid_frames.append(file_name)
        return valid_frames

    def _collect_processed_frames(self, processed_dir):
        try:
            return sorted([name for name in os.listdir(processed_dir) if name.lower().endswith(".jpg")])
        except FileNotFoundError:
            return []

    def _select_frame_file(self, valid_frames):
        if self.frame_selection == "middle":
            return valid_frames[len(valid_frames) // 2]
        return random.choice(valid_frames)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        processed_dir = sample.get("processed_rgb_path")
        if processed_dir is not None:
            processed_dir = processed_dir.replace("\\", os.sep).replace("/", os.sep)

        if processed_dir is None:
            if self.processed_rgb_root is not None:
                processed_dir = os.path.join(
                    self.processed_rgb_root,
                    f"sub{int(sample['subject']):02d}",
                    sample["sequence"],
                )
        elif self.processed_rgb_root is not None and not os.path.isabs(processed_dir):
            processed_dir = os.path.join(self.processed_rgb_root, processed_dir)

        if processed_dir is not None:
            valid_frames = self._collect_processed_frames(processed_dir)
            frame_dir = processed_dir
        else:
            sequence_path = sample["raw_video_path"]
            onset = sample["onset"]
            offset = sample["offset"]
            valid_frames = self._collect_valid_frames(sequence_path, onset, offset)
            frame_dir = sequence_path

        if not valid_frames:
            return torch.zeros(3, 224, 224), -1

        selected_frame = self._select_frame_file(valid_frames)
        image_path = os.path.join(frame_dir, selected_frame)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return torch.zeros(3, 224, 224), -1

        image = self._extract_face_roi(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
