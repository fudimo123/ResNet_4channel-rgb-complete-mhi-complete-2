import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class SAMMSingleRGBDataset(Dataset):
    def __init__(self, samples, transform=None, frame_selection="random", processed_rgb_root=None):
        self.samples = samples
        self.transform = transform
        self.frame_selection = frame_selection
        self.processed_rgb_root = processed_rgb_root

    def __len__(self):
        return len(self.samples)

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

        if processed_dir is None and self.processed_rgb_root is not None:
            processed_dir = os.path.join(self.processed_rgb_root, sample["subject"], sample["sequence"])
        elif processed_dir is not None and self.processed_rgb_root is not None and not os.path.isabs(processed_dir):
            processed_dir = os.path.join(self.processed_rgb_root, processed_dir)

        valid_frames = self._collect_processed_frames(processed_dir) if processed_dir is not None else []
        if not valid_frames:
            return torch.zeros(3, 224, 224), -1

        selected_frame = self._select_frame_file(valid_frames)
        image_path = os.path.join(processed_dir, selected_frame)

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return torch.zeros(3, 224, 224), -1

        if self.transform is not None:
            image = self.transform(image)
        return image, label
