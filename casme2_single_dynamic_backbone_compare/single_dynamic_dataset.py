import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CASME2SingleDynamicDataset(Dataset):
    """
    Single-channel dynamic-image dataset for CASME2.

    Each sample corresponds to one offline dynamic image generated from the
    original onset-offset sequence using the legacy rank-pooling pipeline.
    """

    def __init__(self, samples, transform=None, processed_dynamic_root=None):
        self.samples = samples
        self.transform = transform
        self.processed_dynamic_root = processed_dynamic_root

    def __len__(self):
        return len(self.samples)

    def _resolve_dynamic_path(self, sample):
        dynamic_path = sample.get("processed_dynamic_path")
        if dynamic_path is not None:
            dynamic_path = dynamic_path.replace("\\", os.sep).replace("/", os.sep)

        if dynamic_path is None and self.processed_dynamic_root is not None:
            dynamic_path = os.path.join(
                f"sub{int(sample['subject']):02d}",
                f"s{int(sample['subject'])}_{sample['sequence']}.jpg",
            )

        if dynamic_path is None:
            return None

        if self.processed_dynamic_root is not None and not os.path.isabs(dynamic_path):
            dynamic_path = os.path.join(self.processed_dynamic_root, dynamic_path)

        return dynamic_path

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]
        dynamic_path = self._resolve_dynamic_path(sample)

        if dynamic_path is None or not os.path.exists(dynamic_path):
            return torch.zeros(3, 224, 224), -1

        try:
            image = Image.open(dynamic_path).convert("RGB")
        except (FileNotFoundError, OSError):
            return torch.zeros(3, 224, 224), -1

        if self.transform is not None:
            image = self.transform(image)
        return image, label
