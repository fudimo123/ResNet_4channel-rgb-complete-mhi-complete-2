import os
from typing import List, Dict
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}

class JAFFEDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, s['label']

    @staticmethod
    def build_samples(root_dir: str, emotion_map: Dict[str, int]) -> List[Dict]:
        samples = []
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"JAFFE root_dir not found: {root_dir}")
        # Expect structure: root_dir/s1/anger/*.jpg, ... s10/surprise/*.jpg
        for subj_name in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subj_name)
            if not os.path.isdir(subj_path):
                continue
            if not subj_name.lower().startswith('s'):
                continue
            try:
                subject_id = int(subj_name[1:])
            except ValueError:
                continue
            for emotion in sorted(os.listdir(subj_path)):
                emo_path = os.path.join(subj_path, emotion)
                if not os.path.isdir(emo_path):
                    continue
                emo_key = emotion.strip().lower()
                if emo_key not in emotion_map:
                    # skip unknown emotion folders
                    continue
                label = emotion_map[emo_key]
                for fname in sorted(os.listdir(emo_path)):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in VALID_EXTS:
                        img_path = os.path.join(emo_path, fname)
                        samples.append({
                            'path': img_path,
                            'subject': subject_id,
                            'emotion': emo_key,
                            'label': label
                        })
        return samples

    @staticmethod
    def get_all_subjects(samples: List[Dict]) -> List[int]:
        return sorted(list({s['subject'] for s in samples}))