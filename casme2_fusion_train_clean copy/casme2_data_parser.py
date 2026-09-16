import os

import pandas as pd


class CASME2DataParser:
    def __init__(self, mapping_file, aligned_rgb_dir, emotion_map):
        self.mapping_file = mapping_file
        self.aligned_rgb_dir = aligned_rgb_dir
        self.emotion_map = emotion_map
        self.samples = []
        self._parse_data()

    def _parse_data(self):
        if not os.path.exists(self.mapping_file):
            print(f"Error: mapping file {self.mapping_file} not found.")
            return

        df = pd.read_csv(self.mapping_file)

        for _, row in df.iterrows():
            subject_num = int(row['subject_num'])
            subject = str(row['subject'])
            sequence = str(row['sequence'])
            emotion_clean = str(row['emotion'])

            if emotion_clean not in self.emotion_map:
                continue

            label = self.emotion_map[emotion_clean]
            sequence_path = os.path.join(self.aligned_rgb_dir, subject, sequence)
            if not os.path.exists(sequence_path):
                print(f"Warning: aligned sequence path does not exist: {sequence_path}")
                continue

            image_files = sorted(f for f in os.listdir(sequence_path) if f.lower().endswith('.jpg'))
            if not image_files:
                continue

            self.samples.append(
                {
                    'subject': subject,
                    'subject_num': subject_num,
                    'sequence': sequence,
                    'aligned_path': sequence_path,
                    'image_files': image_files,
                    'num_frames': len(image_files),
                    'label': label,
                    'emotion': emotion_clean,
                }
            )

    def get_samples(self):
        return self.samples

    def get_all_subjects(self):
        return sorted(list(set(s['subject'] for s in self.samples)))
