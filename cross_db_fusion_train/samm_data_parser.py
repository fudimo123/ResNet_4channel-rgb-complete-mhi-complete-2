import os
import pandas as pd
import numpy as np

class SAMMDataParser:
    def __init__(self, annotation_file, raw_video_dir, emotion_map):
        self.annotation_file = annotation_file
        self.raw_video_dir = raw_video_dir
        self.emotion_map = emotion_map
        self.samples = []
        self._parse_data()

    def _parse_data(self):
        # Read Excel file, header is at row 14 (index 13)
        try:
            df = pd.read_excel(self.annotation_file, header=13)
        except Exception as e:
            print(f"Error reading annotation file: {e}")
            return

        # Iterate through rows
        for index, row in df.iterrows():
            subject = str(row['Subject'])
            # Pad subject with leading zeros if necessary (SAMM uses 006, 007...)
            # The excel might store it as int 6.
            if len(subject) < 3:
                subject = subject.zfill(3)
                
            filename = str(row['Filename'])
            emotion_raw = str(row['Estimated Emotion'])
            onset = int(row['Onset Frame'])
            offset = int(row['Offset Frame'])
            apex = int(row['Apex Frame'])

            # Map emotion
            # Handle potential case variations or whitespace
            emotion_clean = emotion_raw.strip()
            if emotion_clean not in self.emotion_map:
                # Try capital case or check if it's 'Other'
                if 'Other' in emotion_clean:
                    label = -1
                else:
                    print(f"Warning: Unknown emotion '{emotion_clean}' for {filename}. Skipping or marking as -1.")
                    label = -1
            else:
                label = self.emotion_map[emotion_clean]

            if label == -1:
                continue

            # Construct path: raw_video_dir / Subject / Filename
            # Example: data/SAMM/006/006_1_2
            sequence_path = os.path.join(self.raw_video_dir, subject, filename)
            
            if not os.path.exists(sequence_path):
                print(f"Warning: Sequence path does not exist: {sequence_path}")
                continue

            sample = {
                'subject': subject,
                'video_name': filename,
                'raw_video_path': sequence_path,
                'onset': onset,
                'offset': offset,
                'apex': apex,
                'label': label,
                'emotion_raw': emotion_clean
            }
            self.samples.append(sample)

    def get_samples(self):
        return self.samples

    def get_all_subjects(self):
        return sorted(list(set(s['subject'] for s in self.samples)))

if __name__ == '__main__':
    # Test code
    anno_file = r'../data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
    video_dir = r'../data/SAMM'
    
    emotion_map = {
        'Happiness': 0,
        'Anger': 1,
        'Disgust': 1,
        'Fear': 1,
        'Sadness': 1,
        'Contempt': 1,
        'Surprise': 2,
        'Other': -1
    }
    
    if os.path.exists(anno_file):
        parser = SAMMDataParser(anno_file, video_dir, emotion_map)
        print(f"Found {len(parser.get_samples())} samples.")
        print(f"Subjects: {parser.get_all_subjects()}")
    else:
        print("Annotation file not found (relative path might be wrong for standalone run)")
