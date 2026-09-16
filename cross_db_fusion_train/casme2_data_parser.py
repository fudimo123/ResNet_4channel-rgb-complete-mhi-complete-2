import pandas as pd
import os

class CASME2DataParser:
    def __init__(self, annotation_file, raw_video_dir, emotion_map):
        if annotation_file.endswith('.csv'):
            self.df = pd.read_csv(annotation_file)
        else:
            self.df = pd.read_excel(annotation_file)
        self.raw_video_dir = raw_video_dir
        self.emotion_map = emotion_map
        # self.subjects = self.df['Subject'].unique() # This fails if 'Subject' column missing in CSV
        
    def get_samples(self, subjects=None):
        if 'Subject' not in self.df.columns:
             # If loaded wrong file, return empty
             print(f"Warning: 'Subject' column not found in {self.df.columns}. Returning empty samples.")
             return []
             
        self.subjects = self.df['Subject'].unique()
        samples = []
        target_df = self.df
        if subjects:
            target_df = self.df[self.df['Subject'].isin(subjects)]

        for index, row in target_df.iterrows():
            subject = row['Subject']
            sequence = row['Filename']
            onset = row['OnsetFrame']
            apex = row['ApexFrame']
            offset = row['OffsetFrame']
            
            # Check for emotion column, it might be 'Estimated Emotion' or 'Emotion'
            if 'Estimated Emotion' in row:
                emotion = row['Estimated Emotion']
            elif 'Emotion' in row:
                emotion = row['Emotion']
            else:
                continue

            if emotion not in self.emotion_map:
                continue

            label = self.emotion_map[emotion]
            
            sequence_path = os.path.join(self.raw_video_dir, f'sub{subject:02d}', sequence)

            sample = {
                'subject': subject,
                'sequence': sequence,
                'onset': onset,
                'apex': apex,
                'offset': offset,
                'emotion': emotion,
                'label': label,
                'raw_video_path': sequence_path
            }
            samples.append(sample)
        return samples

    def get_all_subjects(self):
        all_subjects = self.subjects.tolist()
        return [s for s in all_subjects if s != 18]