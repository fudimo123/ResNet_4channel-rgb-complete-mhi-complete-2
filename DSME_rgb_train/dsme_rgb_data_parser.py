import os

class DSMERGBDataParser:
    def __init__(self, raw_video_dir, emotion_map):
        self.raw_video_dir = raw_video_dir
        self.emotion_map = emotion_map
        self.samples = []
        self._parse_data()

    def _parse_data(self):
        # DSME structure: DSME_pic / Subject (001) / Emotion (fear) / Sequence (1) / img1.jpg
        if not os.path.exists(self.raw_video_dir):
            print(f"Error: Dataset directory not found: {self.raw_video_dir}")
            return

        subjects = sorted([d for d in os.listdir(self.raw_video_dir) if os.path.isdir(os.path.join(self.raw_video_dir, d))])
        
        for subject in subjects:
            subject_path = os.path.join(self.raw_video_dir, subject)
            emotions = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
            
            for emotion_raw in emotions:
                # Fix typos and map
                emotion_clean = emotion_raw.lower().replace('hapiness', 'happiness').replace('digust', 'disgust').replace('suprise', 'surprise')
                
                if emotion_clean not in self.emotion_map:
                    # print(f"Skipping unknown emotion: {emotion_raw}")
                    continue
                    
                label = self.emotion_map[emotion_clean]
                if label == -1:
                    continue
                
                emotion_path = os.path.join(subject_path, emotion_raw)
                sequences = [d for d in os.listdir(emotion_path) if os.path.isdir(os.path.join(emotion_path, d))]
                
                for seq in sequences:
                    sequence_path = os.path.join(emotion_path, seq)
                    
                    sample = {
                        'subject': subject,
                        'sequence': seq,
                        'emotion': emotion_clean,
                        'label': label,
                        'raw_video_path': sequence_path
                    }
                    self.samples.append(sample)

    def get_samples(self):
        return self.samples

    def get_all_subjects(self):
        return sorted(list(set(s['subject'] for s in self.samples)))

if __name__ == '__main__':
    # Test
    raw_dir = r'../data/DSME_pic'
    emotion_map = {
        'happiness': 0,
        'disgust': 1, 'fear': 1, 'sadness': 1, 'repression': 1, 'anger': 1,
        'surprise': 2
    }
    if os.path.exists(raw_dir):
        parser = DSMERGBDataParser(raw_dir, emotion_map)
        print(f"Found {len(parser.get_samples())} samples.")
        print(f"Subjects: {parser.get_all_subjects()}")
