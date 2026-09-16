import os
import re

class DSMEDataParser:
    def __init__(self, raw_video_dir):
        self.raw_video_dir = raw_video_dir
        self._synonyms = {
            'hapiness': 'happiness',
            'suprise': 'surprise',
            'digust': 'disgust'
        }
        self._canonical_order = ['happiness', 'sadness', 'fear', 'disgust', 'surprise']
        self.class_names = ['positive', 'negative', 'surprise']
        self.emotion_map = self._discover_emotions()
        self.subjects = self._discover_subjects()

    def _discover_subjects(self):
        subs = []
        for name in os.listdir(self.raw_video_dir):
            if os.path.isdir(os.path.join(self.raw_video_dir, name)) and re.match(r"^\d{3}$", name):
                subs.append(int(name))
        subs.sort()
        return subs

    def _canonicalize(self, name):
        n = name.strip().lower()
        return self._synonyms.get(n, n)

    def _discover_emotions(self):
        present = set()
        for name in os.listdir(self.raw_video_dir):
            subj_path = os.path.join(self.raw_video_dir, name)
            if not os.path.isdir(subj_path):
                continue
            for emo in os.listdir(subj_path):
                emo_path = os.path.join(subj_path, emo)
                if os.path.isdir(emo_path):
                    present.add(self._canonicalize(emo))
        ordered = [e for e in self._canonical_order if e in present]
        # map canonical emotions into three classes
        three_class_map = {
            'happiness': 0,  # positive
            'sadness': 1,    # negative
            'fear': 1,       # negative
            'disgust': 1,    # negative
            'surprise': 2    # surprise
        }
        return {emo: three_class_map[emo] for emo in ordered}

    def get_all_subjects(self):
        return self.subjects

    def get_samples(self, subjects=None):
        samples = []
        target_subjects = subjects if subjects is not None else self.subjects

        for subject in target_subjects:
            subj_dir = os.path.join(self.raw_video_dir, f"{subject:03d}")
            if not os.path.isdir(subj_dir):
                continue
            for emotion in os.listdir(subj_dir):
                emo_dir = os.path.join(subj_dir, emotion)
                if not os.path.isdir(emo_dir):
                    continue
                canon_emotion = self._canonicalize(emotion)
                if canon_emotion not in self.emotion_map:
                    continue
                label = self._to_three_class_label(canon_emotion)
                for segment in os.listdir(emo_dir):
                    seg_dir = os.path.join(emo_dir, segment)
                    if not os.path.isdir(seg_dir):
                        continue
                    image_files = [f for f in os.listdir(seg_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
                    # sort by numeric frame index if possible
                    def frame_index(fn):
                        m = re.search(r"(\d+)", fn)
                        return int(m.group(1)) if m else 0
                    image_files = sorted(image_files, key=frame_index)
                    if len(image_files) == 0:
                        continue
                    onset = frame_index(image_files[0])
                    offset = frame_index(image_files[-1])
                    sequence_name = f"{canon_emotion}_{segment}"
                    samples.append({
                        'subject': int(subject),
                        'sequence': sequence_name,
                        'onset': onset,
                        'offset': offset,
                        'emotion': canon_emotion,
                        'raw_emotion': emotion,
                        'segment': segment,
                        'label': label,
                        'raw_video_path': seg_dir,
                        'image_files': image_files
                    })
        return samples
    def _to_three_class_label(self, canon_emotion):
        if canon_emotion == 'happiness':
            return 0
        if canon_emotion == 'surprise':
            return 2
        return 1