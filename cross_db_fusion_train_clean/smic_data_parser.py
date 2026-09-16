import os

class SMICDataParser:
    def __init__(self, aligned_rgb_dir, emotion_map):
        """
        aligned_rgb_dir: path to 'smic_aligned_rgb'
        emotion_map: e.g., {'positive': 0, 'negative': 1, 'surprise': 2}
        """
        self.aligned_rgb_dir = aligned_rgb_dir
        self.emotion_map = emotion_map
        self.subjects = self._get_all_subjects()

    def _get_all_subjects(self):
        """获取所有受试者列表"""
        subjects = []
        for item in os.listdir(self.aligned_rgb_dir):
            if os.path.isdir(os.path.join(self.aligned_rgb_dir, item)) and item.startswith('s'):
                try:
                    subject_num = int(item[1:])
                    subjects.append(subject_num)
                except ValueError:
                    continue
        return sorted(subjects)

    def get_samples(self, subjects=None):
        """获取预处理后的样本数据"""
        samples = []
        target_subjects = subjects if subjects else self.subjects
        
        # Mapping from sequence code to full emotion name
        code_to_emotion = {
            'ne': 'negative',
            'po': 'positive',
            'sur': 'surprise'
        }
        
        for subject_num in target_subjects:
            subject_dir = os.path.join(self.aligned_rgb_dir, f's{subject_num}')
            if not os.path.exists(subject_dir):
                continue
                
            for sequence in os.listdir(subject_dir):
                sequence_path = os.path.join(subject_dir, sequence)
                if not os.path.isdir(sequence_path):
                    continue
                
                # Infer emotion from sequence name, e.g. "s1_ne_01" -> "ne"
                parts = sequence.split('_')
                if len(parts) < 2:
                    continue
                emo_code = parts[1]
                emotion = code_to_emotion.get(emo_code)
                
                if emotion not in self.emotion_map:
                    continue
                    
                # 获取该序列中的所有预处理图片文件 (.jpg)
                image_files = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
                if not image_files:
                    continue
                    
                # 按文件名排序确保顺序正确
                image_files.sort()
                
                sample = {
                    'subject': f's{subject_num}', # keep as string e.g. 's1'
                    'sequence': sequence,
                    'emotion': emotion,
                    'label': self.emotion_map[emotion],
                    'aligned_path': sequence_path,
                    'image_files': image_files,
                    'num_frames': len(image_files)
                }
                samples.append(sample)
        return samples

    def get_all_subjects(self):
        """返回所有受试者列表"""
        return self.subjects
