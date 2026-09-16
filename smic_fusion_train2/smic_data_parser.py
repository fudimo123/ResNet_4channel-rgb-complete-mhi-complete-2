import os

class SMICDataParser:
    def __init__(self, raw_video_dir, emotion_map):
        self.raw_video_dir = raw_video_dir
        self.emotion_map = emotion_map
        self.subjects = self._get_all_subjects()

    def _get_all_subjects(self):
        """获取所有受试者列表"""
        subjects = []
        for item in os.listdir(self.raw_video_dir):
            if os.path.isdir(os.path.join(self.raw_video_dir, item)) and item.startswith('s'):
                try:
                    subject_num = int(item[1:])
                    subjects.append(subject_num)
                except ValueError:
                    continue
        return sorted(subjects)

    def get_samples(self, subjects=None):
        """获取样本数据，只处理micro表情"""
        samples = []
        target_subjects = subjects if subjects else self.subjects
        
        for subject_num in target_subjects:
            subject_dir = os.path.join(self.raw_video_dir, f's{subject_num}')
            if not os.path.exists(subject_dir):
                continue
                
            # 只处理micro表情
            micro_dir = os.path.join(subject_dir, 'micro')
            if os.path.exists(micro_dir):
                for emotion in os.listdir(micro_dir):
                    if emotion not in self.emotion_map:
                        continue
                        
                    emotion_dir = os.path.join(micro_dir, emotion)
                    if not os.path.isdir(emotion_dir):
                        continue
                        
                    for sequence in os.listdir(emotion_dir):
                        sequence_path = os.path.join(emotion_dir, sequence)
                        if not os.path.isdir(sequence_path):
                            continue
                            
                        # 获取该序列中的所有图片文件
                        image_files = [f for f in os.listdir(sequence_path) if f.endswith('.bmp')]
                        if not image_files:
                            continue
                            
                        # 按文件名排序确保顺序正确
                        image_files.sort()
                        
                        sample = {
                            'subject': subject_num,
                            'sequence': sequence,
                            'emotion': emotion,
                            'label': self.emotion_map[emotion],
                            'raw_video_path': sequence_path,
                            'image_files': image_files,
                            'num_frames': len(image_files)
                        }
                        samples.append(sample)
        return samples

    def get_all_subjects(self):
        """返回所有受试者列表"""
        return self.subjects