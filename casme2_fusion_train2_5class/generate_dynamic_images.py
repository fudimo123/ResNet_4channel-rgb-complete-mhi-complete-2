import os
import cv2
import numpy as np
import pandas as pd
from dynamic_image_generator import DynamicImageGenerator

class CASME2DataParser:
    def __init__(self, annotation_file, raw_video_dir, emotion_map):
        self.df = pd.read_excel(annotation_file)
        self.raw_video_dir = raw_video_dir
        self.emotion_map = emotion_map
        self.subjects = self.df['Subject'].unique()

    def get_samples(self, subjects=None):
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
            emotion = row['Estimated Emotion']

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

def load_image_sequence(sequence_path, onset, offset):
    """
    Load image sequence from onset to offset frames
    """
    if not os.path.exists(sequence_path):
        print(f"Warning: Sequence path does not exist: {sequence_path}")
        return []
    
    image_files = []
    for i in range(onset, offset + 1):
        img_path = os.path.join(sequence_path, f'img{i}.jpg')
        if os.path.exists(img_path):
            image_files.append(img_path)
    
    if not image_files:
        print(f"Warning: No images found in range {onset}-{offset} for {sequence_path}")
        return []
    
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")
    
    return images

def main():
    # Configuration
    script_dir = os.path.dirname(__file__)
    annotation_file = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx'))
    raw_video_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2_RAW_selected'))
    output_dir = os.path.join(script_dir, 'dynamic_data')
    
    # Emotion mapping (consistent with fusion training)
    emotion_map = {
        'disgust': 0,
        'happiness': 1,
        'repression': 2,
        'surprise': 3,
        'sadness': 4,
        'fear': 4,
        'others': 4
    }
    
    # Initialize components
    parser = CASME2DataParser(annotation_file, raw_video_dir, emotion_map)
    generator = DynamicImageGenerator(output_dir)
    
    # Get all samples
    samples = parser.get_samples()
    
    print(f"Found {len(samples)} samples to process")
    
    processed_count = 0
    failed_count = 0
    
    for sample in samples:
        try:
            subject = sample['subject']
            sequence = sample['sequence']
            onset = sample['onset']
            offset = sample['offset']
            sequence_path = sample['raw_video_path']
            
            print(f"Processing: s{subject}_{sequence} (onset: {onset}, offset: {offset})")
            
            # Load image sequence
            image_sequence = load_image_sequence(sequence_path, onset, offset)
            
            if len(image_sequence) == 0:
                print(f"Skipping s{subject}_{sequence}: No images loaded")
                failed_count += 1
                continue
            
            # Generate dynamic image
            dynamic_image = generator.generate_dynamic_image(image_sequence)
            
            if dynamic_image is not None:
                # Save dynamic image
                output_path = generator.save_dynamic_image(subject, sequence, dynamic_image)
                print(f"Generated: {output_path}")
                processed_count += 1
            else:
                print(f"Failed to generate dynamic image for s{subject}_{sequence}")
                failed_count += 1
                
        except Exception as e:
            print(f"Error processing s{subject}_{sequence}: {str(e)}")
            failed_count += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(samples)}")

if __name__ == '__main__':
    main()
