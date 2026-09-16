import os
import cv2
import numpy as np
from samm_data_parser import SAMMDataParser
from dynamic_image_generator import DynamicImageGenerator

def generate_images():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_video_dir = os.path.join(base_dir, '../data/SAMM')
    anno_file = os.path.join(base_dir, '../data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx')
    output_dir = os.path.join(base_dir, 'dynamic_data')
    
    emotion_map = {
        'Happiness': 0, 'Anger': 1, 'Disgust': 1, 'Fear': 1, 'Sadness': 1, 'Contempt': 1, 'Surprise': 2, 'Other': -1
    }
    
    # Initialize parser
    print("Parsing annotation file...")
    parser = SAMMDataParser(anno_file, raw_video_dir, emotion_map)
    samples = parser.get_samples()
    print(f"Found {len(samples)} samples.")
    
    # Initialize generator
    generator = DynamicImageGenerator(output_dir)
    
    for i, sample in enumerate(samples):
        subject = sample['subject']
        video_name = sample['video_name']
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']
        
        # Load frames
        frames = []
        if not os.path.exists(sequence_path):
            continue
            
        file_list = sorted(os.listdir(sequence_path))
        valid_files = []
        
        for f in file_list:
            if f.lower().endswith('.jpg'):
                try:
                    # Format: Subject_Frame.jpg (e.g. 006_0180.jpg)
                    parts = f.split('_')
                    if len(parts) >= 2:
                        frame_str = parts[-1].split('.')[0]
                        frame_num = int(frame_str)
                        if onset <= frame_num <= offset:
                            valid_files.append(os.path.join(sequence_path, f))
                except ValueError:
                    continue
        
        if not valid_files:
            print(f"No valid frames for {video_name}")
            continue
            
        # Read images
        image_sequence = []
        for vf in valid_files:
            img = cv2.imread(vf)
            if img is not None:
                image_sequence.append(img)
        
        if not image_sequence:
            continue
            
        # Generate
        dynamic_img = generator.generate_dynamic_image(image_sequence)
        
        # Save manually to match SAMMFusionDataset expectation
        # Expectation: output_dir / sub{subject} / s{subject}_{video_name}.jpg
        # subject is '006'
        sub_dir = os.path.join(output_dir, f"sub{subject}")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        save_name = f"s{subject}_{video_name}.jpg"
        save_path = os.path.join(sub_dir, save_name)
        cv2.imwrite(save_path, dynamic_img)
        
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(samples)}")

    print("Dynamic image generation complete.")

if __name__ == '__main__':
    generate_images()
