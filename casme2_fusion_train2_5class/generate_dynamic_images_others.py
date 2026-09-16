import os
import cv2
import numpy as np
import pandas as pd
from dynamic_image_generator import DynamicImageGenerator
from casme2_data_parser import CASME2DataParser


def load_image_sequence(sequence_path, onset, offset):
    if not os.path.exists(sequence_path):
        return []
    image_files = []
    for i in range(onset, offset + 1):
        img_path = os.path.join(sequence_path, f'img{i}.jpg')
        if os.path.exists(img_path):
            image_files.append(img_path)
    images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images


def main():
    script_dir = os.path.dirname(__file__)
    default_anno = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx'))
    default_raw = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CASME2_RAW_selected', 'CASME2_RAW_selected'))
    annotation_file = os.environ.get('CASME2_ANNOTATION', default_anno)
    raw_video_dir = os.environ.get('CASME2_RAW_DIR', default_raw)
    output_dir = os.path.join(script_dir, 'dynamic_data')

    emotion_map = {
        'happiness': 0,
        'disgust': 1,
        'repression': 2,
        'sadness': 3,
        'fear': 4,
        'surprise': 5,
        'others': 6
    }

    parser = CASME2DataParser(annotation_file, raw_video_dir, emotion_map)
    generator = DynamicImageGenerator(output_dir)

    samples = parser.get_samples()
    target_samples = [s for s in samples if s['emotion'] == 'others']

    processed, skipped, failed = 0, 0, 0
    for sample in target_samples:
        subject = sample['subject']
        sequence = sample['sequence']
        onset = sample['onset']
        offset = sample['offset']
        sequence_path = sample['raw_video_path']

        subject_dir = os.path.join(output_dir, f"sub{subject:02d}")
        os.makedirs(subject_dir, exist_ok=True)
        out_path = os.path.join(subject_dir, f"s{subject}_{sequence}.jpg")
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            image_sequence = load_image_sequence(sequence_path, onset, offset)
            if len(image_sequence) == 0:
                failed += 1
                continue
            dynamic_image = generator.generate_dynamic_image(image_sequence)
            if dynamic_image is None:
                failed += 1
                continue
            generator.save_dynamic_image(subject, sequence, dynamic_image)
            processed += 1
        except Exception:
            failed += 1
            continue

    report_path = os.path.join(output_dir, 'others_generation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Processed: {processed}\n")
        f.write(f"Skipped (exists): {skipped}\n")
        f.write(f"Failed: {failed}\n")


if __name__ == '__main__':
    main()
