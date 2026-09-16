import os
import cv2
import numpy as np
from casme2_data_parser import CASME2DataParser

RAW_VIDEO_DIR = '../data/CASME2_RAW_selected/CASME2_RAW_selected'
ANNOTATION_FILE = '../data/CASME2_RAW_selected/CASME2-coding-20140508.xlsx'
OUTPUT_ROOT = './dynamic_data_steps'

EMOTION_MAP = {
    'happiness': 0,
    'disgust': 1,
    'repression': 1,
    'sadness': 1,
    'fear': 1,
    'surprise': 2
}

def compute_coefficients(n):
    coeffs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = 0.0
        for j in range(i, n):
            s += (2 * (j + 1) - n - 1) / (j + 1)
        coeffs[i] = s
    denom = np.sum(np.abs(coeffs))
    if denom == 0:
        denom = 1.0
    coeffs /= denom
    return coeffs

def normalize_image(x):
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(x)

def load_sequence_frames(seq_path, onset, offset):
    try:
        files = sorted(os.listdir(seq_path))
    except FileNotFoundError:
        return []
    frames = []
    for f in files:
        if f.startswith('img') and f.endswith('.jpg'):
            try:
                idx = int(f[3:-4])
            except ValueError:
                continue
            if onset <= idx <= offset:
                p = os.path.join(seq_path, f)
                img = cv2.imread(p)
                if img is None:
                    continue
                frames.append(img.astype(np.float32))
    return frames

def save_steps(subject, sequence, steps):
    subdir = os.path.join(OUTPUT_ROOT, f'sub{subject:02d}')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    for i, img in enumerate(steps, 1):
        name = f's{subject}_{sequence}_step{i}.jpg'
        cv2.imwrite(os.path.join(subdir, name), img)

def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    parser = CASME2DataParser(ANNOTATION_FILE, RAW_VIDEO_DIR, EMOTION_MAP)
    samples = parser.get_samples()
    for s in samples:
        subject = s['subject']
        sequence = s['sequence']
        onset = s['onset']
        offset = s['offset']
        seq_path = s['raw_video_path']
        frames = load_sequence_frames(seq_path, onset, offset)
        if len(frames) == 0:
            continue
        coeffs = compute_coefficients(len(frames))
        steps = []
        acc = np.zeros_like(frames[0], dtype=np.float32)
        for i, frame in enumerate(frames):
            acc += coeffs[i] * frame
            steps.append(normalize_image(acc))
        save_steps(subject, sequence, steps)

if __name__ == '__main__':
    main()