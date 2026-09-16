import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import pandas as pd
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = '../data/SAMM'
EXCEL_PATH = '../data/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
OUTPUT_RGB_DIR = 'samm_aligned_rgb'
OUTPUT_DYN_DIR = 'samm_dynamic_data_clean'
TARGET_FRAMES = 32
IMG_SIZE = (224, 224)
MARGIN = 20

# 3-Class Mapping
EMOTION_MAP = {
    'Happiness': 'positive',
    'Anger': 'negative',
    'Sadness': 'negative',
    'Disgust': 'negative',
    'Fear': 'negative',
    'Contempt': 'negative',
    'Surprise': 'surprise'
}

def generate_dynamic_image(image_sequence):
    num_frames = len(image_sequence)
    if num_frames == 0:
        return None

    coefficients = np.zeros(num_frames)
    for i in range(num_frames):
        coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])

    coefficients /= np.sum(np.abs(coefficients))

    first_frame = image_sequence[0]
    dynamic_image = np.zeros_like(first_frame, dtype=np.float32)

    for i, frame in enumerate(image_sequence):
        dynamic_image += coefficients[i] * frame

    dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
    dynamic_image = np.uint8(dynamic_image)
    return dynamic_image

def get_face_box(mtcnn, img_pil):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is not None:
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(img_pil.width, x2 + MARGIN)
        y2 = min(img_pil.height, y2 + MARGIN)
        return (x1, y1, x2, y2)
    return None

def process_sequence(seq_path, subject, seq_name, mapped_emotion, onset, offset, mtcnn):
    # Read frames in the given onset-offset range
    all_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith('.jpg')])
    
    valid_files = []
    for f in all_files:
        try:
            # SAMM format: {Subject}_{FrameNum}.jpg (e.g. 006_05562.jpg)
            frame_num = int(f.split('_')[-1].split('.')[0])
            if onset <= frame_num <= offset:
                valid_files.append(f)
        except Exception:
            continue
            
    if not valid_files:
        print(f"Warning: No valid frames found for {seq_name} between {onset}-{offset}")
        return False
        
    frame_pils = []
    for f in valid_files:
        path = os.path.join(seq_path, f)
        try:
            img = Image.open(path).convert('RGB')
            frame_pils.append(img)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    if not frame_pils:
        return False

    # 1. Face Alignment / Cropping
    box = None
    # Use first few frames to find a good box
    for img in frame_pils[:min(5, len(frame_pils))]:
        box = get_face_box(mtcnn, img)
        if box is not None:
            break
            
    if box is None:
        print(f"Warning: No face detected in sequence {seq_name}. Using full frame.")
        box = (0, 0, frame_pils[0].width, frame_pils[0].height)

    cropped_frames = []
    for img in frame_pils:
        crop_img = img.crop(box)
        crop_img = crop_img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        cropped_frames.append(crop_img)

    # 2. Time Interpolation (TIM)
    indices = np.linspace(0, len(cropped_frames) - 1, TARGET_FRAMES).astype(int)
    tim_frames = [cropped_frames[i] for i in indices]

    # Save aligned and interpolated RGB frames
    out_seq_dir = os.path.join(OUTPUT_RGB_DIR, subject, seq_name)
    os.makedirs(out_seq_dir, exist_ok=True)
    
    cv2_frames = []
    for i, frame in enumerate(tim_frames):
        # We save frame as img_000x.jpg to make it easy for parser
        frame.save(os.path.join(out_seq_dir, f"img_{i:04d}.jpg"))
        cv2_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        cv2_frames.append(np.array(frame)) # keeping RGB numpy for dynamic image

    # 3. Dynamic Image Generation
    dyn_img = generate_dynamic_image(cv2_frames)
    if dyn_img is not None:
        dyn_img_bgr = cv2.cvtColor(dyn_img, cv2.COLOR_RGB2BGR)
        out_sub_dyn_dir = os.path.join(OUTPUT_DYN_DIR, subject)
        os.makedirs(out_sub_dyn_dir, exist_ok=True)
        # Format: s{subject}_{seq_name}.jpg (e.g. s006_006_1_2.jpg)
        dyn_path = os.path.join(out_sub_dyn_dir, f"s{subject}_{seq_name}.jpg")
        cv2.imwrite(dyn_path, dyn_img_bgr)
        
    return True

def main():
    print("Loading SAMM Excel Data...")
    df = pd.read_excel(EXCEL_PATH, skiprows=13)
    
    # Clean Data
    df_clean = df[df['Estimated Emotion'] != 'Other'].copy()
    print(f"Total samples after removing 'Other': {len(df_clean)}")
    
    print("Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=False, device='cpu')
    
    os.makedirs(OUTPUT_RGB_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DYN_DIR, exist_ok=True)

    success_count = 0
    for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing SAMM"):
        # Format subject to '006'
        subject_id = str(row['Subject']).zfill(3)
        seq_name = str(row['Filename'])
        raw_emotion = str(row['Estimated Emotion']).strip()
        onset = int(row['Onset Frame'])
        offset = int(row['Offset Frame'])
        
        mapped_emotion = EMOTION_MAP.get(raw_emotion)
        if mapped_emotion is None:
            continue
            
        seq_path = os.path.join(RAW_DATA_DIR, subject_id, seq_name)
        if not os.path.exists(seq_path):
            print(f"Sequence missing: {seq_path}")
            continue
            
        if process_sequence(seq_path, subject_id, seq_name, mapped_emotion, onset, offset, mtcnn):
            # We don't write to csv here to avoid file lock issues.
            # We will generate the label mapping file separately after the loop.
            success_count += 1
            
    print(f"Preprocessing completed! Successfully processed {success_count} samples.")
    
    # Save mapping info for parser later
    print("Generating label mapping file...")
    mapping_file = os.path.join(OUTPUT_RGB_DIR, 'samm_clean_labels.csv')
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            f.write("subject,sequence,emotion\n")
            for idx, row in df_clean.iterrows():
                subject_id = str(row['Subject']).zfill(3)
                seq_name = str(row['Filename'])
                raw_emotion = str(row['Estimated Emotion']).strip()
                mapped_emotion = EMOTION_MAP.get(raw_emotion)
                if mapped_emotion is not None:
                    # Only write if we successfully created the folder
                    if os.path.exists(os.path.join(OUTPUT_RGB_DIR, subject_id, seq_name)):
                        f.write(f"{subject_id},{seq_name},{mapped_emotion}\n")
        print("Label mapping file created successfully.")
    except Exception as e:
        print(f"Failed to create mapping file: {e}")

if __name__ == '__main__':
    main()
