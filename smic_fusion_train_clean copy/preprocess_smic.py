import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import shutil
import glob
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = '../data/SMIC2'
OUTPUT_RGB_DIR = 'smic_aligned_rgb'
OUTPUT_DYN_DIR = 'smic_dynamic_data_clean'
TARGET_FRAMES = 16
IMG_SIZE = (224, 224)
MARGIN = 20

def generate_dynamic_image(image_sequence):
    num_frames = len(image_sequence)
    if num_frames == 0:
        return None

    # Calculate weights (coefficients) for each frame
    coefficients = np.zeros(num_frames)
    for i in range(num_frames):
        coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])

    # Normalize coefficients
    coefficients /= np.sum(np.abs(coefficients))

    # Create the dynamic image
    first_frame = image_sequence[0]
    dynamic_image = np.zeros_like(first_frame, dtype=np.float32)

    for i, frame in enumerate(image_sequence):
        dynamic_image += coefficients[i] * frame

    # Normalize the dynamic image to 0-255 and convert to uint8
    dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
    dynamic_image = np.uint8(dynamic_image)

    return dynamic_image

def get_face_box(mtcnn, img_pil):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is not None:
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Add margin
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(img_pil.width, x2 + MARGIN)
        y2 = min(img_pil.height, y2 + MARGIN)
        return (x1, y1, x2, y2)
    return None

def process_sequence(seq_dir, subject, emotion, seq_name, mtcnn):
    # Find all frames
    frames = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith(('.bmp', '.jpg', '.png'))])
    if not frames:
        return False
        
    # Read all frames into PIL
    frame_pils = []
    for f in frames:
        path = os.path.join(seq_dir, f)
        try:
            img = Image.open(path).convert('RGB')
            frame_pils.append(img)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    if not frame_pils:
        return False

    # 1. Face Alignment / Cropping
    # Try to find a bounding box using the first few frames
    box = None
    for img in frame_pils[:min(5, len(frame_pils))]:
        box = get_face_box(mtcnn, img)
        if box is not None:
            break
            
    # If no face detected, just use full image (though rare for SMIC)
    if box is None:
        print(f"Warning: No face detected in sequence {seq_name}. Using full frame.")
        box = (0, 0, frame_pils[0].width, frame_pils[0].height)

    cropped_frames = []
    for img in frame_pils:
        crop_img = img.crop(box)
        crop_img = crop_img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        cropped_frames.append(crop_img)

    # 2. Time Interpolation (TIM)
    # Select TARGET_FRAMES indices evenly spaced
    indices = np.linspace(0, len(cropped_frames) - 1, TARGET_FRAMES).astype(int)
    tim_frames = [cropped_frames[i] for i in indices]

    # Save aligned and interpolated RGB frames
    out_seq_dir = os.path.join(OUTPUT_RGB_DIR, subject, seq_name)
    os.makedirs(out_seq_dir, exist_ok=True)
    
    cv2_frames = []
    for i, frame in enumerate(tim_frames):
        frame.save(os.path.join(out_seq_dir, f"img_{i:04d}.jpg"))
        # Convert to cv2 format (BGR) for dynamic image generation
        cv2_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        # However, generate_dynamic_image doesn't strictly care about RGB vs BGR if it's just pixel differences,
        # but to be consistent with PIL/RGB, let's keep it as RGB numpy array
        cv2_frames.append(np.array(frame))

    # 3. Dynamic Image Generation
    dyn_img = generate_dynamic_image(cv2_frames)
    if dyn_img is not None:
        # dyn_img is RGB numpy array here, need to convert to BGR for cv2.imwrite
        dyn_img_bgr = cv2.cvtColor(dyn_img, cv2.COLOR_RGB2BGR)
        out_sub_dyn_dir = os.path.join(OUTPUT_DYN_DIR, subject)
        os.makedirs(out_sub_dyn_dir, exist_ok=True)
        dyn_path = os.path.join(out_sub_dyn_dir, f"{subject}_{seq_name}.jpg")
        cv2.imwrite(dyn_path, dyn_img_bgr)
        
    return True

def main():
    print("Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=False, device='cpu')
    
    os.makedirs(OUTPUT_RGB_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DYN_DIR, exist_ok=True)

    print(f"Scanning raw data in {RAW_DATA_DIR}...")
    # Expected structure: RAW_DATA_DIR/s1/micro/negative/s1_ne_01
    
    subjects = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    for subject in tqdm(subjects, desc="Subjects"):
        sub_path = os.path.join(RAW_DATA_DIR, subject, 'micro')
        if not os.path.exists(sub_path):
            continue
            
        emotions = os.listdir(sub_path)
        for emotion in emotions:
            emo_path = os.path.join(sub_path, emotion)
            if not os.path.isdir(emo_path):
                continue
                
            sequences = os.listdir(emo_path)
            for seq in sequences:
                seq_path = os.path.join(emo_path, seq)
                if not os.path.isdir(seq_path):
                    continue
                    
                process_sequence(seq_path, subject, emotion, seq, mtcnn)

    print("Preprocessing completed!")

if __name__ == '__main__':
    main()
