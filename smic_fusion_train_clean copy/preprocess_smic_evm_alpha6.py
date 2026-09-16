import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


TARGET_FRAMES = 16
IMG_SIZE = (224, 224)
MARGIN = 20


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Offline EVM preprocessing for SMIC (alpha=6): align/crop -> EVM -> TIM(16) -> dynamic image."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(project_root / "data" / "SMIC2"),
        help="Raw SMIC2 root directory.",
    )
    parser.add_argument(
        "--xlsx-path",
        default=str(project_root / "data" / "SMIC2" / "SMIC_HS_E_spotting.xlsx"),
        help="SMIC spotting xlsx path used to validate micro sequence names.",
    )
    parser.add_argument(
        "--output-rgb-dir",
        default=str(script_dir / "smic_aligned_rgb_evm_alpha6"),
        help="Output directory for EVM-processed aligned RGB sequences.",
    )
    parser.add_argument(
        "--output-dyn-dir",
        default=str(script_dir / "smic_dynamic_data_evm_alpha6"),
        help="Output directory for dynamic images built from EVM-processed sequences.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for MTCNN face detection.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=100.0,
        help="Frame rate used by temporal filtering in EVM.",
    )
    parser.add_argument(
        "--freq-low",
        type=float,
        default=0.4,
        help="Low cut frequency for EVM band-pass filter.",
    )
    parser.add_argument(
        "--freq-high",
        type=float,
        default=3.0,
        help="High cut frequency for EVM band-pass filter.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=6.0,
        help="Amplification factor for EVM.",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=1,
        help="Gaussian pyramid level used by EVM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output even if the target sequence already exists.",
    )
    return parser.parse_args()

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


def load_valid_micro_sequences(xlsx_path):
    if pd is None or not os.path.exists(xlsx_path):
        return None

    df = pd.read_excel(xlsx_path)
    me_type_columns = [col for col in df.columns if str(col).startswith("ME Type")]
    valid_sequences = set()

    for col in me_type_columns:
        for value in df[col].dropna():
            seq_name = str(value).strip()
            if seq_name:
                valid_sequences.add(seq_name)

    return valid_sequences or None


def reduce_pyramid(frames, level):
    reduced = []
    for frame in frames:
        cur = frame
        for _ in range(level):
            cur = cv2.pyrDown(cur)
        reduced.append(cur)
    return np.stack(reduced, axis=0)


def expand_frame(frame, level, target_hw):
    cur = frame
    target_h, target_w = target_hw
    for _ in range(level):
        cur = cv2.pyrUp(cur)
    if cur.shape[:2] != (target_h, target_w):
        cur = cv2.resize(cur, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cur


def apply_evm(frames_rgb, fps=100.0, freq_low=0.4, freq_high=3.0, alpha=8.0, pyramid_level=1):
    if len(frames_rgb) < 3:
        return frames_rgb

    frames_float = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
    original_hw = frames_float.shape[1:3]
    reduced = reduce_pyramid(frames_float, pyramid_level)

    freqs = np.fft.fftfreq(reduced.shape[0], d=1.0 / fps)
    fft_video = np.fft.fft(reduced, axis=0)
    mask = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
    fft_video[~mask] = 0
    filtered = np.fft.ifft(fft_video, axis=0).real

    amplified_frames = []
    for t in range(frames_float.shape[0]):
        amplified_low = filtered[t] * alpha
        amplified_full = expand_frame(amplified_low, pyramid_level, original_hw)
        out = np.clip(frames_float[t] + amplified_full, 0.0, 1.0)
        amplified_frames.append(np.uint8(np.round(out * 255.0)))

    return amplified_frames

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

def process_sequence(
    seq_dir,
    subject,
    seq_name,
    mtcnn,
    output_rgb_dir,
    output_dyn_dir,
    fps,
    freq_low,
    freq_high,
    alpha,
    pyramid_level,
    overwrite=False,
):
    # Find all frames
    frames = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith(('.bmp', '.jpg', '.png'))])
    if not frames:
        return {"status": "failed", "subject": subject, "sequence": seq_name, "reason": "no_frames"}

    out_seq_dir = os.path.join(output_rgb_dir, subject, seq_name)
    dyn_path = os.path.join(output_dyn_dir, subject, f"{subject}_{seq_name}.jpg")
    if not overwrite and os.path.isdir(out_seq_dir) and os.path.exists(dyn_path):
        return {"status": "skipped", "subject": subject, "sequence": seq_name}
        
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
        return {"status": "failed", "subject": subject, "sequence": seq_name, "reason": "read_failed"}

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

    evm_frames = apply_evm(
        [np.array(frame) for frame in cropped_frames],
        fps=fps,
        freq_low=freq_low,
        freq_high=freq_high,
        alpha=alpha,
        pyramid_level=pyramid_level,
    )

    # 2. Time Interpolation (TIM)
    # Select TARGET_FRAMES indices evenly spaced
    indices = np.linspace(0, len(evm_frames) - 1, TARGET_FRAMES).astype(int)
    tim_frames = [Image.fromarray(evm_frames[i]) for i in indices]

    # Save aligned and interpolated RGB frames
    os.makedirs(out_seq_dir, exist_ok=True)
    
    cv2_frames = []
    for i, frame in enumerate(tim_frames):
        frame.save(os.path.join(out_seq_dir, f"img_{i:04d}.jpg"))
        cv2_frames.append(np.array(frame))

    # 3. Dynamic Image Generation
    dyn_img = generate_dynamic_image(cv2_frames)
    if dyn_img is not None:
        dyn_img_bgr = cv2.cvtColor(dyn_img, cv2.COLOR_RGB2BGR)
        out_sub_dyn_dir = os.path.join(output_dyn_dir, subject)
        os.makedirs(out_sub_dyn_dir, exist_ok=True)
        cv2.imwrite(dyn_path, dyn_img_bgr)
        
    return {"status": "done", "subject": subject, "sequence": seq_name}

def main():
    args = parse_args()

    print("Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=False, device=args.device)
    
    os.makedirs(args.output_rgb_dir, exist_ok=True)
    os.makedirs(args.output_dyn_dir, exist_ok=True)

    valid_sequences = load_valid_micro_sequences(args.xlsx_path)
    summary = {
        "raw_dir": os.path.abspath(args.raw_dir),
        "xlsx_path": os.path.abspath(args.xlsx_path),
        "output_rgb_dir": os.path.abspath(args.output_rgb_dir),
        "output_dyn_dir": os.path.abspath(args.output_dyn_dir),
        "fps": args.fps,
        "freq_low": args.freq_low,
        "freq_high": args.freq_high,
        "alpha": args.alpha,
        "pyramid_level": args.pyramid_level,
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    print(f"Scanning raw data in {args.raw_dir}...")
    
    subjects = [d for d in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, d))]
    
    for subject in subjects:
        sub_path = os.path.join(args.raw_dir, subject, 'micro')
        if not os.path.exists(sub_path):
            continue
            
        emotions = os.listdir(sub_path)
        for emotion in emotions:
            emo_path = os.path.join(sub_path, emotion)
            if not os.path.isdir(emo_path):
                continue
                
            sequences = os.listdir(emo_path)
            for seq in sequences:
                if valid_sequences is not None and seq not in valid_sequences:
                    continue
                seq_path = os.path.join(emo_path, seq)
                if not os.path.isdir(seq_path):
                    continue
                    
                result = process_sequence(
                    seq_path,
                    subject,
                    seq,
                    mtcnn,
                    args.output_rgb_dir,
                    args.output_dyn_dir,
                    args.fps,
                    args.freq_low,
                    args.freq_high,
                    args.alpha,
                    args.pyramid_level,
                    overwrite=args.overwrite,
                )
                summary[result["status"]] += 1

    print("Preprocessing completed!")
    summary_path = Path(args.output_rgb_dir).parent / "preprocess_smic_evm_alpha6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
