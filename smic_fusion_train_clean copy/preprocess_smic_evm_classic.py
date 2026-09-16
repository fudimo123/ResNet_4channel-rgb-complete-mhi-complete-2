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
        description="Offline classic EVM preprocessing for SMIC: align/crop -> classic EVM -> TIM(16) -> dynamic image."
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
        default=str(script_dir / "smic_aligned_rgb_evm_classic"),
        help="Output directory for classic-EVM-processed aligned RGB sequences.",
    )
    parser.add_argument(
        "--output-dyn-dir",
        default=str(script_dir / "smic_dynamic_data_evm_classic"),
        help="Output directory for dynamic images built from classic-EVM-processed sequences.",
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
        default=12.0,
        help="Maximum amplification factor for classic EVM.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Number of Laplacian pyramid levels used by classic EVM.",
    )
    parser.add_argument(
        "--lambda-cutoff",
        type=float,
        default=16.0,
        help="Spatial wavelength cutoff used by classic EVM amplification control.",
    )
    parser.add_argument(
        "--chrom-attenuation",
        type=float,
        default=0.1,
        help="Chrominance attenuation factor used to reduce color artifacts.",
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


def rgb_to_yiq(frames):
    transform = np.array(
        [
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ],
        dtype=np.float32,
    )
    return np.tensordot(frames, transform.T, axes=([3], [0]))


def yiq_to_rgb(frames):
    inverse_transform = np.array(
        [
            [1.0, 0.956, 0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106, 1.703],
        ],
        dtype=np.float32,
    )
    return np.tensordot(frames, inverse_transform.T, axes=([3], [0]))


def build_laplacian_pyramid(frame, levels):
    gaussian = [frame]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    pyramid = []
    for level in range(levels):
        expanded = cv2.pyrUp(
            gaussian[level + 1],
            dstsize=(gaussian[level].shape[1], gaussian[level].shape[0]),
        )
        if expanded.shape != gaussian[level].shape:
            expanded = cv2.resize(
                expanded,
                (gaussian[level].shape[1], gaussian[level].shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        pyramid.append(gaussian[level] - expanded)
    pyramid.append(gaussian[-1])
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    current = pyramid[-1]
    for level in range(len(pyramid) - 2, -1, -1):
        expanded = cv2.pyrUp(
            current,
            dstsize=(pyramid[level].shape[1], pyramid[level].shape[0]),
        )
        if expanded.shape != pyramid[level].shape:
            expanded = cv2.resize(
                expanded,
                (pyramid[level].shape[1], pyramid[level].shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        current = expanded + pyramid[level]
    return current


def temporal_ideal_bandpass(video, fps, freq_low, freq_high):
    freqs = np.fft.fftfreq(video.shape[0], d=1.0 / fps)
    fft_video = np.fft.fft(video, axis=0)
    mask = (np.abs(freqs) >= freq_low) & (np.abs(freqs) <= freq_high)
    fft_video[~mask] = 0
    return np.fft.ifft(fft_video, axis=0).real


def apply_evm_classic(
    frames_rgb,
    fps=100.0,
    freq_low=0.4,
    freq_high=3.0,
    alpha=12.0,
    levels=3,
    lambda_cutoff=16.0,
    chrom_attenuation=0.1,
):
    if len(frames_rgb) < 3:
        return frames_rgb

    frames_float = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
    frames_yiq = rgb_to_yiq(frames_float)

    pyramid_per_frame = [build_laplacian_pyramid(frame, levels) for frame in frames_yiq]
    num_bands = len(pyramid_per_frame[0])
    pyramid_video = []
    for band in range(num_bands):
        pyramid_video.append(np.stack([pyr[band] for pyr in pyramid_per_frame], axis=0))

    filtered_bands = [temporal_ideal_bandpass(band_video, fps, freq_low, freq_high) for band_video in pyramid_video]

    h, w = frames_float.shape[1:3]
    lambda_base = np.sqrt(float(h * h + w * w)) / 3.0
    delta = lambda_cutoff / (8.0 * (1.0 + alpha))
    amplified_bands = []

    for band_idx, filtered in enumerate(filtered_bands):
        if band_idx == 0 or band_idx == num_bands - 1:
            amplified_bands.append(np.zeros_like(filtered))
            continue

        lambda_band = lambda_base / (2 ** (num_bands - 1 - band_idx))
        current_alpha = lambda_band / delta / 8.0 - 1.0
        gain = max(0.0, min(alpha, current_alpha))
        amplified_bands.append(filtered * gain)

    amplified_frames = []
    for t in range(frames_float.shape[0]):
        amplified_pyramid = []
        for band_idx in range(num_bands):
            amplified_pyramid.append(pyramid_video[band_idx][t] + amplified_bands[band_idx][t])

        reconstructed = reconstruct_laplacian_pyramid(amplified_pyramid)
        delta_frame = reconstructed - frames_yiq[t]
        delta_frame[:, :, 1:] *= chrom_attenuation
        out_yiq = frames_yiq[t] + delta_frame
        out_rgb = yiq_to_rgb(out_yiq[None, ...])[0]
        out_rgb = np.clip(out_rgb, 0.0, 1.0)
        amplified_frames.append(np.uint8(np.round(out_rgb * 255.0)))

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
    levels,
    lambda_cutoff,
    chrom_attenuation,
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

    evm_frames = apply_evm_classic(
        [np.array(frame) for frame in cropped_frames],
        fps=fps,
        freq_low=freq_low,
        freq_high=freq_high,
        alpha=alpha,
        levels=levels,
        lambda_cutoff=lambda_cutoff,
        chrom_attenuation=chrom_attenuation,
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
        "levels": args.levels,
        "lambda_cutoff": args.lambda_cutoff,
        "chrom_attenuation": args.chrom_attenuation,
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
                    args.levels,
                    args.lambda_cutoff,
                    args.chrom_attenuation,
                    overwrite=args.overwrite,
                )
                summary[result["status"]] += 1

    print("Preprocessing completed!")
    summary_path = Path(args.output_rgb_dir).parent / "preprocess_smic_evm_classic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
