import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm


TARGET_FRAMES = 32
IMG_SIZE = (224, 224)
MARGIN = 20

EMOTION_MAP = {
    'Happiness': 'positive',
    'Anger': 'negative',
    'Sadness': 'negative',
    'Disgust': 'negative',
    'Fear': 'negative',
    'Contempt': 'negative',
    'Surprise': 'surprise'
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Offline classic EVM preprocessing for SAMM: align/crop -> classic EVM -> TIM(32) -> dynamic image."
    )
    parser.add_argument("--raw-dir", default=str(project_root / "data" / "SAMM"))
    parser.add_argument("--excel-path", default=str(project_root / "data" / "SAMM" / "SAMM_Micro_FACS_Codes_v2.xlsx"))
    parser.add_argument("--output-rgb-dir", default=str(script_dir / "samm_aligned_rgb_evm_classic"))
    parser.add_argument("--output-dyn-dir", default=str(script_dir / "samm_dynamic_data_evm_classic"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--freq-low", type=float, default=0.4)
    parser.add_argument("--freq-high", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--lambda-cutoff", type=float, default=16.0)
    parser.add_argument("--chrom-attenuation", type=float, default=0.1)
    parser.add_argument("--filter-type", default="butterworth", choices=("butterworth", "ideal"))
    parser.add_argument("--butter-order", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate_dynamic_image(image_sequence):
    num_frames = len(image_sequence)
    if num_frames == 0:
        return None
    coefficients = np.zeros(num_frames)
    for i in range(num_frames):
        coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])
    coefficients /= np.sum(np.abs(coefficients))
    dynamic_image = np.zeros_like(image_sequence[0], dtype=np.float32)
    for i, frame in enumerate(image_sequence):
        dynamic_image += coefficients[i] * frame
    dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(dynamic_image)


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


def temporal_butter_bandpass(video, fps, freq_low, freq_high, order):
    nyquist = 0.5 * fps
    low = max(freq_low / nyquist, 1e-6)
    high = min(freq_high / nyquist, 0.999999)
    if low >= high:
        raise ValueError(f"Invalid bandpass range: low={freq_low}, high={freq_high}, fps={fps}")
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, video, axis=0)


def apply_evm_classic(
    frames_rgb,
    fps=200.0,
    freq_low=0.4,
    freq_high=3.0,
    alpha=10.0,
    levels=3,
    lambda_cutoff=16.0,
    chrom_attenuation=0.1,
    filter_type="butterworth",
    butter_order=1,
):
    if len(frames_rgb) < 3:
        return frames_rgb

    frames_float = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0
    frames_yiq = rgb_to_yiq(frames_float)
    pyramid_per_frame = [build_laplacian_pyramid(frame, levels) for frame in frames_yiq]
    num_bands = len(pyramid_per_frame[0])
    pyramid_video = [np.stack([pyr[band] for pyr in pyramid_per_frame], axis=0) for band in range(num_bands)]

    filtered_bands = []
    for band_video in pyramid_video:
        if filter_type == "butterworth":
            filtered = temporal_butter_bandpass(band_video, fps, freq_low, freq_high, butter_order)
        else:
            filtered = temporal_ideal_bandpass(band_video, fps, freq_low, freq_high)
        filtered_bands.append(filtered)

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
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(img_pil.width, x2 + MARGIN)
        y2 = min(img_pil.height, y2 + MARGIN)
        return x1, y1, x2, y2
    return None


def collect_valid_files(seq_path, onset, offset):
    all_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith('.jpg')])
    valid_files = []
    for f in all_files:
        try:
            frame_num = int(f.split('_')[-1].split('.')[0])
            if onset <= frame_num <= offset:
                valid_files.append(f)
        except Exception:
            continue
    return valid_files


def process_sequence(
    seq_path,
    subject,
    seq_name,
    onset,
    offset,
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
    filter_type,
    butter_order,
    overwrite=False,
):
    valid_files = collect_valid_files(seq_path, onset, offset)
    if not valid_files:
        return {"status": "failed", "subject": subject, "sequence": seq_name, "reason": "no_frames"}

    out_seq_dir = os.path.join(output_rgb_dir, subject, seq_name)
    dyn_path = os.path.join(output_dyn_dir, subject, f"s{subject}_{seq_name}.jpg")
    if not overwrite and os.path.isdir(out_seq_dir) and os.path.exists(dyn_path):
        return {"status": "skipped", "subject": subject, "sequence": seq_name}

    frame_pils = []
    for f in valid_files:
        path = os.path.join(seq_path, f)
        try:
            frame_pils.append(Image.open(path).convert('RGB'))
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if not frame_pils:
        return {"status": "failed", "subject": subject, "sequence": seq_name, "reason": "read_failed"}

    box = None
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

    evm_frames = apply_evm_classic(
        [np.array(frame) for frame in cropped_frames],
        fps=fps,
        freq_low=freq_low,
        freq_high=freq_high,
        alpha=alpha,
        levels=levels,
        lambda_cutoff=lambda_cutoff,
        chrom_attenuation=chrom_attenuation,
        filter_type=filter_type,
        butter_order=butter_order,
    )

    indices = np.linspace(0, len(evm_frames) - 1, TARGET_FRAMES).astype(int)
    tim_frames = [Image.fromarray(evm_frames[i]) for i in indices]

    os.makedirs(out_seq_dir, exist_ok=True)
    cv2_frames = []
    for i, frame in enumerate(tim_frames):
        frame.save(os.path.join(out_seq_dir, f"img_{i:04d}.jpg"))
        cv2_frames.append(np.array(frame))

    dyn_img = generate_dynamic_image(cv2_frames)
    if dyn_img is not None:
        out_sub_dyn_dir = os.path.join(output_dyn_dir, subject)
        os.makedirs(out_sub_dyn_dir, exist_ok=True)
        dyn_img_bgr = cv2.cvtColor(dyn_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dyn_path, dyn_img_bgr)

    return {"status": "done", "subject": subject, "sequence": seq_name}


def main():
    args = parse_args()

    print("Loading SAMM Excel Data...")
    df = pd.read_excel(args.excel_path, skiprows=13)
    df_clean = df[df['Estimated Emotion'] != 'Other'].copy()
    print(f"Total samples after removing 'Other': {len(df_clean)}")

    print("Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=False, device=args.device)

    os.makedirs(args.output_rgb_dir, exist_ok=True)
    os.makedirs(args.output_dyn_dir, exist_ok=True)

    summary = {
        "raw_dir": os.path.abspath(args.raw_dir),
        "excel_path": os.path.abspath(args.excel_path),
        "output_rgb_dir": os.path.abspath(args.output_rgb_dir),
        "output_dyn_dir": os.path.abspath(args.output_dyn_dir),
        "fps": args.fps,
        "freq_low": args.freq_low,
        "freq_high": args.freq_high,
        "alpha": args.alpha,
        "levels": args.levels,
        "lambda_cutoff": args.lambda_cutoff,
        "chrom_attenuation": args.chrom_attenuation,
        "filter_type": args.filter_type,
        "butter_order": args.butter_order,
        "total_sequences": 0,
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing SAMM Classic EVM"):
        subject_id = str(row['Subject']).zfill(3)
        seq_name = str(row['Filename'])
        raw_emotion = str(row['Estimated Emotion']).strip()
        onset = int(row['Onset Frame'])
        offset = int(row['Offset Frame'])

        mapped_emotion = EMOTION_MAP.get(raw_emotion)
        if mapped_emotion is None:
            continue

        seq_path = os.path.join(args.raw_dir, subject_id, seq_name)
        if not os.path.exists(seq_path):
            print(f"Sequence missing: {seq_path}")
            continue

        summary["total_sequences"] += 1
        result = process_sequence(
            seq_path,
            subject_id,
            seq_name,
            onset,
            offset,
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
            args.filter_type,
            args.butter_order,
            overwrite=args.overwrite,
        )
        summary[result["status"]] += 1

    summary_path = Path(args.output_rgb_dir).parent / "samm_classic_evm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
