import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


TARGET_FRAMES = 32
IMG_SIZE = (224, 224)
MARGIN = 20

EMOTION_MAP = {
    'happiness': 'positive',
    'disgust': 'negative',
    'repression': 'negative',
    'sadness': 'negative',
    'fear': 'negative',
    'surprise': 'surprise',
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Offline EVM preprocessing for CASME2: align/crop -> EVM -> TIM(32) -> dynamic image."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(project_root / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected"),
        help="Raw CASME2 root directory.",
    )
    parser.add_argument(
        "--excel-path",
        default=str(project_root / "data" / "CASME2_RAW_selected" / "CASME2-coding-20140508.xlsx"),
        help="CASME2 label excel path.",
    )
    parser.add_argument(
        "--output-rgb-dir",
        default=str(script_dir / "casme2_aligned_rgb_evm"),
        help="Output directory for EVM-processed aligned RGB sequences.",
    )
    parser.add_argument(
        "--output-dyn-dir",
        default=str(script_dir / "casme2_dynamic_data_evm"),
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
        default=200.0,
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
        default=8.0,
        help="Amplification factor for EVM.",
    )
    parser.add_argument(
        "--pyramid-level",
        type=int,
        default=1,
        help="Gaussian pyramid level used by EVM.",
    )
    parser.add_argument("--subject-min", type=int, default=None, help="Optional minimum CASME2 subject id to process.")
    parser.add_argument("--subject-max", type=int, default=None, help="Optional maximum CASME2 subject id to process.")
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

    coefficients = np.zeros(num_frames)
    for i in range(num_frames):
        coefficients[i] = np.sum([(2 * (j + 1) - num_frames - 1) / (j + 1) for j in range(i, num_frames)])

    coefficients /= np.sum(np.abs(coefficients))
    dynamic_image = np.zeros_like(image_sequence[0], dtype=np.float32)

    for i, frame in enumerate(image_sequence):
        dynamic_image += coefficients[i] * frame

    dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(dynamic_image)


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


def apply_evm(frames_rgb, fps=200.0, freq_low=0.4, freq_high=3.0, alpha=8.0, pyramid_level=1):
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
            frame_num = int(f[3:-4])
            if onset <= frame_num <= offset:
                valid_files.append(f)
        except Exception:
            continue
    return valid_files


def process_sequence(
    seq_path,
    subject,
    subject_num,
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
    pyramid_level,
    overwrite=False,
):
    valid_files = collect_valid_files(seq_path, onset, offset)
    if not valid_files:
        return {"status": "failed", "subject": subject, "sequence": seq_name, "reason": "no_frames"}

    out_seq_dir = os.path.join(output_rgb_dir, subject, seq_name)
    dyn_path = os.path.join(output_dyn_dir, subject, f"s{subject_num}_{seq_name}.jpg")
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

    evm_frames = apply_evm(
        [np.array(frame) for frame in cropped_frames],
        fps=fps,
        freq_low=freq_low,
        freq_high=freq_high,
        alpha=alpha,
        pyramid_level=pyramid_level,
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

    print("Loading CASME2 Excel Data...")
    df = pd.read_excel(args.excel_path)
    df_clean = df[df['Estimated Emotion'].notna()].copy()
    print(f"Total samples in annotation: {len(df_clean)}")

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
        "pyramid_level": args.pyramid_level,
        "subject_min": args.subject_min,
        "subject_max": args.subject_max,
        "total_sequences": 0,
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Processing CASME2 EVM"):
        subject_num = int(row['Subject'])
        if subject_num == 18:
            continue
        if args.subject_min is not None and subject_num < args.subject_min:
            continue
        if args.subject_max is not None and subject_num > args.subject_max:
            continue
        subject_id = f"sub{subject_num:02d}"
        seq_name = str(row['Filename'])
        raw_emotion = str(row['Estimated Emotion']).strip().lower()
        onset = int(row['OnsetFrame'])
        offset = int(row['OffsetFrame'])

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
            subject_num,
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
            args.pyramid_level,
            overwrite=args.overwrite,
        )
        summary[result["status"]] += 1

    summary_path = Path(args.output_rgb_dir).parent / "casme2_evm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
