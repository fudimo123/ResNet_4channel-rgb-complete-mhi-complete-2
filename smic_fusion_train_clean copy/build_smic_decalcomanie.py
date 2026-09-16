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
IMAGE_EXTS = (".bmp", ".jpg", ".jpeg", ".png")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Build offline O/L/R static Decalcomanie samples from raw SMIC2."
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
        "--output-dir",
        default=str(script_dir / "smic_static_decalcomanie"),
        help="Output directory for O/L/R samples.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=TARGET_FRAMES,
        help="Number of TIM-sampled frames per sequence.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for MTCNN face detection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild output even if the target sequence already exists.",
    )
    return parser.parse_args()


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


def find_sequence_dirs(raw_dir, valid_sequences=None):
    sequences = []
    raw_root = Path(raw_dir)

    for subject_dir in sorted(raw_root.glob("s*")):
        micro_dir = subject_dir / "micro"
        if not micro_dir.is_dir():
            continue

        for emotion_dir in sorted(p for p in micro_dir.iterdir() if p.is_dir()):
            for seq_dir in sorted(p for p in emotion_dir.iterdir() if p.is_dir()):
                if valid_sequences is not None and seq_dir.name not in valid_sequences:
                    continue

                sequences.append(
                    {
                        "subject": subject_dir.name,
                        "emotion": emotion_dir.name,
                        "sequence": seq_dir.name,
                        "sequence_dir": seq_dir,
                    }
                )

    return sequences


def load_sequence_frames(seq_dir):
    frame_paths = sorted(
        p for p in Path(seq_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    frames = []
    for frame_path in frame_paths:
        try:
            frames.append(Image.open(frame_path).convert("RGB"))
        except Exception as exc:
            print(f"Skip unreadable frame {frame_path}: {exc}")
    return frames


def get_face_box(mtcnn, img_pil):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        return None

    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    x1 = max(0, x1 - MARGIN)
    y1 = max(0, y1 - MARGIN)
    x2 = min(img_pil.width, x2 + MARGIN)
    y2 = min(img_pil.height, y2 + MARGIN)
    return x1, y1, x2, y2


def preprocess_sequence(frame_pils, mtcnn, target_frames):
    box = None
    for img in frame_pils[: min(5, len(frame_pils))]:
        box = get_face_box(mtcnn, img)
        if box is not None:
            break

    if box is None:
        box = (0, 0, frame_pils[0].width, frame_pils[0].height)

    cropped_frames = []
    for img in frame_pils:
        cropped = img.crop(box)
        cropped = cropped.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        cropped_frames.append(cropped)

    if len(cropped_frames) == 1:
        return [cropped_frames[0]] * target_frames

    indices = np.linspace(0, len(cropped_frames) - 1, target_frames).astype(int)
    return [cropped_frames[idx] for idx in indices]


def make_left_variant(image):
    arr = np.array(image)
    width = arr.shape[1]
    mid = width // 2
    left = arr[:, :mid, :]
    mirrored_left = np.flip(left, axis=1)
    if width % 2 == 0:
        out = np.concatenate([left, mirrored_left], axis=1)
    else:
        center = arr[:, mid : mid + 1, :]
        out = np.concatenate([left, center, mirrored_left], axis=1)
    return Image.fromarray(out)


def make_right_variant(image):
    arr = np.array(image)
    width = arr.shape[1]
    mid = width // 2
    if width % 2 == 0:
        right = arr[:, mid:, :]
        mirrored_right = np.flip(right, axis=1)
        out = np.concatenate([mirrored_right, right], axis=1)
    else:
        center = arr[:, mid : mid + 1, :]
        right = arr[:, mid + 1 :, :]
        mirrored_right = np.flip(right, axis=1)
        out = np.concatenate([mirrored_right, center, right], axis=1)
    return Image.fromarray(out)


def save_variant_frames(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(output_dir, f"img_{idx:04d}.jpg"), quality=95)


def output_exists(output_root, subject, sequence, target_frames):
    for variant in ("O", "L", "R"):
        variant_dir = Path(output_root) / variant / subject / sequence
        if not variant_dir.is_dir():
            return False
        count = len(list(variant_dir.glob("*.jpg")))
        if count != target_frames:
            return False
    return True


def process_one_sequence(seq_info, mtcnn, output_root, target_frames, overwrite=False):
    subject = seq_info["subject"]
    sequence = seq_info["sequence"]

    if not overwrite and output_exists(output_root, subject, sequence, target_frames):
        return {"status": "skipped", "subject": subject, "sequence": sequence}

    frame_pils = load_sequence_frames(seq_info["sequence_dir"])
    if not frame_pils:
        return {"status": "failed", "subject": subject, "sequence": sequence, "reason": "no_frames"}

    sampled_frames = preprocess_sequence(frame_pils, mtcnn, target_frames)
    left_frames = [make_left_variant(frame) for frame in sampled_frames]
    right_frames = [make_right_variant(frame) for frame in sampled_frames]

    save_variant_frames(sampled_frames, Path(output_root) / "O" / subject / sequence)
    save_variant_frames(left_frames, Path(output_root) / "L" / subject / sequence)
    save_variant_frames(right_frames, Path(output_root) / "R" / subject / sequence)

    return {"status": "done", "subject": subject, "sequence": sequence}


def main():
    args = parse_args()

    valid_sequences = load_valid_micro_sequences(args.xlsx_path)
    sequences = find_sequence_dirs(args.raw_dir, valid_sequences=valid_sequences)

    os.makedirs(args.output_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=False, device=args.device)

    summary = {
        "raw_dir": os.path.abspath(args.raw_dir),
        "xlsx_path": os.path.abspath(args.xlsx_path),
        "output_dir": os.path.abspath(args.output_dir),
        "target_frames": args.target_frames,
        "total_sequences": len(sequences),
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    print(f"Found {len(sequences)} micro sequences under {args.raw_dir}")
    for idx, seq_info in enumerate(sequences, start=1):
        print(f"[{idx}/{len(sequences)}] {seq_info['subject']} / {seq_info['sequence']}")
        result = process_one_sequence(
            seq_info,
            mtcnn,
            args.output_dir,
            args.target_frames,
            overwrite=args.overwrite,
        )
        summary[result["status"]] += 1

    summary_path = Path(args.output_dir) / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
