import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build offline O/L/R static Decalcomanie samples from preprocessed CASME2 RGB sequences."
    )
    parser.add_argument(
        "--input-dir",
        default=str(script_dir / "casme2_aligned_rgb"),
        help="Input directory containing preprocessed CASME2 RGB sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "casme2_static_decalcomanie"),
        help="Output directory for O/L/R samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild output even if the target sequence already exists.",
    )
    return parser.parse_args()


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


def load_sequence_frames(sequence_dir):
    frame_paths = sorted(
        p for p in Path(sequence_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    frames = []
    for frame_path in frame_paths:
        try:
            frames.append(Image.open(frame_path).convert("RGB"))
        except Exception as exc:
            print(f"Skip unreadable frame {frame_path}: {exc}")
    return frames


def save_variant_frames(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(os.path.join(output_dir, f"img_{idx:04d}.jpg"), quality=95)


def output_exists(output_root, subject, sequence, expected_count):
    for variant in ("O", "L", "R"):
        variant_dir = Path(output_root) / variant / subject / sequence
        if not variant_dir.is_dir():
            return False
        count = len(list(variant_dir.glob("*.jpg")))
        if count != expected_count:
            return False
    return True


def process_one_sequence(subject_dir, sequence_dir, output_root, overwrite=False):
    subject = subject_dir.name
    sequence = sequence_dir.name

    frame_pils = load_sequence_frames(sequence_dir)
    if not frame_pils:
        return {"status": "failed", "subject": subject, "sequence": sequence, "reason": "no_frames"}

    if not overwrite and output_exists(output_root, subject, sequence, len(frame_pils)):
        return {"status": "skipped", "subject": subject, "sequence": sequence}

    left_frames = [make_left_variant(frame) for frame in frame_pils]
    right_frames = [make_right_variant(frame) for frame in frame_pils]

    save_variant_frames(frame_pils, Path(output_root) / "O" / subject / sequence)
    save_variant_frames(left_frames, Path(output_root) / "L" / subject / sequence)
    save_variant_frames(right_frames, Path(output_root) / "R" / subject / sequence)
    return {"status": "done", "subject": subject, "sequence": sequence}


def find_sequences(input_dir):
    input_root = Path(input_dir)
    sequences = []
    for subject_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        for sequence_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            sequences.append((subject_dir, sequence_dir))
    return sequences


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sequences = find_sequences(args.input_dir)
    summary = {
        "input_dir": os.path.abspath(args.input_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "total_sequences": len(sequences),
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    print(f"Found {len(sequences)} CASME2 sequences under {args.input_dir}")
    for idx, (subject_dir, sequence_dir) in enumerate(sequences, start=1):
        print(f"[{idx}/{len(sequences)}] {subject_dir.name} / {sequence_dir.name}")
        result = process_one_sequence(
            subject_dir,
            sequence_dir,
            args.output_dir,
            overwrite=args.overwrite,
        )
        summary[result["status"]] += 1

    summary_path = Path(args.output_dir) / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Done.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
