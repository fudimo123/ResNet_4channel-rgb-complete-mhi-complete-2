import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm


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
        description="Offline aligned RGB preprocessing for CASME2."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(project_root / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected"),
        help="CASME2 raw root directory.",
    )
    parser.add_argument(
        "--excel-path",
        default=str(project_root / "data" / "CASME2_RAW_selected" / "CASME2-coding-20140508.xlsx"),
        help="CASME2 annotation excel path.",
    )
    parser.add_argument(
        "--output-rgb-dir",
        default=str(script_dir / "casme2_aligned_rgb"),
        help="Output directory for offline aligned RGB sequences.",
    )
    parser.add_argument(
        "--mapping-file",
        default=str(script_dir / "casme2_clean_labels.csv"),
        help="Output CSV manifest for offline aligned training.",
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


def collect_valid_files(seq_path, onset, offset):
    valid_files = []
    for file_name in sorted(os.listdir(seq_path)):
        if not file_name.lower().endswith('.jpg'):
            continue
        if not file_name.startswith('img'):
            continue
        try:
            frame_num = int(file_name[3:-4])
        except ValueError:
            continue
        if onset <= frame_num <= offset:
            valid_files.append(file_name)
    return valid_files


def preprocess_sequence(frame_pils, mtcnn):
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
    return cropped_frames


def process_sequence(sample, mtcnn, output_root, overwrite=False):
    subject_num = sample['subject_num']
    subject = sample['subject']
    sequence = sample['sequence']
    seq_path = sample['sequence_path']
    onset = sample['onset']
    offset = sample['offset']

    valid_files = collect_valid_files(seq_path, onset, offset)
    if not valid_files:
        return {"status": "failed", "subject": subject, "sequence": sequence, "reason": "no_frames"}

    out_seq_dir = os.path.join(output_root, subject, sequence)
    if not overwrite and os.path.isdir(out_seq_dir):
        existing = [f for f in os.listdir(out_seq_dir) if f.lower().endswith('.jpg')]
        if len(existing) == len(valid_files):
            return {"status": "skipped", "subject": subject, "sequence": sequence}

    frame_pils = []
    for file_name in valid_files:
        path = os.path.join(seq_path, file_name)
        try:
            frame_pils.append(Image.open(path).convert('RGB'))
        except Exception as exc:
            print(f"Error reading {path}: {exc}")

    if not frame_pils:
        return {"status": "failed", "subject": subject, "sequence": sequence, "reason": "read_failed"}

    cropped_frames = preprocess_sequence(frame_pils, mtcnn)
    os.makedirs(out_seq_dir, exist_ok=True)
    for idx, frame in enumerate(cropped_frames):
        frame.save(os.path.join(out_seq_dir, f"img_{idx:04d}.jpg"), quality=95)

    return {
        "status": "done",
        "subject": subject,
        "subject_num": subject_num,
        "sequence": sequence,
        "emotion": sample['emotion'],
        "label": sample['emotion_clean'],
        "num_frames": len(cropped_frames),
    }


def build_samples(excel_path, raw_dir):
    df = pd.read_excel(excel_path)
    samples = []
    for _, row in df.iterrows():
        emotion = str(row['Estimated Emotion']).strip().lower()
        if emotion not in EMOTION_MAP:
            continue

        subject_num = int(row['Subject'])
        if subject_num == 18:
            continue

        sequence = str(row['Filename']).strip()
        subject = f"sub{subject_num:02d}"
        seq_path = os.path.join(raw_dir, subject, sequence)
        if not os.path.isdir(seq_path):
            continue

        apex_value = row.get('ApexFrame', None)
        try:
            apex_value = int(apex_value)
        except (TypeError, ValueError):
            apex_value = None

        samples.append(
            {
                'subject_num': subject_num,
                'subject': subject,
                'sequence': sequence,
                'emotion': emotion,
                'emotion_clean': EMOTION_MAP[emotion],
                'onset': int(row['OnsetFrame']),
                'apex': apex_value,
                'offset': int(row['OffsetFrame']),
                'sequence_path': seq_path,
            }
        )
    return samples


def main():
    args = parse_args()
    os.makedirs(args.output_rgb_dir, exist_ok=True)
    mtcnn = MTCNN(keep_all=False, device=args.device)

    samples = build_samples(args.excel_path, args.raw_dir)
    summary = {
        "raw_dir": os.path.abspath(args.raw_dir),
        "excel_path": os.path.abspath(args.excel_path),
        "output_rgb_dir": os.path.abspath(args.output_rgb_dir),
        "mapping_file": os.path.abspath(args.mapping_file),
        "total_sequences": len(samples),
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }
    manifest_rows = []

    for sample in tqdm(samples, desc="Building CASME2 aligned RGB"):
        result = process_sequence(sample, mtcnn, args.output_rgb_dir, overwrite=args.overwrite)
        summary[result["status"]] += 1
        if result["status"] in {"done", "skipped"}:
            manifest_rows.append(
                {
                    "subject_num": sample['subject_num'],
                    "subject": sample['subject'],
                    "sequence": sample['sequence'],
                    "emotion": sample['emotion_clean'],
                    "label": ['positive', 'negative', 'surprise'].index(sample['emotion_clean']),
                    "num_frames": result.get("num_frames", -1),
                }
            )

    pd.DataFrame(manifest_rows).to_csv(args.mapping_file, index=False)
    summary_path = Path(args.output_rgb_dir) / "preprocess_casme2_aligned_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
