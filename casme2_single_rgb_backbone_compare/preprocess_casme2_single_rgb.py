import json
import os
import re
from pathlib import Path

import pandas as pd
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected"
EXCEL_PATH = PROJECT_ROOT / "data" / "CASME2_RAW_selected" / "CASME2-coding-20140508.xlsx"
OUTPUT_RGB_DIR = Path(__file__).resolve().parent / "casme2_aligned_rgb_offline"
MANIFEST_FILE = Path(__file__).resolve().parent / "casme2_single_rgb_manifest.csv"
SUMMARY_FILE = Path(__file__).resolve().parent / "casme2_single_rgb_manifest_summary.json"
IMG_SIZE = (224, 224)
MARGIN = 20

EMOTION_MAP = {
    "happiness": 0,
    "disgust": 1,
    "repression": 1,
    "sadness": 1,
    "fear": 1,
    "surprise": 2,
}


def get_face_box(mtcnn, img_pil):
    boxes, _ = mtcnn.detect(img_pil)
    if boxes is None:
        return None

    box = boxes[0]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1 - MARGIN)
    y1 = max(0, y1 - MARGIN)
    x2 = min(img_pil.width, x2 + MARGIN)
    y2 = min(img_pil.height, y2 + MARGIN)
    return x1, y1, x2, y2


def collect_valid_files(seq_path, onset, offset):
    all_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(".jpg")])
    valid_files = []
    for file_name in all_files:
        match = re.search(r"img(\d+)\.jpg", file_name, re.IGNORECASE)
        if match is None:
            continue
        frame_num = int(match.group(1))
        if onset <= frame_num <= offset:
            valid_files.append(file_name)
    return valid_files


def load_frames(seq_path, valid_files):
    frames = []
    for file_name in valid_files:
        path = seq_path / file_name
        try:
            frames.append(Image.open(path).convert("RGB"))
        except Exception:
            continue
    return frames


def process_sequence(seq_path, out_seq_dir, valid_files, mtcnn):
    frames = load_frames(seq_path, valid_files)
    if not frames:
        return False, "no_frames"

    box = None
    for img in frames[: min(5, len(frames))]:
        box = get_face_box(mtcnn, img)
        if box is not None:
            break
    if box is None:
        box = (0, 0, frames[0].width, frames[0].height)

    out_seq_dir.mkdir(parents=True, exist_ok=True)
    for img, file_name in zip(frames, valid_files):
        crop = img.crop(box)
        crop = crop.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        crop.save(out_seq_dir / file_name)
    return True, "done"


def existing_complete(out_seq_dir, valid_files):
    if not out_seq_dir.exists():
        return False
    existing = sorted([f.name for f in out_seq_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
    return existing == valid_files


def parse_optional_int(value, default=-1):
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main():
    print("Loading CASME2 Excel data...")
    df = pd.read_excel(EXCEL_PATH)
    OUTPUT_RGB_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing MTCNN on {device}...")
    mtcnn = MTCNN(keep_all=False, device=device)

    manifest_rows = []
    done = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing CASME2 single RGB"):
        if pd.isna(row["OnsetFrame"]) or pd.isna(row["OffsetFrame"]):
            continue

        raw_emotion = str(row["Estimated Emotion"]).strip()
        if raw_emotion not in EMOTION_MAP:
            continue

        subject = int(row["Subject"])
        sequence = str(row["Filename"])
        onset = int(row["OnsetFrame"])
        apex = parse_optional_int(row["ApexFrame"], default=-1)
        offset = int(row["OffsetFrame"])

        seq_path = RAW_DATA_DIR / f"sub{subject:02d}" / sequence
        if not seq_path.exists():
            failed += 1
            continue

        valid_files = collect_valid_files(seq_path, onset, offset)
        if not valid_files:
            failed += 1
            continue

        out_seq_dir = OUTPUT_RGB_DIR / f"sub{subject:02d}" / sequence
        if existing_complete(out_seq_dir, valid_files):
            skipped += 1
        else:
            success, _ = process_sequence(seq_path, out_seq_dir, valid_files, mtcnn)
            if not success:
                failed += 1
                continue
            done += 1

        rel_processed_dir = Path(f"sub{subject:02d}") / sequence
        manifest_rows.append(
            {
                "subject": subject,
                "sequence": sequence,
                "emotion": raw_emotion,
                "label": EMOTION_MAP[raw_emotion],
                "onset": onset,
                "apex": apex,
                "offset": offset,
                "frame_count": len(valid_files),
                "processed_rgb_path": str(rel_processed_dir),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["subject", "sequence"]).reset_index(drop=True)
    manifest_df.to_csv(MANIFEST_FILE, index=False, encoding="utf-8")

    summary = {
        "output_rgb_dir": str(OUTPUT_RGB_DIR),
        "manifest_file": str(MANIFEST_FILE),
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "total_manifest_rows": len(manifest_rows),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
