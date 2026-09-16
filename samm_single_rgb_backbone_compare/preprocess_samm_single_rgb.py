import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "SAMM"
EXCEL_PATH = RAW_DATA_DIR / "SAMM_Micro_FACS_Codes_v2.xlsx"
OUTPUT_RGB_DIR = Path(__file__).resolve().parent / "samm_aligned_rgb_offline"
MANIFEST_FILE = Path(__file__).resolve().parent / "samm_single_rgb_manifest.csv"
SUMMARY_FILE = Path(__file__).resolve().parent / "samm_single_rgb_manifest_summary.json"
TARGET_FRAMES = 32
IMG_SIZE = (224, 224)
MARGIN = 20

EMOTION_MAP = {
    "Happiness": 0,
    "Anger": 1,
    "Sadness": 1,
    "Disgust": 1,
    "Fear": 1,
    "Contempt": 1,
    "Surprise": 2,
}
EMOTION_NAME_MAP = {0: "positive", 1: "negative", 2: "surprise"}


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
    valid_files = []
    for file_name in sorted([f for f in os.listdir(seq_path) if f.lower().endswith(".jpg")]):
        try:
            frame_num = int(file_name.split("_")[-1].split(".")[0])
        except Exception:
            continue
        if onset <= frame_num <= offset:
            valid_files.append(file_name)
    return valid_files


def existing_complete(out_seq_dir):
    if not out_seq_dir.exists():
        return False, []
    existing = sorted([f.name for f in out_seq_dir.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"])
    return len(existing) == TARGET_FRAMES, existing


def process_sequence(seq_path, out_seq_dir, valid_files, mtcnn):
    frame_pils = []
    for file_name in valid_files:
        try:
            frame_pils.append(Image.open(seq_path / file_name).convert("RGB"))
        except Exception:
            continue

    if not frame_pils:
        return False

    box = None
    for image in frame_pils[: min(5, len(frame_pils))]:
        box = get_face_box(mtcnn, image)
        if box is not None:
            break
    if box is None:
        box = (0, 0, frame_pils[0].width, frame_pils[0].height)

    cropped_frames = []
    for image in frame_pils:
        crop = image.crop(box)
        crop = crop.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        cropped_frames.append(crop)

    indices = np.linspace(0, len(cropped_frames) - 1, TARGET_FRAMES).astype(int)
    tim_frames = [cropped_frames[i] for i in indices]

    out_seq_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(tim_frames):
        frame.save(out_seq_dir / f"img_{idx:04d}.jpg")
    return True


def main():
    print("Loading SAMM Excel data...")
    df = pd.read_excel(EXCEL_PATH, skiprows=13)
    df_clean = df[df["Estimated Emotion"] != "Other"].copy()
    OUTPUT_RGB_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing MTCNN on {device}...")
    mtcnn = MTCNN(keep_all=False, device=device)

    manifest_rows = []
    done = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df_clean.iterrows(), total=len(df_clean), desc="Preprocessing SAMM single RGB"):
        raw_emotion = str(row["Estimated Emotion"]).strip()
        if raw_emotion not in EMOTION_MAP:
            continue
        if pd.isna(row["Onset Frame"]) or pd.isna(row["Offset Frame"]):
            failed += 1
            continue

        subject = str(row["Subject"]).zfill(3)
        sequence = str(row["Filename"])
        onset = int(row["Onset Frame"])
        offset = int(row["Offset Frame"])
        seq_path = RAW_DATA_DIR / subject / sequence
        if not seq_path.exists():
            failed += 1
            continue

        valid_files = collect_valid_files(seq_path, onset, offset)
        if not valid_files:
            failed += 1
            continue

        out_seq_dir = OUTPUT_RGB_DIR / subject / sequence
        complete, existing_files = existing_complete(out_seq_dir)
        if complete:
            frame_count = len(existing_files)
            skipped += 1
        else:
            success = process_sequence(seq_path, out_seq_dir, valid_files, mtcnn)
            if not success:
                failed += 1
                continue
            frame_count = TARGET_FRAMES
            done += 1

        manifest_rows.append(
            {
                "subject": subject,
                "sequence": sequence,
                "emotion": EMOTION_NAME_MAP[EMOTION_MAP[raw_emotion]],
                "label": EMOTION_MAP[raw_emotion],
                "onset": onset,
                "offset": offset,
                "frame_count": frame_count,
                "processed_rgb_path": str(Path(subject) / sequence),
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
