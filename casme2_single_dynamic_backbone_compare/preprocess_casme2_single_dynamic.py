import json
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from dynamic_image_generator import DynamicImageGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "CASME2_RAW_selected" / "CASME2_RAW_selected"
EXCEL_PATH = PROJECT_ROOT / "data" / "CASME2_RAW_selected" / "CASME2-coding-20140508.xlsx"
OUTPUT_DYNAMIC_DIR = Path(__file__).resolve().parent / "casme2_dynamic_images_offline"
MANIFEST_FILE = Path(__file__).resolve().parent / "casme2_single_dynamic_manifest.csv"
SUMMARY_FILE = Path(__file__).resolve().parent / "casme2_single_dynamic_manifest_summary.json"

EMOTION_MAP = {
    "happiness": 0,
    "disgust": 1,
    "repression": 1,
    "sadness": 1,
    "fear": 1,
    "surprise": 2,
}


def parse_optional_int(value, default=-1):
    if pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_image_sequence(sequence_path, onset, offset):
    if not sequence_path.exists():
        return []

    images = []
    for frame_idx in range(onset, offset + 1):
        image_path = sequence_path / f"img{frame_idx}.jpg"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is not None:
            images.append(image)
    return images


def existing_complete(output_path):
    if not output_path.exists():
        return False
    image = cv2.imread(str(output_path))
    return image is not None and image.size > 0


def main():
    print("Loading CASME2 Excel data...")
    df = pd.read_excel(EXCEL_PATH)
    OUTPUT_DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    generator = DynamicImageGenerator(str(OUTPUT_DYNAMIC_DIR))

    manifest_rows = []
    done = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing CASME2 single dynamic"):
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

        rel_output_path = Path(f"sub{subject:02d}") / f"s{subject}_{sequence}.jpg"
        full_output_path = OUTPUT_DYNAMIC_DIR / rel_output_path
        full_output_path.parent.mkdir(parents=True, exist_ok=True)

        frame_count = max(offset - onset + 1, 0)
        if existing_complete(full_output_path):
            skipped += 1
        else:
            image_sequence = load_image_sequence(seq_path, onset, offset)
            if len(image_sequence) == 0:
                failed += 1
                continue

            dynamic_image = generator.generate_dynamic_image(image_sequence)
            if dynamic_image is None:
                failed += 1
                continue

            generator.save_dynamic_image(subject, sequence, dynamic_image)
            if not existing_complete(full_output_path):
                failed += 1
                continue
            done += 1

        manifest_rows.append(
            {
                "subject": subject,
                "sequence": sequence,
                "emotion": raw_emotion,
                "label": EMOTION_MAP[raw_emotion],
                "onset": onset,
                "apex": apex,
                "offset": offset,
                "frame_count": frame_count,
                "processed_dynamic_path": str(rel_output_path),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["subject", "sequence"]).reset_index(drop=True)
    manifest_df.to_csv(MANIFEST_FILE, index=False, encoding="utf-8")

    summary = {
        "output_dynamic_dir": str(OUTPUT_DYNAMIC_DIR),
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
