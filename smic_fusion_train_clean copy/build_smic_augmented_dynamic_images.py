import argparse
import json
import os
from pathlib import Path

import cv2

from dynamic_image_generator import DynamicImageGenerator


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build dynamic images for L/R Decalcomanie static sequences."
    )
    parser.add_argument(
        "--static-aug-dir",
        default=str(script_dir / "smic_static_decalcomanie"),
        help="Root directory containing O/L/R augmented static frame sequences.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "smic_dynamic_data_decalcomanie"),
        help="Output directory for augmented dynamic images.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["L", "R"],
        help="Augmented variants to convert into dynamic images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate output images even if they already exist.",
    )
    return parser.parse_args()


def load_image_sequence(sequence_dir):
    frame_paths = sorted(
        p for p in Path(sequence_dir).iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )
    images = []
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is not None:
            images.append(img)
    return images


def save_dynamic_image(output_root, variant, subject, sequence, dynamic_image):
    variant_dir = Path(output_root) / variant / subject
    variant_dir.mkdir(parents=True, exist_ok=True)
    output_path = variant_dir / f"{subject}_{sequence}.jpg"
    cv2.imwrite(str(output_path), dynamic_image)
    return output_path


def main():
    args = parse_args()

    static_aug_root = Path(args.static_aug_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    generator = DynamicImageGenerator(str(output_root))
    summary = {
        "static_aug_dir": str(static_aug_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "variants": args.variants,
        "done": 0,
        "skipped": 0,
        "failed": 0,
    }

    for variant in args.variants:
        variant_root = static_aug_root / variant
        if not variant_root.is_dir():
            continue

        for subject_dir in sorted(p for p in variant_root.iterdir() if p.is_dir()):
            subject = subject_dir.name
            for sequence_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
                sequence = sequence_dir.name
                output_path = output_root / variant / subject / f"{subject}_{sequence}.jpg"

                if output_path.exists() and not args.overwrite:
                    summary["skipped"] += 1
                    continue

                image_sequence = load_image_sequence(sequence_dir)
                if not image_sequence:
                    summary["failed"] += 1
                    print(f"Skip empty sequence: {variant}/{subject}/{sequence}")
                    continue

                dynamic_image = generator.generate_dynamic_image(image_sequence)
                if dynamic_image is None:
                    summary["failed"] += 1
                    print(f"Failed to generate dynamic image: {variant}/{subject}/{sequence}")
                    continue

                save_dynamic_image(output_root, variant, subject, sequence, dynamic_image)
                summary["done"] += 1

    summary_path = output_root / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
