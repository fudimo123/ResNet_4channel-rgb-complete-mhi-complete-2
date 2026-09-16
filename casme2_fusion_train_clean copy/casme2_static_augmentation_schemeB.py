import os
import random


def append_augmented_static_samples(
    base_samples,
    augmented_root,
    variants=("L", "R"),
    allowed_emotions=None,
    max_variants_per_sample=1,
):
    if not augmented_root or not os.path.isdir(augmented_root):
        return list(base_samples)

    expanded_samples = list(base_samples)
    allowed_emotions = set(allowed_emotions) if allowed_emotions is not None else None

    for sample in base_samples:
        emotion = sample.get("emotion")
        if allowed_emotions is not None and emotion not in allowed_emotions:
            continue

        subject = sample["subject"]
        sequence = sample["sequence"]
        candidate_variants = []

        for variant in variants:
            variant_dir = os.path.join(augmented_root, variant, subject, sequence)
            if not os.path.isdir(variant_dir):
                continue

            image_files = sorted(
                f for f in os.listdir(variant_dir) if f.lower().endswith(".jpg")
            )
            if not image_files:
                continue

            aug_sample = dict(sample)
            aug_sample["aligned_path"] = variant_dir
            aug_sample["image_files"] = image_files
            aug_sample["static_variant"] = variant
            candidate_variants.append(aug_sample)

        if not candidate_variants:
            continue

        keep_count = min(max_variants_per_sample, len(candidate_variants))
        expanded_samples.extend(random.sample(candidate_variants, k=keep_count))

    return expanded_samples
