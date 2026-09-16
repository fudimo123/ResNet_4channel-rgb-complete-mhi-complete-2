import math
import os
import random


def append_augmented_static_samples(
    base_samples,
    augmented_root,
    variants=("L", "R"),
    allowed_emotions=None,
    emotion_variant_ratios=None,
    default_ratio=1.0,
):
    """
    Duplicate training samples with alternative static-frame directories while
    keeping the original dynamic-image pairing untouched.
    """
    if not augmented_root or not os.path.isdir(augmented_root):
        return list(base_samples)

    expanded_samples = list(base_samples)
    allowed_emotions = set(allowed_emotions) if allowed_emotions is not None else None
    emotion_variant_ratios = emotion_variant_ratios or {}

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

        ratio = emotion_variant_ratios.get(emotion, default_ratio)
        keep_count = int(math.floor(len(candidate_variants) * ratio + 1e-8))
        keep_count = max(0, min(len(candidate_variants), keep_count))

        if keep_count == 0:
            continue
        if keep_count < len(candidate_variants):
            candidate_variants = random.sample(candidate_variants, k=keep_count)

        expanded_samples.extend(candidate_variants)

    return expanded_samples
