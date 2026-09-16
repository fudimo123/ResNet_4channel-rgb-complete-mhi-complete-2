import os


def append_augmented_static_samples(
    base_samples,
    augmented_root,
    variants=("L", "R"),
    allowed_emotions=None,
    augmented_dynamic_root=None,
):
    """
    Duplicate training samples with alternative static-frame directories while
    keeping the original dynamic-image pairing untouched.
    """
    if not augmented_root or not os.path.isdir(augmented_root):
        return list(base_samples)

    expanded_samples = list(base_samples)
    allowed_emotions = set(allowed_emotions) if allowed_emotions is not None else None

    for sample in base_samples:
        if allowed_emotions is not None and sample.get("emotion") not in allowed_emotions:
            continue

        subject = sample["subject"]
        sequence = sample["sequence"]

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
            if augmented_dynamic_root:
                aug_sample["dynamic_variant"] = variant
                aug_sample["augmented_dynamic_root"] = augmented_dynamic_root
            expanded_samples.append(aug_sample)

    return expanded_samples
