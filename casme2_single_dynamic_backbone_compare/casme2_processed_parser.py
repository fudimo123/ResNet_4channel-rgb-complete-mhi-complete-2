import os

import pandas as pd


class CASME2ProcessedDataParser:
    def __init__(self, manifest_file):
        self.manifest_file = manifest_file
        self.df = pd.read_csv(manifest_file)
        self.subjects = sorted(self.df["subject"].unique().tolist())

    def get_samples(self, subjects=None):
        target_df = self.df
        if subjects is not None:
            target_df = self.df[self.df["subject"].isin(subjects)]

        path_key = None
        for candidate in ("processed_dynamic_path", "processed_rgb_path"):
            if candidate in target_df.columns:
                path_key = candidate
                break

        samples = []
        for _, row in target_df.iterrows():
            sample = {
                "subject": int(row["subject"]),
                "sequence": str(row["sequence"]),
                "onset": int(row["onset"]),
                "apex": int(row["apex"]) if not pd.isna(row["apex"]) else -1,
                "offset": int(row["offset"]),
                "emotion": str(row["emotion"]),
                "label": int(row["label"]),
            }
            if path_key is not None and not pd.isna(row[path_key]):
                sample[path_key] = os.path.normpath(
                    str(row[path_key]).replace("\\", os.sep).replace("/", os.sep)
                )
            samples.append(sample)
        return samples

    def get_all_subjects(self):
        return [subject for subject in self.subjects if subject != 18]
