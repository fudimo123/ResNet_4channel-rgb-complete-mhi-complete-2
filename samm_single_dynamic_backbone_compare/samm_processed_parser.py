import os

import pandas as pd


class SAMMProcessedDataParser:
    def __init__(self, manifest_file):
        self.manifest_file = manifest_file
        self.df = pd.read_csv(manifest_file, dtype={"subject": str})
        self.df["subject"] = self.df["subject"].astype(str).str.zfill(3)
        self.subjects = sorted(self.df["subject"].unique().tolist())

    def get_samples(self, subjects=None):
        target_df = self.df
        if subjects is not None:
            subject_set = {str(subject).zfill(3) for subject in subjects}
            target_df = self.df[self.df["subject"].isin(subject_set)]

        samples = []
        for _, row in target_df.iterrows():
            samples.append(
                {
                    "subject": str(row["subject"]).zfill(3),
                    "sequence": str(row["sequence"]),
                    "emotion": str(row["emotion"]),
                    "label": int(row["label"]),
                    "processed_dynamic_path": os.path.normpath(
                        str(row["processed_dynamic_path"]).replace("\\", os.sep).replace("/", os.sep)
                    ),
                }
            )
        return samples

    def get_all_subjects(self):
        return self.subjects
