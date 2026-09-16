import os
from casme2_data_parser import CASME2DataParser
from samm_data_parser import SAMMDataParser
from smic_data_parser import SMICDataParser

# Use relative paths to the cleaned data directories we just created
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cleaned CASME2 Data
CASME2_ANNO = os.path.join(BASE_DIR, 'casme2_fusion_train_clean', 'casme2_aligned_rgb', 'casme2_clean_labels.csv')
CASME2_VIDEO_DIR = os.path.join(BASE_DIR, 'casme2_fusion_train_clean', 'casme2_aligned_rgb')

# Cleaned SAMM Data
SAMM_ANNO = os.path.join(BASE_DIR, 'SAMM_fusion_train_clean', 'samm_aligned_rgb', 'samm_clean_labels.csv')
SAMM_VIDEO_DIR = os.path.join(BASE_DIR, 'SAMM_fusion_train_clean', 'samm_aligned_rgb')

# Cleaned SMIC Data
SMIC_ANNO = os.path.join(BASE_DIR, 'smic_fusion_train_clean', 'smic_aligned_rgb', 'smic_clean_labels.csv')
SMIC_VIDEO_DIR = os.path.join(BASE_DIR, 'smic_fusion_train_clean', 'smic_aligned_rgb')

# Check if cleaned paths exist (SMIC doesn't need an annotation CSV, so we only check CASME2 and SAMM)
for path in [CASME2_ANNO, SAMM_ANNO]:
    if not os.path.exists(path):
        print(f"Warning: Cleaned annotation not found at {path}")

# Emotion Maps for Cleaned Data (They are already simplified to 3 classes in the clean CSVs)
CLEAN_EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}

class CombinedDataParser:
    def __init__(self):
        self.samples = []
        self._load_casme2()
        self._load_samm()
        self._load_smic()

    def _load_casme2(self):
        print("Loading Cleaned CASME II samples...")
        try:
            parser = CASME2DataParser(CASME2_ANNO, CASME2_VIDEO_DIR, CLEAN_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                s['dataset'] = 'casme2'
                s['original_subject'] = s['subject']
                s['subject'] = f"casme2_{s['subject']}"
            self.samples.extend(samples)
            print(f"Added {len(samples)} CASME II samples.")
        except Exception as e:
            print(f"Error loading CASME II: {e}")

    def _load_samm(self):
        print("Loading Cleaned SAMM samples...")
        try:
            parser = SAMMDataParser(SAMM_ANNO, SAMM_VIDEO_DIR, CLEAN_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                s['dataset'] = 'samm'
                s['original_subject'] = s['subject']
                s['subject'] = f"samm_{s['subject']}"
            self.samples.extend(samples)
            print(f"Added {len(samples)} SAMM samples.")
        except Exception as e:
            print(f"Error loading SAMM: {e}")

    def _load_smic(self):
        print("Loading Cleaned SMIC samples...")
        try:
            # SMICDataParser in clean folder expects (aligned_rgb_dir, emotion_map) - it doesn't need a CSV
            parser = SMICDataParser(SMIC_VIDEO_DIR, CLEAN_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                s['dataset'] = 'smic'
                # original_subject is something like 's1'
                s['original_subject'] = s['subject']
                # remove the 's' prefix for original subject if needed, or keep it. We'll just prefix it to 'smic_s1'
                s['subject'] = f"smic_{s['subject']}"
                # For compatibility with fusion_dataset which expects 'video_name'
                s['video_name'] = s['sequence']
            self.samples.extend(samples)
            print(f"Added {len(samples)} SMIC samples.")
        except Exception as e:
            print(f"Error loading SMIC: {e}")

    def get_samples(self):
        return self.samples

    def get_all_subjects(self):
        return sorted(list(set(s['subject'] for s in self.samples)))

if __name__ == "__main__":
    parser = CombinedDataParser()
    print(f"Total combined samples: {len(parser.get_samples())}")
    subjects = parser.get_all_subjects()
    print(f"Total unique subjects: {len(subjects)}")
    print(f"Subjects: {subjects}")
