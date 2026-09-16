import os
import shutil
import glob
from casme2_data_parser import CASME2DataParser
from samm_data_parser import SAMMDataParser
from smic_data_parser import SMICDataParser

# Import necessary data parsers
# Assuming these files are copied to the current directory or accessible
# We will need to copy them first

# Paths - Updated to absolute paths based on user environment
# CASME2_ANNO = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2-coding-20140508.xlsx'
# CASME2_VIDEO_DIR = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected'

# SAMM_ANNO = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SAMM\SAMM_Micro_FACS_Codes_v2.xlsx'
# SAMM_VIDEO_DIR = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SAMM'

# SMIC_VIDEO_DIR = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SMIC2'

# Use relative paths for cloud environment compatibility, assuming data is in a 'data' folder relative to project root
# Or better, check environment variables or try default locations.
# Let's set default relative paths that assume the data structure is preserved.

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

CASME2_ANNO = os.path.join(DATA_DIR, 'CASME2_RAW_selected', 'CASME2-coding-20140508.xlsx')
CASME2_VIDEO_DIR = os.path.join(DATA_DIR, 'CASME2_RAW_selected', 'CASME2_RAW_selected')

SAMM_ANNO = os.path.join(DATA_DIR, 'SAMM', 'SAMM_Micro_FACS_Codes_v2.xlsx')
SAMM_VIDEO_DIR = os.path.join(DATA_DIR, 'SAMM')

SMIC_VIDEO_DIR = os.path.join(DATA_DIR, 'SMIC2')

# Fallback for local testing if needed, or if user environment is specific
if not os.path.exists(CASME2_ANNO):
    # Try the absolute path from user's local machine just in case (though won't work on cloud)
    # Better: Print a warning if paths don't exist
    print(f"Warning: CASME2 annotation not found at {CASME2_ANNO}")

if not os.path.exists(SAMM_ANNO):
    print(f"Warning: SAMM annotation not found at {SAMM_ANNO}")
    
if not os.path.exists(SMIC_VIDEO_DIR):
    print(f"Warning: SMIC video dir not found at {SMIC_VIDEO_DIR}")


# Emotion Maps
CASME2_EMOTION_MAP = {
    'happiness': 0,
    'disgust': 1,
    'repression': 1,
    'surprise': 2
}

SAMM_EMOTION_MAP = {
    'Happiness': 0,
    'Anger': 1,
    'Disgust': 1,
    'Fear': 1,
    'Sadness': 1,
    'Contempt': 1,
    'Surprise': 2
}

SMIC_EMOTION_MAP = {
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
        print("Loading CASME II samples...")
        try:
            parser = CASME2DataParser(CASME2_ANNO, CASME2_VIDEO_DIR, CASME2_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                # Add dataset prefix to subject to ensure uniqueness
                # CASME2 subjects are ints 1-26
                s['dataset'] = 'casme2'
                s['original_subject'] = s['subject']
                s['subject'] = f"casme2_{s['subject']}"
                # Construct dynamic filename expectation
                # CASME2 dynamic file format in merge_data.py: casme2_{filename}
                # But wait, CASME2 dynamic files are named like 'sub01_EP02_01f.jpg' inside 'sub01' folder
                # In merge_data.py, we copy all files and prefix with 'casme2_'
                # So 'sub01/sub01_EP02_01f.jpg' becomes 'casme2_sub01_EP02_01f.jpg'
                
                # We need to match this.
                # In CASME2 parser, 'sequence' is like 'EP02_01f'
                # We need to know the exact filename of the dynamic image. 
                # Usually it is f"sub{subject:02d}_{sequence}.jpg"
                # Let's verify standard naming convention or search logic in Dataset
                pass
            self.samples.extend(samples)
            print(f"Added {len(samples)} CASME II samples.")
        except Exception as e:
            print(f"Error loading CASME II: {e}")

    def _load_samm(self):
        print("Loading SAMM samples...")
        try:
            parser = SAMMDataParser(SAMM_ANNO, SAMM_VIDEO_DIR, SAMM_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                s['dataset'] = 'samm'
                s['original_subject'] = s['subject'] # string '006'
                s['subject'] = f"samm_{s['subject']}"
            self.samples.extend(samples)
            print(f"Added {len(samples)} SAMM samples.")
        except Exception as e:
            print(f"Error loading SAMM: {e}")

    def _load_smic(self):
        print("Loading SMIC samples...")
        try:
            parser = SMICDataParser(SMIC_VIDEO_DIR, SMIC_EMOTION_MAP)
            samples = parser.get_samples()
            for s in samples:
                s['dataset'] = 'smic'
                s['original_subject'] = s['subject'] # int 1
                s['subject'] = f"smic_{s['subject']}"
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
