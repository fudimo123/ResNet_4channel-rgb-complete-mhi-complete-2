import os
import cv2
import pandas as pd

# Paths
paths = {
    'CASME II': {
        'path': r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\casme2_fusion_train2_3class\dynamic_data',
        'fps': 200, # Standard for CASME II
        'parser_map': 'casme2'
    },
    'SMIC': {
        'path': r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\smic_fusion_train2\smic_dynamic_data2',
        'fps': 100, # Standard for SMIC-HS
        'parser_map': 'smic'
    },
    'DSME': {
        'path': r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\DSME_pic', # As per user instruction
        'fps': 25, # Standard for SMIC-VIS (DSME source)
        'parser_map': 'dsme'
    }
}

# Mapping rules (based on previous analysis of code)
# CASME II: 
#   happiness -> Positive
#   disgust, repression, sadness, fear -> Negative
#   surprise -> Surprise
#   others -> ? (Usually excluded or Negative depending on paper, code says class_names=['positive', 'negative', 'surprise'])
#   Let's check file names or structure. 
#   CASME II dynamic_data structure: sub01/s1_EP02_01f.jpg. 
#   We need the label from the filename or a mapping? 
#   Wait, the dynamic images in CASME II folder are just images. We might not know the emotion from the filename alone easily unless we have the excel.
#   However, the user wants the count "after resampling".
#   The `casme2_data_parser.py` uses `CASME2-coding-20140508.xlsx`.
#   Maybe I can just count the total and try to infer classes if possible, or just report Total if I can't classify easily without the excel.
#   BUT, for DSME I have the class folders.
#   For SMIC, `smic_dynamic_data2` has `s1/s1_s1_ne_01.jpg`. `ne` likely means Negative. `po` -> Positive. `sur` -> Surprise.
#   For CASME II, I might need to rely on the code's config or just report what I can.
#   Actually, the user asks for "Positive (n), Negative (n), Surprise (n)".
#   I should try to parse the emotions.

def get_casme2_emotion(filename):
    # This is hard without the excel. 
    # But wait, the `casme2_fusion_train2_3class` has a `fusion_train.py` that loads data.
    # It filters samples.
    # Let's try to count based on what we can find.
    # If I can't find the excel, I will approximate or search for a map file.
    pass

def analyze_dataset(name, info):
    root = info['path']
    if not os.path.exists(root):
        return None
    
    participants = set()
    pos_count = 0
    neg_count = 0
    sur_count = 0
    total_count = 0
    resolution = "N/A"
    
    # Walk through files
    for root_dir, dirs, files in os.walk(root):
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.bmp')):
                continue
            
            # Update resolution from the first image found
            if resolution == "N/A":
                img = cv2.imread(os.path.join(root_dir, file))
                if img is not None:
                    resolution = f"{img.shape[1]}x{img.shape[0]}"
            
            # Logic per dataset
            if name == 'DSME':
                # Structure: Subject/Emotion/Sequence/img.jpg
                # We need to count sequences, not frames!
                # In DSME_pic: 001/fear/1/img1.jpg. "1" is the sequence.
                # So we count the parent folder of the image.
                pass 
            elif name == 'SMIC':
                # Structure: s1/s1_s1_ne_01.jpg
                # Each jpg is a dynamic image (one per sequence).
                pass
            elif name == 'CASME II':
                # Structure: sub01/s1_EP02_01f.jpg
                # Each jpg is a dynamic image.
                pass
    
    # Re-implement counting logic properly
    if name == 'DSME':
        # Count sequences (folders)
        # DSME_pic/Subject/Emotion/Sequence/*.jpg
        # We need to count the *Sequence* folders.
        subjects = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        participants = set(subjects)
        
        for sub in subjects:
            sub_path = os.path.join(root, sub)
            emotions = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
            for emo in emotions:
                emo_clean = emo.lower().replace('hapiness', 'happiness').replace('digust', 'disgust').replace('suprise', 'surprise')
                
                # Map to 3 classes
                cls = None
                if emo_clean in ['happiness']:
                    cls = 'Positive'
                elif emo_clean in ['disgust', 'fear', 'sadness', 'repression', 'anger']:
                    cls = 'Negative'
                elif emo_clean in ['surprise']:
                    cls = 'Surprise'
                
                if cls:
                    seq_path = os.path.join(sub_path, emo)
                    # Count subfolders
                    seqs = [d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))]
                    count = len(seqs)
                    if cls == 'Positive': pos_count += count
                    elif cls == 'Negative': neg_count += count
                    elif cls == 'Surprise': sur_count += count
                    total_count += count

    elif name == 'SMIC':
        # smic_dynamic_data2/s1/s1_s1_ne_01.jpg
        # File naming convention: s{subject}_s{subject}_{emotion}_{seq_id}.jpg
        # emotion: ne (negative), po (positive), sur (surprise)
        subjects = os.listdir(root)
        for sub in subjects:
            if not sub.startswith('s'): continue
            participants.add(sub)
            sub_path = os.path.join(root, sub)
            if not os.path.isdir(sub_path): continue
            
            files = [f for f in os.listdir(sub_path) if f.endswith('.jpg')]
            for f in files:
                # Parse filename
                # s1_s1_ne_01.jpg
                parts = f.split('_')
                if len(parts) >= 3:
                    emo_code = parts[2] # ne, po, sur
                    if 'po' in emo_code:
                        pos_count += 1
                    elif 'ne' in emo_code:
                        neg_count += 1
                    elif 'sur' in emo_code:
                        sur_count += 1
                    total_count += 1

    elif name == 'CASME II':
        # dynamic_data/sub01/s1_EP02_01f.jpg
        # We need to map filenames to emotions.
        # Since we don't have the excel loaded, let's use the code's `casme2_data_parser.py` if possible.
        # Alternatively, assume the user only cares about DSME if we can't get CASME II easily.
        # But wait, `casme2_fusion_train2_3class` has `fusion_train.py` which prints `all_samples`.
        # I can write a script to import the parsers and get the counts! This is much more accurate.
        pass
        
    return {
        'Dataset': name,
        'Participants': len(participants),
        'Resolution': resolution,
        'FPS': info['fps'],
        'Positive (n)': pos_count,
        'Negative (n)': neg_count,
        'Surprise (n)': sur_count,
        'Total': total_count
    }

def main():
    # We will use the existing parsers to get accurate counts for CASME II and SMIC
    # For DSME, we stick to the manual count logic since the parser might be tricky to import if paths are relative.
    # Actually, we can try to import all if we set up paths correctly.
    pass

if __name__ == '__main__':
    # Print what we found
    pass
