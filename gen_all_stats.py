import os
import sys
import pandas as pd
import cv2

# Add paths to sys.path to import parsers
base_path = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2'
sys.path.append(os.path.join(base_path, 'casme2_fusion_train2_3class'))
sys.path.append(os.path.join(base_path, 'smic_fusion_train2'))

from casme2_data_parser import CASME2DataParser
from smic_data_parser import SMICDataParser

def get_resolution(path, is_dir_of_images=True):
    # Find first image
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.bmp') or f.endswith('.png'):
                img = cv2.imread(os.path.join(root, f))
                if img is not None:
                    return f"{img.shape[1]}x{img.shape[0]}"
    return "N/A"

def count_dsme():
    root = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\DSME_pic'
    subjects = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    
    pos = 0
    neg = 0
    sur = 0
    
    for sub in subjects:
        sub_path = os.path.join(root, sub)
        emotions = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]
        for emo in emotions:
            emo_clean = emo.lower().replace('hapiness', 'happiness').replace('digust', 'disgust').replace('suprise', 'surprise')
            seq_path = os.path.join(sub_path, emo)
            count = len([d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))])
            
            if emo_clean == 'happiness':
                pos += count
            elif emo_clean in ['disgust', 'fear', 'sadness']:
                neg += count
            elif emo_clean == 'surprise':
                sur += count
                
    res = get_resolution(root)
    return {
        'Dataset': 'DSME (Self-Collected)',
        'Participants': len(subjects),
        'Resolution': '640x480 (Upsampled)', # Based on typical webcam/processing, will check actual
        'FPS': '25 (->75)',
        'Positive (n)': pos,
        'Negative (n)': neg,
        'Surprise (n)': sur,
        'Total': pos + neg + sur
    }

def count_casme2():
    anno_file = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2-coding-20140508.xlsx'
    raw_dir = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected'
    
    emotion_map = {
        'happiness': 0, 
        'disgust': 1, 'repression': 1, 'sadness': 1, 'fear': 1,
        'surprise': 2
    }
    
    try:
        parser = CASME2DataParser(anno_file, raw_dir, emotion_map)
        samples = parser.get_samples()
        subjects = parser.get_all_subjects()
        
        pos = sum(1 for s in samples if s['label'] == 0)
        neg = sum(1 for s in samples if s['label'] == 1)
        sur = sum(1 for s in samples if s['label'] == 2)
        
        return {
            'Dataset': 'CASME II',
            'Participants': len(subjects),
            'Resolution': '640x480',
            'FPS': 200,
            'Positive (n)': pos,
            'Negative (n)': neg,
            'Surprise (n)': sur,
            'Total': len(samples)
        }
    except Exception as e:
        print(f"CASME2 Error: {e}")
        return None

def count_smic():
    raw_dir = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\SMIC2'
    emotion_map = {'positive': 0, 'negative': 1, 'surprise': 2}
    
    try:
        parser = SMICDataParser(raw_dir, emotion_map)
        samples = parser.get_samples()
        subjects = parser.get_all_subjects()
        
        pos = sum(1 for s in samples if s['label'] == 0)
        neg = sum(1 for s in samples if s['label'] == 1)
        sur = sum(1 for s in samples if s['label'] == 2)
        
        return {
            'Dataset': 'SMIC (HS)',
            'Participants': len(subjects),
            'Resolution': '640x480',
            'FPS': 100,
            'Positive (n)': pos,
            'Negative (n)': neg,
            'Surprise (n)': sur,
            'Total': len(samples)
        }
    except Exception as e:
        print(f"SMIC Error: {e}")
        return None

def main():
    rows = []
    
    # 1. CASME II
    c2 = count_casme2()
    if c2: rows.append(c2)
    
    # 2. SMIC
    sm = count_smic()
    if sm: rows.append(sm)
    
    # 3. DSME
    ds = count_dsme()
    # Check actual resolution of DSME images
    ds_res = get_resolution(r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\DSME_pic')
    ds['Resolution'] = ds_res
    rows.append(ds)
    
    df = pd.DataFrame(rows)
    df = df[['Dataset', 'Participants', 'Resolution', 'FPS', 'Positive (n)', 'Negative (n)', 'Surprise (n)', 'Total']]
    
    out_path = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\All_Datasets_Statistics.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(df)

if __name__ == '__main__':
    main()
