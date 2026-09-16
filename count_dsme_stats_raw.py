import os
import csv
import pandas as pd

# Path to the dataset
dataset_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\DSME_pic'
output_csv_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\DSME_Dataset_Statistics_Raw.csv'

# Standardize raw emotion names (fix typos only, do NOT map to 3 classes)
emotion_correction = {
    'hapiness': 'happiness',
    'digust': 'disgust',
    'suprise': 'surprise',
    'happiness': 'happiness',
    'disgust': 'disgust',
    'surprise': 'surprise',
    'fear': 'fear',
    'sadness': 'sadness'
}

def generate_statistics():
    # 1. Discover all unique emotions first
    all_emotions = set()
    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    if not subjects:
        print(f"No subjects found in {dataset_path}")
        return

    # First pass: collect all emotion categories
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        raw_emotion_folders = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        
        for raw_emo in raw_emotion_folders:
            # Fix typos but keep original category logic
            clean_emo = emotion_correction.get(raw_emo.lower(), raw_emo.lower())
            all_emotions.add(clean_emo)
    
    sorted_emotions = sorted(list(all_emotions))
    print(f"Found emotions: {sorted_emotions}")

    # 2. Count samples per subject per emotion
    data = []
    
    # Initialize totals counters
    totals = {emo: 0 for emo in sorted_emotions}
    total_all_subjects = 0

    for subject in subjects:
        row = {'Subject ID': subject}
        subject_total = 0
        subject_path = os.path.join(dataset_path, subject)
        
        # Initialize counts for this subject
        for emo in sorted_emotions:
            row[emo.capitalize()] = 0

        # Iterate through actual folders
        raw_emotion_folders = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        
        for raw_emo in raw_emotion_folders:
            clean_emo = emotion_correction.get(raw_emo.lower(), raw_emo.lower())
            
            if clean_emo not in sorted_emotions:
                continue
                
            emotion_path = os.path.join(subject_path, raw_emo)
            # Count subdirectories (video sequences)
            sequences = [d for d in os.listdir(emotion_path) if os.path.isdir(os.path.join(emotion_path, d))]
            count = len(sequences)
            
            # Add to row
            row[clean_emo.capitalize()] += count
            
            # Add to totals
            totals[clean_emo] += count
            subject_total += count
            
        row['Total'] = subject_total
        total_all_subjects += subject_total
        data.append(row)

    # 3. Add Total Row
    total_row = {'Subject ID': 'Total'}
    for emo in sorted_emotions:
        total_row[emo.capitalize()] = totals[emo]
    total_row['Total'] = total_all_subjects
    data.append(total_row)

    # Create DataFrame
    # Define column order: Subject ID, [Emotions...], Total
    columns = ['Subject ID'] + [e.capitalize() for e in sorted_emotions] + ['Total']
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Statistics saved to {output_csv_path}")
    print(df)

if __name__ == '__main__':
    generate_statistics()
