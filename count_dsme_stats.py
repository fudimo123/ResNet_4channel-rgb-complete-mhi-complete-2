import os
import csv
import pandas as pd

# Path to the dataset
dataset_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\DSME_pic'
output_csv_path = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\DSME_Dataset_Statistics.csv'

# Emotion mapping
# Standardize raw emotion names first
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

# Map to 3 classes
class_map = {
    'happiness': 'Positive',
    'disgust': 'Negative',
    'fear': 'Negative',
    'sadness': 'Negative',
    'surprise': 'Surprise'
}

def generate_statistics():
    stats = {} # {subject: {'Positive': 0, 'Negative': 0, 'Surprise': 0}}
    
    # Traverse the dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return

    subjects = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for subject in subjects:
        stats[subject] = {'Positive': 0, 'Negative': 0, 'Surprise': 0}
        subject_path = os.path.join(dataset_path, subject)
        
        emotions = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
        
        for emotion_raw in emotions:
            emotion_clean = emotion_correction.get(emotion_raw.lower(), emotion_raw.lower())
            
            if emotion_clean not in class_map:
                print(f"Warning: Unknown emotion '{emotion_raw}' for subject {subject}")
                continue
                
            emotion_class = class_map[emotion_clean]
            
            # Count sequences (folders inside emotion folder)
            emotion_path = os.path.join(subject_path, emotion_raw)
            sequences = [d for d in os.listdir(emotion_path) if os.path.isdir(os.path.join(emotion_path, d))]
            count = len(sequences)
            
            stats[subject][emotion_class] += count

    # Create DataFrame
    data = []
    total_positive = 0
    total_negative = 0
    total_surprise = 0
    total_all = 0

    for subject in subjects:
        pos = stats[subject]['Positive']
        neg = stats[subject]['Negative']
        sur = stats[subject]['Surprise']
        total = pos + neg + sur
        
        data.append({
            'Subject ID': subject,
            'Positive (Happiness)': pos,
            'Negative (Disgust, Fear, Sadness)': neg,
            'Surprise': sur,
            'Total': total
        })
        
        total_positive += pos
        total_negative += neg
        total_surprise += sur
        total_all += total

    # Add Total row
    data.append({
        'Subject ID': 'Total',
        'Positive (Happiness)': total_positive,
        'Negative (Disgust, Fear, Sadness)': total_negative,
        'Surprise': total_surprise,
        'Total': total_all
    })

    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Statistics saved to {output_csv_path}")
    print(df)

if __name__ == '__main__':
    generate_statistics()
