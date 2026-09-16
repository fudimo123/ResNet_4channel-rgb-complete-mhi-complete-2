import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Data from latest experiments
datasets = {
    'DSME': {
        'matrix': np.array([
            [28.00, 4.00, 0.00],
            [2.67, 77.67, 3.67],
            [1.00, 7.00, 20.00]
        ]),
        'title': 'DSME Dataset (3-class)',
        'filename': 'Confusion_Matrix_DSME_New.png'
    },
    'SAMM': {
        'matrix': np.array([
            [17.00, 8.33, 0.67],
            [3.00, 88.67, 0.33],
            [1.00, 5.67, 8.33]
        ]),
        'title': 'SAMM Dataset (3-class)',
        'filename': 'Confusion_Matrix_SAMM_New.png'
    },
    'SMIC': {
        'matrix': np.array([
            [32.33, 14.67, 4.00],
            [10.33, 51.67, 8.00],
            [5.00, 8.33, 29.67]
        ]),
        'title': 'SMIC Dataset (3-class)',
        'filename': 'Confusion_Matrix_SMIC_New.png'
    }
}

class_names = ['Positive', 'Negative', 'Surprise']
output_dir = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set style
sns.set(style='white')

for name, data in datasets.items():
    matrix = data['matrix']
    title = data['title']
    filename = data['filename']
    
    # Row Normalization
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(matrix, row_sums, where=row_sums!=0)
    
    plt.figure(figsize=(6, 5))
    
    # Plot heatmap
    ax = sns.heatmap(norm_matrix, annot=True, fmt='.2f', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 14}, cbar=True, vmin=0, vmax=1)
    
    # Set labels
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Add title at bottom
    plt.figtext(0.5, 0.02, title, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')
    
    # Save
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated normalized confusion matrix for {name}: {save_path}")
