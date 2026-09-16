import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Updated data for CASME II
casme2_data = np.array([
    [28.00, 3.33, 0.67],
    [3.00, 95.33, 0.67],
    [0.33, 3.67, 21.00]
])

class_names = ['Positive', 'Negative', 'Surprise']
output_dir = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization'
filename = 'Confusion_Matrix_CASME2_New.png'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set style
sns.set(style='white')

# Row Normalization
row_sums = casme2_data.sum(axis=1, keepdims=True)
norm_matrix = np.divide(casme2_data, row_sums, where=row_sums!=0)

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
title = 'CASME II Dataset (3-class)'
plt.figtext(0.5, 0.02, title, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')

# Save
save_path = os.path.join(output_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated normalized confusion matrix: {save_path}")
