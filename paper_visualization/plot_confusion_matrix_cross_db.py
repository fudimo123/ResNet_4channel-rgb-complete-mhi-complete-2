import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Data from Cross-DB Fusion Experiment
# D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\cross_db_fusion_train\fusion_result_cross_db\final_metrics.txt
cross_db_data = np.array([
    [77.33, 24.67, 7.00],
    [24.33, 213.00, 14.67],
    [8.67, 17.00, 57.33]
])

class_names = ['Positive', 'Negative', 'Surprise']
output_dir = r'd:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization'
filename = 'Confusion_Matrix_CrossDB.png'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set style
sns.set(style='white')

# Row Normalization
row_sums = cross_db_data.sum(axis=1, keepdims=True)
norm_matrix = np.divide(cross_db_data, row_sums, where=row_sums!=0)

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
title = 'Cross-Database Fusion (3-class)'
plt.figtext(0.5, 0.02, title, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')

# Save
save_path = os.path.join(output_dir, filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated normalized confusion matrix for Cross-DB: {save_path}")
