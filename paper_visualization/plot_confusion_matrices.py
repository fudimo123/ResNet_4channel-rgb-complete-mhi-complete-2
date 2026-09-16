import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# 配置数据 (原始 Count 数据)
datasets = {
    'SAMM': {
        'matrix': np.array([[17.00, 6.67, 2.33],
                           [5.67, 81.33, 5.00],
                           [2.00, 5.67, 7.33]]),
        'title': 'SAMM Dataset (3-class)',
        'filename': 'Confusion_Matrix_SAMM.png'
    },
    'SMIC': {
        'matrix': np.array([[29.00, 16.67, 5.33],
                           [5.00, 58.67, 6.33],
                           [1.67, 12.67, 28.67]]),
        'title': 'SMIC Dataset (3-class)',
        'filename': 'Confusion_Matrix_SMIC.png'
    },
    'DSME': {
        'matrix': np.array([[27.00, 4.67, 0.33],
                           [1.00, 77.00, 6.00],
                           [0.67, 6.00, 21.33]]),
        'title': 'DSME Dataset (3-class)',
        'filename': 'Confusion_Matrix_DSME.png'
    },
    'CASME II': {
        'matrix': np.array([[26.67, 4.67, 0.67],
                           [3.67, 94.00, 1.33],
                           [0.33, 3.33, 21.33]]),
        'title': 'CASME II Dataset (3-class)',
        'filename': 'Confusion_Matrix_CASME2.png'
    }
}

class_names = ['Positive', 'Negative', 'Surprise']
output_dir = r'D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘图风格设置
sns.set(style='white')

for name, data in datasets.items():
    matrix = data['matrix']
    title = data['title']
    filename = data['filename']
    
    # --- 关键修正：按行归一化 (Row Normalization) ---
    # axis=1 表示按行求和，keepdims=True 保持维度以便广播除法
    row_sums = matrix.sum(axis=1, keepdims=True)
    # 避免除以0 (虽然这里不太可能)
    norm_matrix = np.divide(matrix, row_sums, where=row_sums!=0)
    
    plt.figure(figsize=(6, 5))
    
    # 使用 Blues 色系
    # fmt='.2f' 保留两位小数 (0.xx 格式)
    ax = sns.heatmap(norm_matrix, annot=True, fmt='.2f', cmap='Blues', 
                     xticklabels=class_names, yticklabels=class_names,
                     annot_kws={"size": 14}, cbar=True, vmin=0, vmax=1) # vmin/vmax 确保颜色范围一致
    
    # 设置轴标签
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # 底部标题
    plt.figtext(0.5, 0.02, title, wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated normalized confusion matrix for {name}")
