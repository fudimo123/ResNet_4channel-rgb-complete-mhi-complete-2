import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix():
    # 1. 输入你的实验结果 (Mean Matrix)
    # 顺序对应: [Positive, Negative, Surprise]
    cm_mean = np.array([
        [26.67, 4.67, 0.67],  # Positive (真实标签)
        [3.67, 94.00, 1.33],  # Negative (真实标签)
        [0.33, 3.33, 21.33]   # Surprise (真实标签)
    ])

    # 2. 标签定义 (与你的数据顺序一致)
    classes = ['Positive', 'Negative', 'Surprise']

    # 3. 计算归一化矩阵 (按行求和，计算百分比)
    # 这一步会将数值转换为Recall，例如 26.67 / 32.01 = 0.833
    cm_norm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis]

    # 4. 设置绘图风格
    plt.figure(figsize=(8, 6), dpi=300) # 设置画布大小和分辨率
    sns.set(font_scale=1.2) # 调整字体大小
    
    # 5. 绘制热力图
    # annot=True: 显示数值
    # fmt='.2%': 数值格式为百分比 (保留两位小数)
    # cmap='Blues': 颜色主题 (蓝色系，学术界常用)
    # cbar=True: 显示颜色条
    ax = sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     square=True, linewidths=1, linecolor='black', # 格子边框
                     cbar_kws={"label": "Accuracy (Recall)"})

    # 6. 设置标题和坐标轴标签
    plt.title('Confusion Matrix (Normalized)', fontsize=15, pad=20, weight='bold')
    plt.xlabel('Predicted Label', fontsize=13, labelpad=10)
    plt.ylabel('True Label', fontsize=13, labelpad=10)

    # 7. 调整布局并保存
    plt.tight_layout()
    plt.savefig('Confusion_Matrix_Result.png', bbox_inches='tight')
    plt.show()

    print(f"矩阵对角线验证 (应与Recall一致):")
    print(f"Positive: {cm_norm[0][0]:.4f}")
    print(f"Negative: {cm_norm[1][1]:.4f}")
    print(f"Surprise: {cm_norm[2][2]:.4f}")

if __name__ == '__main__':
    plot_confusion_matrix()