import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, labels=None):
    """
    Calculate various performance metrics for micro-expression recognition.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label indices
    
    Returns:
        Dictionary containing various metrics
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Calculate per-class metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    per_class_accuracy = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-10)
    
    # Calculate UAR (Unweighted Average Recall)
    uar = np.mean(per_class_accuracy)
    
    # Calculate UF1 (Unweighted F1-score)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = 2 * (per_class_precision * per_class_recall) / (per_class_precision + per_class_recall + 1e-10)
    uf1 = np.mean(per_class_f1)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'uar': uar,
        'uf1': uf1,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save the confusion matrix plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, class_names, save_path):
    """
    Save metrics to a text file.
    
    Args:
        metrics: Dictionary containing various metrics
        class_names: List of class names
        save_path: Path to save the metrics
    """
    with open(save_path, 'w') as f:
        f.write('Overall Metrics:\n')
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-score: {metrics['f1_score']:.4f}\n")
        f.write(f"UAR: {metrics['uar']:.4f}\n")
        f.write(f"UF1: {metrics['uf1']:.4f}\n\n")
        
        f.write('Per-class Metrics:\n')
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Accuracy: {metrics['per_class_accuracy'][i]:.4f}\n")
            f.write(f"  Precision: {metrics['per_class_precision'][i]:.4f}\n")
            f.write(f"  Recall: {metrics['per_class_recall'][i]:.4f}\n")
            f.write(f"  F1-score: {metrics['per_class_f1'][i]:.4f}\n")