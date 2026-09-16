import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


DEFAULT_CLASS_NAMES = ["positive", "negative", "surprise"]


def calculate_metrics(y_true, y_pred, labels=None, class_names=None):
    """
    Calculate overall and per-class metrics with consistent definitions.

    Overall block:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - F1 score (weighted)
    - UAR (macro recall)
    - UF1 (macro F1)
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if labels is None:
        labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    uar = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    uf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_accuracy = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-10)
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": per_class_precision[i] if i < len(per_class_precision) else 0.0,
            "recall": per_class_recall[i] if i < len(per_class_recall) else 0.0,
            "f1-score": per_class_f1[i] if i < len(per_class_f1) else 0.0,
        }

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "uar": uar,
        "uf1": uf1,
        "per_class_accuracy": per_class_accuracy,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix,
    }


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
