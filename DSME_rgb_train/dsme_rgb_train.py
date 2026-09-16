import os
import sys
import json
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import matplotlib.pyplot as plt

# Import custom modules
try:
    from model import resnet18
    from dataset import DSMERGBDataset
    from dsme_rgb_data_parser import DSMERGBDataParser
    from focal_loss import FocalLoss
    from early_stopping import EarlyStopping
    from metrics import calculate_metrics, plot_confusion_matrix
except ImportError:
    # Fallback for package run
    from .model import resnet18
    from .dataset import DSMERGBDataset
    from .dsme_rgb_data_parser import DSMERGBDataParser
    from .focal_loss import FocalLoss
    from .early_stopping import EarlyStopping
    from .metrics import calculate_metrics, plot_confusion_matrix

# --- Configuration ---
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, 'data', 'DSME_pic'))
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'dsme_rgb_result')
RESULT_NAME = 'result_dsme_rgb_resnet18'

NUM_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

# Map DSME emotions to 3 classes (Positive, Negative, Surprise)
EMOTION_MAP = {
    'happiness': 0, # positive
    'disgust': 1,   # negative
    'fear': 1,      # negative
    'sadness': 1,   # negative
    'repression': 1,# negative
    'anger': 1,     # negative
    'surprise': 2,  # surprise
    'other': -1
}
CLASS_NAMES = ['positive', 'negative', 'surprise']

# --- Utils ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_metrics_with_std(metrics_list, class_names, filepath):
    """Saves final metrics including mean and std dev over multiple runs."""
    mean_metrics = {}
    std_metrics = {}
    
    overall_metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'uar', 'uf1']
    per_class_metric_keys = ['precision', 'recall', 'f1-score']

    # --- Overall Metrics ---
    for key in overall_metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)

    # --- Per-Class Metrics ---
    mean_metrics['per_class_metrics'] = {c: {} for c in class_names}
    std_metrics['per_class_metrics'] = {c: {} for c in class_names}
    for c_name in class_names:
        for pc_key in per_class_metric_keys:
            values = [m['per_class_metrics'][c_name][pc_key] for m in metrics_list if 'per_class_metrics' in m and c_name in m['per_class_metrics'] and pc_key in m['per_class_metrics'][c_name]]
            if values:
                mean_metrics['per_class_metrics'][c_name][pc_key] = np.mean(values)
                std_metrics['per_class_metrics'][c_name][pc_key] = np.std(values)

    # --- Confusion Matrix ---
    cm_list = [m['confusion_matrix'] for m in metrics_list if 'confusion_matrix' in m]
    if cm_list:
        cm_stack = np.stack(cm_list, axis=0)
        mean_metrics['confusion_matrix'] = np.mean(cm_stack, axis=0)
        std_metrics['confusion_matrix'] = np.std(cm_stack, axis=0)

    with open(filepath, 'w') as f:
        f.write(f"--- Final Metrics (Mean ± Std over {len(metrics_list)} runs) ---\n\n")
        
        f.write("--- Overall Performance ---\n")
        for key in overall_metric_keys:
            if key in mean_metrics:
                f.write(f"{key.replace('_', ' ').capitalize():<18}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
        f.write("\n")

        f.write("--- Per-class Metrics ---\n")
        for class_name in class_names:
            f.write(f"  Class: {class_name}\n")
            for pc_key in per_class_metric_keys:
                if pc_key in mean_metrics['per_class_metrics'][class_name]:
                    mean_val = mean_metrics['per_class_metrics'][class_name][pc_key]
                    std_val = std_metrics['per_class_metrics'][class_name][pc_key]
                    f.write(f"    {pc_key.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")
        
        f.write("--- Confusion Matrix ---\n")
        f.write("Mean:\n")
        f.write(np.array2string(mean_metrics['confusion_matrix'], formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write("\n\nStd Dev:\n")
        f.write(np.array2string(std_metrics['confusion_matrix'], formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write("\n")

def plot_average_learning_curves(history, result_dir):
    if len(history) == 0:
        return
    avg_history = {}
    for key in history[0].keys():
        max_len = max(len(h[key]) for h in history)
        padded_histories = []
        for h in history:
            # Pad with last value
            padded = h[key] + [h[key][-1]] * (max_len - len(h[key]))
            padded_histories.append(padded)
        avg_history[key] = np.mean(padded_histories, axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_history['train_loss'], label='Train Loss')
    plt.plot(avg_history['val_loss'], label='Validation Loss')
    plt.title('Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_history['train_acc'], label='Train Accuracy')
    plt.plot(avg_history['val_acc'], label='Validation Accuracy')
    plt.title('Average Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_learning_curves.png'))
    plt.close()

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, RESULT_NAME), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse Data
    parser = DSMERGBDataParser(DATA_DIR, EMOTION_MAP)
    samples = parser.get_samples()
    subjects = parser.get_all_subjects()
    
    print(f"Total samples: {len(samples)}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Subjects: {subjects}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Config
    config = {
        'DATA_DIR': DATA_DIR,
        'RESULT_DIR': RESULT_NAME,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'ResNet-18 (Pure RGB)',
        'EMOTION_MAP': EMOTION_MAP,
        'CLASS_NAMES': CLASS_NAMES
    }
    with open(os.path.join(RESULT_DIR, RESULT_NAME, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    all_runs_metrics = []
    all_runs_histories = []

    # Multiple runs
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS, start=1):
        print(f"\n=== Run {run_idx}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)
        
        run_true_labels, run_pred_labels = [], []
        run_histories = []
        
        # LOSO
        for fold_idx, test_subject in enumerate(subjects, start=1):
            print(f"\n--- Fold {fold_idx}/{len(subjects)}: Test Subject {test_subject} ---")
            
            train_subjects = [s for s in subjects if s != test_subject]
            train_samples = [s for s in samples if s['subject'] in train_subjects]
            val_samples = [s for s in samples if s['subject'] == test_subject]
            
            if len(val_samples) == 0:
                print(f"No validation samples for subject {test_subject}, skipping...")
                continue

            # Datasets
            train_dataset = DSMERGBDataset(train_samples, transform=train_transform)
            val_dataset = DSMERGBDataset(val_samples, transform=val_transform)
            
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)

            # Model (Pure ResNet18)
            model = resnet18(num_classes=len(CLASS_NAMES), pretrained=True, in_channels=3).to(device)
            
            # Loss & Optimizer
            criterion = FocalLoss(gamma=2)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            
            early_stopping = EarlyStopping(patience=30, verbose=True, 
                                         path=os.path.join(RESULT_DIR, RESULT_NAME, f'run_{run_idx}_fold_{fold_idx}_best.pt'))

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            # Training Loop
            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                for inputs, labels in train_loader:
                    # Filter invalid
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    
                    inputs = inputs[valid_mask].to(device)
                    labels = labels[valid_mask].long().to(device)
                    
                    # Focal Loss Alpha
                    class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
                    class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
                    criterion.alpha = class_weights / class_weights.sum()

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        valid_mask = labels != -1
                        if not valid_mask.any(): continue
                        
                        inputs = inputs[valid_mask].to(device)
                        labels = labels[valid_mask].long().to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                train_loss /= len(train_dataset) if len(train_dataset) > 0 else 1
                val_loss /= len(val_dataset) if len(val_dataset) > 0 else 1
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
                
                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            run_histories.append(fold_history)
            
            # Load best
            if os.path.exists(early_stopping.path):
                model.load_state_dict(torch.load(early_stopping.path))
            
            model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    inputs = inputs[valid_mask].to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    run_true_labels.extend(labels[valid_mask].cpu().numpy())
                    run_pred_labels.extend(predicted.cpu().numpy())

        # Calculate metrics for this run
        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))))
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        
        # Save confusion matrix
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, 
                            os.path.join(RESULT_DIR, RESULT_NAME, f'run_{run_idx}_confusion_matrix.png'))

    # Final Save
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, RESULT_NAME, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, os.path.join(RESULT_DIR, RESULT_NAME))
    print("All runs complete.")

if __name__ == '__main__':
    main()
