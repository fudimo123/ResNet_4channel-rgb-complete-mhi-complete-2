import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import hashlib
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from casme2_data_parser import CASME2DataParser
from dataset import CASME2Dataset
from model import resnet18
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping

# --- Configuration ---
RAW_VIDEO_DIR = '../data/CASME2_RAW_selected/CASME2_RAW_selected'
ANNOTATION_FILE = '../data/CASME2_RAW_selected/CASME2-coding-20140508.xlsx'
RESULT_DIR = 'result9'
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

EMOTION_MAP = {
    'happiness': 0, # positive
    'disgust': 1,   # negative
    'repression': 1,# negative
    'sadness': 1,   # negative
    'fear': 1,      # negative
    'surprise': 2   # surprise
}
CLASS_NAMES = ['positive', 'negative', 'surprise']

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """Ensures that dataloader workers have deterministic behavior."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model_weights_hash(model):
    """Computes a hash of the model's initial weights."""
    model_state_dict = model.state_dict()
    hasher = hashlib.sha256()
    for key in sorted(model_state_dict.keys()):
        hasher.update(str(key).encode('utf-8'))
        hasher.update(model_state_dict[key].cpu().numpy().tobytes())
    return hasher.hexdigest()

def get_data_hash(data):
    """Computes a hash of a batch of data."""
    hasher = hashlib.sha256()
    hasher.update(data.cpu().numpy().tobytes())
    return hasher.hexdigest()

def save_metrics_with_std(metrics_list, class_names, filepath):
    """Saves final metrics including mean and std dev over multiple runs."""
    mean_metrics = {}
    std_metrics = {}
    
    # Define all expected keys
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
    """Plots and saves the average learning curves for loss and accuracy."""
    avg_history = {}
    for key in history[0].keys():
        # Pad histories to the same length for averaging
        max_len = max(len(h[key]) for h in history)
        padded_histories = [np.pad(h[key], (0, max_len - len(h[key])), 'edge') for h in history]
        avg_history[key] = np.mean(padded_histories, axis=0)

    epochs = range(1, len(avg_history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Average Learning Curves', fontsize=16)

    # Plotting average training and validation loss
    ax1.plot(epochs, avg_history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, avg_history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.minorticks_on()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plotting average training and validation accuracy
    ax2.plot(epochs, avg_history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    ax2.plot(epochs, avg_history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.minorticks_on()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(result_dir, 'average_learning_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Average learning curves saved to {save_path}")


def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # --- Transformations ---
    train_transform_dict = {
        "RandomHorizontalFlip": {"p": 0.5},
        "RandomRotation": {"degrees": 3},
        "RandomResizedCrop": {"size": (224, 224), "scale": (0.8, 1.0)},
        "GaussianBlur": {"kernel_size": 3, "p": 0.3},
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=train_transform_dict["RandomHorizontalFlip"]["p"]),
        transforms.RandomRotation(train_transform_dict["RandomRotation"]["degrees"]),
        transforms.RandomResizedCrop(size=train_transform_dict["RandomResizedCrop"]["size"], scale=train_transform_dict["RandomResizedCrop"]["scale"]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=train_transform_dict["GaussianBlur"]["kernel_size"])], p=train_transform_dict["GaussianBlur"]["p"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_transform_dict["Normalize"]["mean"], std=train_transform_dict["Normalize"]["std"])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Save Configuration ---
    config = {
        'RESULT_DIR': RESULT_DIR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'ResNet-18 with AdamW, Grad Clip, CosineAnnealingLR',
        'train_transform': train_transform_dict,
        'reproducibility': {
            'seeds': REPRODUCIBILITY_SEEDS,
            'cudnn.benchmark': False,
            'cudnn.deterministic': True,
        }
    }
    with open(os.path.join(RESULT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- Data Loading ---
    data_parser = CASME2DataParser(ANNOTATION_FILE, RAW_VIDEO_DIR, EMOTION_MAP)
    all_subjects = data_parser.get_all_subjects()
    all_samples = data_parser.get_samples()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_runs_metrics = []
    all_runs_histories = []

    # --- LOSO Cross-Validation over Multiple Runs ---
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n--- Starting Run {run_idx + 1}/{len(REPRODUCIBILITY_SEEDS)} with Seed: {seed} ---")
        set_seed(seed)

        run_true_labels = []
        run_pred_labels = []
        
        run_log_file = os.path.join(RESULT_DIR, f'run_{run_idx+1}_seed_{seed}_log.txt')

        with open(run_log_file, 'w') as log_f:
            log_f.write(f"Run {run_idx + 1} with Seed {seed}\n\n")

            for fold_idx, subject_to_leave_out in enumerate(all_subjects):
                log_f.write(f"--- Fold {fold_idx+1}/{len(all_subjects)}: Leaving out subject {subject_to_leave_out} ---\n")
                print(f"--- Fold {fold_idx+1}/{len(all_subjects)}: Leaving out subject {subject_to_leave_out} ---")

                train_samples = [s for s in all_samples if s['subject'] != subject_to_leave_out]
                val_samples = [s for s in all_samples if s['subject'] == subject_to_leave_out]

                train_dataset = CASME2Dataset(train_samples, transform=train_transform)
                val_dataset = CASME2Dataset(val_samples, transform=val_transform)

                g = torch.Generator()
                g.manual_seed(seed)
                
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

                if len(val_loader) == 0:
                    warning_msg = f"WARNING: Skipping fold {fold_idx+1} for subject {subject_to_leave_out} due to empty validation set.\n"
                    print(warning_msg.strip())
                    log_f.write(warning_msg)
                    continue

                num_classes = len(CLASS_NAMES)
                model = resnet18(num_classes=num_classes, pretrained=True, in_channels=3).to(device)
                
                if run_idx == 0 and fold_idx == 0:
                     print(f"Initial model weights hash: {get_model_weights_hash(model)}")

                criterion = FocalLoss(gamma=2) # Alpha is dynamically calculated
                optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                
                warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
                main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
                
                early_stopping = EarlyStopping(patience=30, verbose=True, path=os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt'))

                fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    train_loss, train_correct, train_total = 0, 0, 0
                    
                    # Dynamic alpha for Focal Loss based on current batch
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        
                        class_counts = torch.bincount(labels, minlength=num_classes).float()
                        class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
                        criterion.alpha = class_weights / class_weights.sum()

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                        optimizer.step()
                        
                        train_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        train_correct += (predicted == labels).sum().item()

                    # Adjust LR
                    if epoch < WARMUP_EPOCHS:
                        warmup_scheduler.step()
                    else:
                        main_scheduler.step()

                    model.eval()
                    val_loss, val_correct, val_total = 0, 0, 0
                    epoch_true_labels, epoch_pred_labels = [], []
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item() * images.size(0)
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                            epoch_true_labels.extend(labels.cpu().numpy())
                            epoch_pred_labels.extend(predicted.cpu().numpy())
                    
                    # Store history
                    fold_history['train_loss'].append(train_loss / train_total)
                    fold_history['val_loss'].append(val_loss / val_total)
                    fold_history['train_acc'].append(train_correct / train_total)
                    fold_history['val_acc'].append(val_correct / val_total)

                    val_metrics = calculate_metrics(epoch_true_labels, epoch_pred_labels, labels=list(range(len(CLASS_NAMES))))
                    val_uar = val_metrics['uar']
                    
                    log_line = f"  Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {fold_history['train_loss'][-1]:.4f}, Val Loss: {fold_history['val_loss'][-1]:.4f} | Train Acc: {fold_history['train_acc'][-1]:.4f}, Val Acc: {fold_history['val_acc'][-1]:.4f}, Val UAR: {val_uar:.4f}\n"
                    print(log_line.strip())
                    log_f.write(log_line)

                    early_stopping(val_uar, model)
                    if early_stopping.early_stop:
                        log_f.write(f"Early stopping at epoch {epoch + 1}\n")
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                all_runs_histories.append(fold_history)
                model.load_state_dict(torch.load(early_stopping.path))
                
                model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        run_true_labels.extend(labels.cpu().numpy())
                        run_pred_labels.extend(predicted.cpu().numpy())

        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))))
        all_runs_metrics.append(run_metrics)
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, os.path.join(RESULT_DIR, f'run_{run_idx+1}_confusion_matrix.png'))

    # --- Final Metrics & Plotting ---
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)

    print("\n--- All runs complete! Final averaged metrics saved. ---")

    # --- Final Model Training on All Data ---
    print("\n--- Starting final model training on all data... ---")
    set_seed(42) # Use a fixed seed for the final training
    
    final_train_dataset = CASME2Dataset(all_samples, transform=train_transform)
    g = torch.Generator()
    g.manual_seed(42)
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    final_model = resnet18(num_classes=len(CLASS_NAMES), pretrained=True, in_channels=3).to(device)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_criterion = FocalLoss(gamma=2)

    # No validation, just train for a fixed number of epochs (e.g., average stop epoch from CV)
    # Or a fixed number like NUM_EPOCHS. Let's use NUM_EPOCHS for simplicity.
    for epoch in range(NUM_EPOCHS):
        final_model.train()
        for images, labels in final_train_loader:
            images, labels = images.to(device), labels.to(device)
            
            class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
            class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
            final_criterion.alpha = class_weights / class_weights.sum()

            final_optimizer.zero_grad()
            outputs = final_model(images)
            loss = final_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=CLIP_GRAD_NORM)
            final_optimizer.step()
        print(f"Final training epoch {epoch+1}/{NUM_EPOCHS} complete.")

    final_model_path = os.path.join(RESULT_DIR, 'resnet18_casme2_final_model_for_gui.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"--- Final model for GUI saved to {final_model_path} ---")


if __name__ == '__main__':
    main()