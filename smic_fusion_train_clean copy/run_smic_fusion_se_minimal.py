import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Import project modules
from fusion_model_se import create_fusion_model_se
from focal_loss_ls import FocalLossLabelSmoothing # Changed to Label Smoothing
from smic_data_parser import SMICDataParser
from fusion_dataset import SMICFusionDataset
from early_stopping import EarlyStopping
from metrics import calculate_metrics

# --- Configuration ---
RESULT_DIR = 'fusion_result_smic_se_minimal'
TRANSFER_WEIGHTS_PATH = None

# SMIC Dataset Configuration
SMIC_RGB_DIR = '../data/SMIC2'
DYNAMIC_IMAGE_DIR = 'smic_dynamic_data2'

# SMIC Emotion Map
EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}
CLASS_NAMES = ['positive', 'negative', 'surprise']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            # Low resolution to match SMIC quality
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            
            # --- Moderate Augmentation (Increased from Minimal) ---
            transforms.RandomRotation(degrees=10), # Increased from 5 to 10
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Increased from 0.1 to 0.2
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Added translation
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# --- Mixup REMOVED ---
# We use standard training loop

def make_weighted_sampler(dataset):
    targets = [s['label'] for s in dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for rgb, dyn, labels in dataloader:
        valid_mask = labels != -1
        if not valid_mask.any():
            continue
            
        rgb = rgb[valid_mask].to(device)
        dyn = dyn[valid_mask].to(device)
        labels = labels[valid_mask].to(device).long()
        
        optimizer.zero_grad()
        outputs = model(rgb, dyn)
        loss = criterion(outputs, labels)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * rgb.size(0)
        
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
            
        total_samples += rgb.size(0)
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = running_corrects / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for rgb, dyn, labels in dataloader:
            valid_mask = labels != -1
            if not valid_mask.any():
                continue
                
            rgb = rgb[valid_mask].to(device)
            dyn = dyn[valid_mask].to(device)
            labels = labels[valid_mask].to(device).long()
            
            outputs = model(rgb, dyn)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * rgb.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += rgb.size(0)
            
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = running_corrects / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc

def save_metrics_with_std(metrics_list, class_names, filepath):
    mean_metrics = {}
    std_metrics = {}
    
    overall_metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'uar', 'uf1']
    per_class_metric_keys = ['precision', 'recall', 'f1-score']

    for key in overall_metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)

    mean_metrics['per_class_metrics'] = {c: {} for c in class_names}
    std_metrics['per_class_metrics'] = {c: {} for c in class_names}
    for c_name in class_names:
        for pc_key in per_class_metric_keys:
            values = [m['per_class_metrics'][c_name][pc_key] for m in metrics_list if 'per_class_metrics' in m and c_name in m['per_class_metrics'] and pc_key in m['per_class_metrics'][c_name]]
            if values:
                mean_metrics['per_class_metrics'][c_name][pc_key] = np.mean(values)
                std_metrics['per_class_metrics'][c_name][pc_key] = np.std(values)

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
    avg_history = {}
    if not history: return
    for key in history[0].keys():
        max_len = max(len(h[key]) for h in history)
        padded_histories = []
        for h in history:
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

def plot_confusion_matrix(cm, classes, filename, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    plt.close()

def main():
    print(f"Starting SMIC Fusion Training with Minimalist Strategy...")
    print(f"Architecture: RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE Block")
    print(f"Strategy: LowRes(112) + SGD + WeightedSampler + WeakAug (No Mixup/Erasing)")
    print(f"Results will be saved to: {RESULT_DIR}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading SMIC Data...")
    parser = SMICDataParser(SMIC_RGB_DIR, EMOTION_MAP)
    samples = parser.get_samples()
    print(f"Total samples: {len(samples)}")
    subjects = sorted(list(set(s['subject'] for s in samples)))
    print(f"Total subjects: {len(subjects)}")
    
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    NUM_RUNS = 3
    all_runs_metrics = []
    all_runs_histories = []
    
    for run in range(1, NUM_RUNS + 1):
        print(f"\n=== Run {run}/{NUM_RUNS} (seed={41+run}) ===")
        set_seed(41 + run)
        
        run_preds = []
        run_labels = []
        run_histories = []
        
        for i, test_sub in enumerate(subjects):
            print(f"\n--- Fold {i+1}/{len(subjects)}: Test Subject {test_sub} ---")
            
            train_samples = [s for s in samples if s['subject'] != test_sub]
            test_samples = [s for s in samples if s['subject'] == test_sub]
            
            train_dataset = SMICFusionDataset(
                train_samples, 
                rgb_transform=get_transforms(is_train=True),
                dynamic_transform=get_transforms(is_train=True),
                dynamic_image_dir=DYNAMIC_IMAGE_DIR
            )
            val_dataset = SMICFusionDataset(
                test_samples, 
                rgb_transform=get_transforms(is_train=False),
                dynamic_transform=get_transforms(is_train=False),
                dynamic_image_dir=DYNAMIC_IMAGE_DIR
            )
            
            sampler = make_weighted_sampler(train_dataset)
            
            train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
            
            model = create_fusion_model_se(
                num_classes=3, 
                dropout_p=0.6, 
                pretrained=True, 
                transfer_weights_path=TRANSFER_WEIGHTS_PATH
            )
            model = model.to(device)
            
            # Use FocalLoss with Label Smoothing (0.1) for regularization
            # Note: WeightedRandomSampler is still active, so we rely on smoothing + sampler
            criterion = FocalLossLabelSmoothing(gamma=2.0, smoothing=0.1)
            
            # Reduced LR from 0.01 to 0.005 to stabilize training
            optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
            
            early_stopping = EarlyStopping(patience=30, verbose=True, path=None)
            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(60):
                # Standard training (No Mixup)
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = validate(model, val_loader, criterion, device)
                
                scheduler.step()
                
                print(f"Epoch {epoch+1}/60: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_acc'].append(val_acc)

                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print("EarlyStopping counter: early stop")
                    break
            
            run_histories.append(fold_history)
            
            if early_stopping.best_model_state is not None:
                model.load_state_dict(early_stopping.best_model_state)
            
            model.eval()
            fold_preds = []
            fold_labels = []
            with torch.no_grad():
                for rgb, dyn, target in val_loader:
                    rgb, dyn, target = rgb.to(device), dyn.to(device), target.to(device)
                    outputs = model(rgb, dyn)
                    _, predicted = torch.max(outputs.data, 1)
                    fold_preds.extend(predicted.cpu().numpy())
                    fold_labels.extend(target.cpu().numpy())
            
            run_preds.extend(fold_preds)
            run_labels.extend(fold_labels)
            
        run_metrics = calculate_metrics(run_labels, run_preds, labels=[0, 1, 2])
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        
        print(f"Run {run} Accuracy: {run_metrics['accuracy']:.4f}")
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, os.path.join(RESULT_DIR, f'run_{run}_confusion_matrix.png'))

    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)
    print(f"\nFinal averaged metrics saved to {os.path.join(RESULT_DIR, 'final_metrics.txt')}")

if __name__ == '__main__':
    main()
