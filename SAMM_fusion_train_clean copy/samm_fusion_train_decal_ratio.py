import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import hashlib
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from samm_data_parser import SAMMDataParser
from samm_fusion_dataset import SAMMFusionDataset
from fusion_model import create_fusion_model
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping
from samm_static_augmentation_ratio import append_augmented_static_samples

# --- Configuration ---
# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_FILE = os.path.join(SCRIPT_DIR, 'samm_clean_labels.csv')
ALIGNED_RGB_DIR = os.path.join(SCRIPT_DIR, 'samm_aligned_rgb')
DYNAMIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, 'samm_dynamic_data_clean')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'fusion_result')

USE_STATIC_AUG = False
STATIC_AUG_DIR = os.path.join(SCRIPT_DIR, 'samm_static_decalcomanie')
STATIC_AUG_VARIANTS = ('L', 'R')
STATIC_AUG_EMOTIONS = ('positive', 'surprise')
STATIC_AUG_EMOTION_RATIOS = {
    'positive': 1.0,
    'surprise': 1.0,
}
STATIC_AUG_DEFAULT_RATIO = 0.0

NUM_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}
CLASS_NAMES = ['positive', 'negative', 'surprise']

def set_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model_weights_hash(model):
    model_state_dict = model.state_dict()
    hasher = hashlib.sha256()
    for key in sorted(model_state_dict.keys()):
        hasher.update(str(key).encode('utf-8'))
        hasher.update(model_state_dict[key].cpu().numpy().tobytes())
    return hasher.hexdigest()

def get_data_hash(data):
    hasher = hashlib.sha256()
    for sample in data:
        hasher.update(str(sample).encode('utf-8'))
    return hasher.hexdigest()

def save_metrics_with_std(metrics_list, class_names, filepath):
    """Saves final metrics including mean and std dev over multiple runs."""
    mean_metrics = {}
    std_metrics = {}
    
    overall_metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'uar', 'uf1']
    overall_metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision (weighted)',
        'recall': 'Recall (weighted)',
        'f1_score': 'F1 score (weighted)',
        'uar': 'UAR',
        'uf1': 'UF1',
    }
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
                label = overall_metric_labels.get(key, key.replace('_', ' ').capitalize())
                f.write(f"{label:<22}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
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

def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # --- Transformations ---
    rgb_train_transform_dict = {
        "Resize": {"size": (224, 224)},
        "RandomRotation": {"degrees": 3},
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    rgb_train_transform = transforms.Compose([
        transforms.Resize(rgb_train_transform_dict["Resize"]["size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_train_transform_dict["Normalize"]["mean"], 
                           std=rgb_train_transform_dict["Normalize"]["std"])
    ])

    rgb_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dynamic_train_transform_dict = {
        "Resize": {"size": (224, 224)},
        "RandomRotation": {"degrees": 3},
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    dynamic_train_transform = transforms.Compose([
        transforms.Resize(dynamic_train_transform_dict["Resize"]["size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=dynamic_train_transform_dict["Normalize"]["mean"], 
                           std=dynamic_train_transform_dict["Normalize"]["std"])
    ])

    dynamic_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Data Loading ---
    print("Initializing SAMM Data Parser...")
    data_parser = SAMMDataParser(MAPPING_FILE, ALIGNED_RGB_DIR, EMOTION_MAP)
    all_samples = data_parser.get_samples()
    all_subjects = data_parser.get_all_subjects()
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Total subjects: {len(all_subjects)}")
    
    # --- Cross-validation setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_runs_metrics = []
    all_runs_histories = []
    
    # --- Multiple runs ---
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n=== Run {run_idx+1}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)
        
        run_true_labels, run_pred_labels = [], []
        run_histories = []
        
        # LOSO
        for fold_idx, test_subject in enumerate(all_subjects):
            print(f"\n--- Fold {fold_idx+1}/{len(all_subjects)}: Test Subject {test_subject} ---")
            
            train_subjects = [s for s in all_subjects if s != test_subject]
            train_samples = [s for s in all_samples if s['subject'] in train_subjects]
            val_samples = [s for s in all_samples if s['subject'] == test_subject]

            if USE_STATIC_AUG:
                train_samples = append_augmented_static_samples(
                    train_samples,
                    STATIC_AUG_DIR,
                    variants=STATIC_AUG_VARIANTS,
                    allowed_emotions=STATIC_AUG_EMOTIONS,
                    emotion_variant_ratios=STATIC_AUG_EMOTION_RATIOS,
                    default_ratio=STATIC_AUG_DEFAULT_RATIO,
                )
            
            if len(val_samples) == 0:
                print(f"No validation samples for subject {test_subject}, skipping...")
                continue
            
            train_dataset = SAMMFusionDataset(train_samples, 
                                              rgb_transform=rgb_train_transform, 
                                              dynamic_transform=dynamic_train_transform,
                                              dynamic_image_dir=DYNAMIC_IMAGE_DIR,
                                              frame_selection='random')
            val_dataset = SAMMFusionDataset(val_samples, 
                                            rgb_transform=rgb_val_transform, 
                                            dynamic_transform=dynamic_val_transform,
                                            dynamic_image_dir=DYNAMIC_IMAGE_DIR,
                                            frame_selection='middle')
            
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)
            
            # Create model (using the patched factory if injected)
            model = create_fusion_model(num_classes=len(CLASS_NAMES), dropout_p=0.5, pretrained=True).to(device)
            
            criterion = FocalLoss(gamma=2)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            
            early_stopping = EarlyStopping(patience=30, verbose=True, path=None)

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                for rgb_images, dynamic_images, labels in train_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    
                    rgb_images = rgb_images[valid_mask].to(device)
                    dynamic_images = dynamic_images[valid_mask].to(device)
                    labels = labels[valid_mask].long().to(device)
                    
                    class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
                    class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
                    criterion.alpha = class_weights / class_weights.sum()

                    optimizer.zero_grad()
                    outputs = model(rgb_images, dynamic_images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                    optimizer.step()
                    
                    train_loss += loss.item() * rgb_images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for rgb_images, dynamic_images, labels in val_loader:
                        valid_mask = labels != -1
                        if not valid_mask.any(): continue
                        
                        rgb_images = rgb_images[valid_mask].to(device)
                        dynamic_images = dynamic_images[valid_mask].to(device)
                        labels = labels[valid_mask].long().to(device)
                        
                        outputs = model(rgb_images, dynamic_images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * rgb_images.size(0)
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
                
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            run_histories.append(fold_history)
            
            # Load best model from memory
            if early_stopping.best_model_state is not None:
                model.load_state_dict(early_stopping.best_model_state)
                print("Loaded best model from memory for evaluation.")
            else:
                print("Warning: No best model found, using last epoch model.")
            
            model.eval()
            with torch.no_grad():
                for rgb_images, dynamic_images, labels in val_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    rgb_images, dynamic_images, labels = rgb_images[valid_mask].to(device), dynamic_images[valid_mask].to(device), labels[valid_mask].to(device)
                    outputs = model(rgb_images, dynamic_images)
                    _, predicted = torch.max(outputs.data, 1)
                    run_true_labels.extend(labels.cpu().numpy())
                    run_pred_labels.extend(predicted.cpu().numpy())

        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))))
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        print(
            f"Run {run_idx+1} metrics | "
            f"Accuracy: {run_metrics['accuracy']:.4f} | "
            f"UAR: {run_metrics['uar']:.4f} | "
            f"UF1: {run_metrics['uf1']:.4f}"
        )
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, 
                            os.path.join(RESULT_DIR, f'run_{run_idx+1}_confusion_matrix.png'))

    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)

    print("\n--- All runs complete! ---")

if __name__ == '__main__':
    main()
