import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import hashlib
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Import project modules
from combined_data_parser import CombinedDataParser
from fusion_dataset import CrossDBFusionDataset
from fusion_model_se import create_fusion_model_se
from focal_loss import FocalLoss
from early_stopping import EarlyStopping
from metrics import calculate_metrics, plot_confusion_matrix

# --- Configuration ---
RESULT_DIR = 'fusion_result_cross_db'
NUM_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0001 # 1e-4 from CASME2 config
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

# Emotion Map (Standard 3 class)
EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}
CLASS_NAMES = ['positive', 'negative', 'surprise']

# Dataset Config
DYNAMIC_IMAGE_DIR = './merged_dynamic_data'

def set_seed(seed):
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

def save_metrics_with_std(metrics_list, class_names, filepath):
    """Saves final metrics including mean and std dev over multiple runs."""
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

def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # --- Transforms (Aligned with CASME2 Config) ---
    rgb_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5), # 5 degrees for standard config
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Optional, kept from some configs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rgb_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Reuse for Dynamic
    dynamic_train_transform = rgb_train_transform
    dynamic_val_transform = rgb_val_transform

    # --- Save Configuration ---
    config = {
        'RESULT_DIR': RESULT_DIR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'Late Fusion ResNet-18 SE-Block (Cross-DB)',
        'fusion_architecture': 'RGB(Grid+CBAM) + Dyn(SPP) -> Concat -> SE -> FC',
        'patience': 30
    }
    
    with open(os.path.join(RESULT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- Data Loading ---
    print("Loading Combined Dataset...")
    data_parser = CombinedDataParser()
    all_samples = data_parser.get_samples()
    all_subjects = data_parser.get_all_subjects()
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Total subjects: {len(all_subjects)}")
    
    # Save data hash
    def get_data_hash(data):
        hasher = hashlib.sha256()
        for sample in data:
            hasher.update(str(sample).encode('utf-8'))
        return hasher.hexdigest()

    data_hash = get_data_hash(all_samples)
    with open(os.path.join(RESULT_DIR, 'data_hash.txt'), 'w') as f:
        f.write(f"Data hash: {data_hash}\n")
        f.write(f"Total samples: {len(all_samples)}\n")
        f.write(f"Subjects: {len(all_subjects)}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_runs_metrics = []
    all_runs_histories = []

    # --- Multiple Runs ---
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n=== Run {run_idx+1}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)
        
        run_true_labels, run_pred_labels = [], []
        run_histories = []
        
        # LOSO Cross-Validation
        for fold_idx, test_subject in enumerate(all_subjects):
            print(f"\n--- Fold {fold_idx+1}/{len(all_subjects)}: Test Subject {test_subject} ---")
            
            # Split
            train_subjects = [s for s in all_subjects if s != test_subject]
            train_samples = [s for s in all_samples if s['subject'] in train_subjects]
            val_samples = [s for s in all_samples if s['subject'] == test_subject]
            
            # Print like CASME2
            # print(f"Train subjects: {train_subjects}") # Too many to print
            print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
            
            # Skip if no validation samples (e.g. subject has no valid samples)
            if len(val_samples) == 0:
                print(f"No validation samples for subject {test_subject}, skipping...")
                continue

            # Create Datasets
            train_dataset = CrossDBFusionDataset(train_samples, 
                                               rgb_transform=rgb_train_transform, 
                                               dynamic_transform=dynamic_train_transform,
                                               dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            val_dataset = CrossDBFusionDataset(val_samples, 
                                             rgb_transform=rgb_val_transform, 
                                             dynamic_transform=dynamic_val_transform,
                                             dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)
            
            # Model
            model = create_fusion_model_se(num_classes=3, dropout_p=0.5, pretrained=True)
            model = model.to(device)
            
            # Loss & Optimizer
            criterion = FocalLoss(gamma=2.0)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            
            # Early Stopping - NO SAVE to save disk space
            # We will use in-memory state saving manually if needed, or just let EarlyStopping handle it but with a dummy path that we delete later?
            # Better: EarlyStopping keeps best_model_state in memory? 
            # The current EarlyStopping implementation saves to disk.
            # Let's use a temporary path for this fold and delete it after loading.
            temp_checkpoint_path = os.path.join(RESULT_DIR, f'temp_run{run_idx}_fold{fold_idx}.pt')
            early_stopping = EarlyStopping(patience=30, verbose=False, path=temp_checkpoint_path) # Patience 30 for LOSO stability

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                for rgb, dyn, labels in train_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    
                    rgb, dyn, labels = rgb[valid_mask].to(device), dyn[valid_mask].to(device), labels[valid_mask].to(device)
                    
                    # Dynamic Alpha
                    class_counts = torch.bincount(labels, minlength=3).float()
                    weights = 1.0 / (class_counts + 1e-6)
                    criterion.alpha = (weights / weights.sum()).to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(rgb, dyn)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                    optimizer.step()
                    
                    train_loss += loss.item() * rgb.size(0)
                    _, preds = torch.max(outputs, 1)
                    train_correct += (preds == labels).sum().item()
                    train_total += rgb.size(0)
                
                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()
                    
                # Validation
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                with torch.no_grad():
                    for rgb, dyn, labels in val_loader:
                        valid_mask = labels != -1
                        if not valid_mask.any(): continue
                        rgb, dyn, labels = rgb[valid_mask].to(device), dyn[valid_mask].to(device), labels[valid_mask].to(device)
                        
                        outputs = model(rgb, dyn)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * rgb.size(0)
                        _, preds = torch.max(outputs, 1)
                        val_correct += (preds == labels).sum().item()
                        val_total += rgb.size(0)
                
                train_loss = train_loss / len(train_dataset)
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_loss = val_loss / len(val_dataset)
                val_acc = val_correct / val_total if val_total > 0 else 0
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            run_histories.append(fold_history)
            
            # Load best model
            if os.path.exists(temp_checkpoint_path):
                model.load_state_dict(torch.load(temp_checkpoint_path))
                os.remove(temp_checkpoint_path) # Delete immediately to save space
            
            # Final Inference
            model.eval()
            with torch.no_grad():
                for rgb, dyn, labels in val_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any(): continue
                    rgb, dyn, labels = rgb[valid_mask].to(device), dyn[valid_mask].to(device), labels[valid_mask].to(device)
                    outputs = model(rgb, dyn)
                    _, preds = torch.max(outputs, 1)
                    run_true_labels.extend(labels.cpu().numpy())
                    run_pred_labels.extend(preds.cpu().numpy())
            
            print(f"Fold {fold_idx+1} Done. Val Acc: {fold_history['val_acc'][-1]:.4f} (Best: {early_stopping.best_score:.4f})")

        # Run Metrics
        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=[0, 1, 2])
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        
        print(f"Run {run_idx+1} Accuracy: {run_metrics['accuracy']:.4f}")
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, os.path.join(RESULT_DIR, f'run_{run_idx+1}_confusion_matrix.png'))

    # Final Save
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)
    print(f"\nFinal averaged metrics saved to {os.path.join(RESULT_DIR, 'final_metrics.txt')}")

    print("\n--- All runs complete! Final averaged metrics saved. ---")

    # --- Final Model Training on All Data ---
    print("\n--- Starting final model training on all data... ---")
    set_seed(42) # Use a fixed seed for the final training
    
    # Create final training dataset
    final_train_dataset = CrossDBFusionDataset(all_samples, 
                                            rgb_transform=rgb_train_transform, 
                                            dynamic_transform=dynamic_train_transform,
                                            dynamic_image_dir=DYNAMIC_IMAGE_DIR)
    g = torch.Generator()
    g.manual_seed(42)
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    final_model = create_fusion_model_se(num_classes=3, dropout_p=0.5, pretrained=True)
    final_model = final_model.to(device)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_criterion = FocalLoss(gamma=2.0)

    # No validation, just train for a fixed number of epochs
    for epoch in range(NUM_EPOCHS):
        final_model.train()
        for rgb, dyn, labels in final_train_loader:
            valid_mask = labels != -1
            if not valid_mask.any(): continue
            
            rgb, dyn, labels = rgb[valid_mask].to(device), dyn[valid_mask].to(device), labels[valid_mask].to(device)
            
            class_counts = torch.bincount(labels, minlength=3).float()
            weights = 1.0 / (class_counts + 1e-6)
            final_criterion.alpha = (weights / weights.sum()).to(device)
            
            final_optimizer.zero_grad()
            outputs = final_model(rgb, dyn)
            loss = final_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=CLIP_GRAD_NORM)
            final_optimizer.step()
        print(f"Final training epoch {epoch+1}/{NUM_EPOCHS} complete.")

    final_model_path = os.path.join(RESULT_DIR, 'fusion_resnet18_cross_db_final_model_for_gui.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"--- Final fusion model saved to {final_model_path} ---")

if __name__ == '__main__':
    main()
