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

from casme2_data_parser import CASME2DataParser
from fusion_dataset import CASME2FusionDataset
from fusion_model import create_fusion_model
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping

# --- Configuration ---
RAW_VIDEO_DIR = '../data/CASME2_RAW_selected/CASME2_RAW_selected'
ANNOTATION_FILE = '../data/CASME2_RAW_selected/CASME2-coding-20140508.xlsx'
DYNAMIC_IMAGE_DIR = './dynamic_data'
RESULT_DIR = 'fusion_result'
NUM_EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 0.0005  # Reduced learning rate
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

EMOTION_MAP = {
    'disgust': 0,
    'happiness': 1,
    'repression': 2,
    'surprise': 3,
    'sadness': 4,
    'fear': 4,
    'others': 4
}
CLASS_NAMES = ['disgust', 'happiness', 'repression', 'surprise', 'sadness_fear_others']

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
    """Computes a hash of the data for reproducibility tracking."""
    hasher = hashlib.sha256()
    for sample in data:
        hasher.update(str(sample).encode('utf-8'))
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
        padded_histories = []
        for h in history:
            padded = h[key] + [h[key][-1]] * (max_len - len(h[key]))
            padded_histories.append(padded)
        avg_history[key] = np.mean(padded_histories, axis=0)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_history['train_loss'], label='Train Loss')
    plt.plot(avg_history['val_loss'], label='Validation Loss')
    plt.title('Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
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

    # --- Transformations for RGB (3 channels) ---
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

    # --- Transformations for Dynamic Images (3 channels) ---
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

    # --- Save Configuration ---
    config = {
        'RESULT_DIR': RESULT_DIR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'Late Fusion ResNet-18 with AdamW, Grad Clip, CosineAnnealingLR',
        'rgb_train_transform': rgb_train_transform_dict,
        'dynamic_train_transform': dynamic_train_transform_dict,
        'fusion_architecture': 'Late fusion at feature level (RGB: 512 + Dynamic: 512 -> 1024 -> 5 classes)'
    }
    
    with open(os.path.join(RESULT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # --- Data Loading ---
    data_parser = CASME2DataParser(ANNOTATION_FILE, RAW_VIDEO_DIR, EMOTION_MAP)
    all_samples = data_parser.get_samples()
    all_subjects = data_parser.get_all_subjects()
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Total subjects: {len(all_subjects)}")
    print(f"Subjects: {all_subjects}")
    
    # Save data hash for reproducibility
    data_hash = get_data_hash(all_samples)
    with open(os.path.join(RESULT_DIR, 'data_hash.txt'), 'w') as f:
        f.write(f"Data hash: {data_hash}\n")
        f.write(f"Total samples: {len(all_samples)}\n")
        f.write(f"Subjects: {all_subjects}\n")

    # --- Cross-validation setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    
    all_runs_metrics = []
    all_runs_histories = []
    
    # --- Multiple runs for robustness ---
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n=== Run {run_idx+1}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)
        
        # Leave-One-Subject-Out Cross-Validation
        run_true_labels, run_pred_labels = [], []
        run_histories = []
        
        for fold_idx, test_subject in enumerate(all_subjects):
            print(f"\n--- Fold {fold_idx+1}/{len(all_subjects)}: Test Subject {test_subject} ---")
            
            # Split data
            train_subjects = [s for s in all_subjects if s != test_subject]
            train_samples = [s for s in all_samples if s['subject'] in train_subjects]
            val_samples = [s for s in all_samples if s['subject'] == test_subject]
            
            print(f"Train subjects: {train_subjects}")
            print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
            
            if len(val_samples) == 0:
                print(f"No validation samples for subject {test_subject}, skipping...")
                continue
            
            # Create datasets
            train_dataset = CASME2FusionDataset(train_samples, 
                                              rgb_transform=rgb_train_transform, 
                                              dynamic_transform=dynamic_train_transform,
                                              dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            val_dataset = CASME2FusionDataset(val_samples, 
                                            rgb_transform=rgb_val_transform, 
                                            dynamic_transform=dynamic_val_transform,
                                            dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            
            # Create data loaders
            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                    num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                  num_workers=0, pin_memory=True)
            
            # Create model
            model = create_fusion_model(num_classes=len(CLASS_NAMES), dropout_p=0.5, pretrained=True).to(device)
            
            # Save model hash for reproducibility
            model_hash = get_model_weights_hash(model)
            print(f"Model weights hash: {model_hash[:16]}...")
            
            # Loss and optimizer
            criterion = FocalLoss(gamma=2)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            
            early_stopping = EarlyStopping(patience=40, verbose=True, 
                                         path=os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt'))

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                # Dynamic alpha for Focal Loss based on current batch
                for rgb_images, dynamic_images, labels in train_loader:
                    # Filter out invalid samples (label = -1)
                    valid_mask = labels != -1
                    if not valid_mask.any():
                        continue
                    
                    rgb_images = rgb_images[valid_mask]
                    dynamic_images = dynamic_images[valid_mask]
                    labels = labels[valid_mask]
                    
                    rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.long().to(device)
                    
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

                # Adjust LR
                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                epoch_true_labels, epoch_pred_labels = [], []
                with torch.no_grad():
                    for rgb_images, dynamic_images, labels in val_loader:
                        # Filter out invalid samples (label = -1)
                        valid_mask = labels != -1
                        if not valid_mask.any():
                            continue
                        
                        rgb_images = rgb_images[valid_mask]
                        dynamic_images = dynamic_images[valid_mask]
                        labels = labels[valid_mask]
                        
                        rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.long().to(device)
                        outputs = model(rgb_images, dynamic_images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * rgb_images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        epoch_true_labels.extend(labels.cpu().numpy())
                        epoch_pred_labels.extend(predicted.cpu().numpy())

                train_loss /= len(train_dataset)
                val_loss /= len(val_dataset)
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Early stopping based on validation accuracy
                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            run_histories.append(fold_history)
            
            # Load best model for final evaluation
            model.load_state_dict(torch.load(os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt')))
            model.eval()
            with torch.no_grad():
                for rgb_images, dynamic_images, labels in val_loader:
                    rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.to(device)
                    outputs = model(rgb_images, dynamic_images)
                    _, predicted = torch.max(outputs.data, 1)
                    run_true_labels.extend(labels.cpu().numpy())
                    run_pred_labels.extend(predicted.cpu().numpy())

        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))), class_names=CLASS_NAMES)
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, 
                            os.path.join(RESULT_DIR, f'run_{run_idx+1}_confusion_matrix.png'))

    # --- Final Metrics & Plotting ---
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)

    print("\n--- All runs complete! Final averaged metrics saved. ---")

    # --- Final Model Training on All Data ---
    print("\n--- Starting final model training on all data... ---")
    set_seed(42) # Use a fixed seed for the final training
    
    # Create final training dataset
    final_train_dataset = CASME2FusionDataset(all_samples, 
                                            rgb_transform=rgb_train_transform, 
                                            dynamic_transform=dynamic_train_transform,
                                            dynamic_image_dir=DYNAMIC_IMAGE_DIR)
    g = torch.Generator()
    g.manual_seed(42)
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    final_model = create_fusion_model(num_classes=len(CLASS_NAMES), dropout_p=0.5, pretrained=True).to(device)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_criterion = FocalLoss(gamma=2)

    # No validation, just train for a fixed number of epochs
    for epoch in range(NUM_EPOCHS):
        final_model.train()
        for rgb_images, mhi_images, labels in final_train_loader:
            rgb_images, mhi_images, labels = rgb_images.to(device), mhi_images.to(device), labels.to(device)
            
            class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
            class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
            final_criterion.alpha = class_weights / class_weights.sum()

            final_optimizer.zero_grad()
            outputs = final_model(rgb_images, mhi_images)
            loss = final_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=CLIP_GRAD_NORM)
            final_optimizer.step()
        print(f"Final training epoch {epoch+1}/{NUM_EPOCHS} complete.")

    final_model_path = os.path.join(RESULT_DIR, 'fusion_resnet18_casme2_final_model_for_gui.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"--- Final fusion model saved to {final_model_path} ---")


if __name__ == '__main__':
    main()
