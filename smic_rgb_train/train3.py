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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import torch.nn.functional as F

from smic_data_parser import SMICDataParser
from dataset import CASME2Dataset
from model3 import resnet18_enhanced, resnet34_enhanced
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping


class AdaptiveLabelSmoothingCrossEntropy(nn.Module):
    """Adaptive label smoothing that adjusts based on training progress"""
    def __init__(self, initial_smoothing=0.15, min_smoothing=0.05):
        super(AdaptiveLabelSmoothingCrossEntropy, self).__init__()
        self.initial_smoothing = initial_smoothing
        self.min_smoothing = min_smoothing
        self.current_smoothing = initial_smoothing
        
    def update_smoothing(self, epoch, total_epochs):
        # Gradually reduce smoothing as training progresses
        progress = epoch / total_epochs
        self.current_smoothing = self.initial_smoothing * (1 - progress) + self.min_smoothing * progress
        
    def forward(self, pred, target):
        confidence = 1. - self.current_smoothing
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.current_smoothing * smooth_loss
        return loss.mean()


class MixUp:
    """MixUp data augmentation"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


# --- Enhanced Configuration ---
RAW_VIDEO_DIR = '../data/SMIC'
RESULT_DIR = 'result7'  # Changed to result7
NUM_EPOCHS = 80  # Increased epochs for better convergence
BATCH_SIZE = 10  # Slightly reduced for better gradient updates
LEARNING_RATE = 0.002  # Slightly increased initial learning rate
WEIGHT_DECAY = 2e-5  # Reduced weight decay
WARMUP_EPOCHS = 15  # Extended warmup
CLIP_GRAD_NORM = 1.0  # Increased gradient clipping
REPRODUCIBILITY_SEEDS = [42, 123, 456]  # Different seeds for better diversity

# Enhanced augmentation parameters
MIXUP_ALPHA = 0.3
CUTMIX_ALPHA = 1.0
AUG_PROB = 0.5  # Probability of applying MixUp/CutMix

EMOTION_MAP = {
    'positive': 0,  # positive
    'negative': 1,  # negative
    'surprise': 2   # surprise
}
CLASS_NAMES = ['positive', 'negative', 'surprise']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model_weights_hash(model):
    """Generate a hash of model weights for reproducibility verification"""
    weights_str = ''
    for param in model.parameters():
        weights_str += str(param.data.cpu().numpy().flatten()[:10])  # First 10 elements
    return hashlib.md5(weights_str.encode()).hexdigest()[:8]


def get_data_hash(data):
    """Generate a hash of data for reproducibility verification"""
    return hashlib.md5(str(data).encode()).hexdigest()[:8]


def save_metrics_with_std(metrics_list, class_names, filepath):
    """Save metrics with mean and standard deviation across multiple runs"""
    
    # Calculate mean and std for overall metrics
    overall_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'uar', 'uf1']
    overall_stats = {}
    
    for key in overall_keys:
        values = [run_metrics[key] for run_metrics in metrics_list]
        overall_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Calculate mean and std for per-class metrics
    per_class_stats = {}
    for i, class_name in enumerate(class_names):
        per_class_stats[class_name] = {}
        class_keys = ['precision', 'recall', 'f1']
        
        for key in class_keys:
            if key == 'precision':
                values = [run_metrics['per_class_precision'][i] for run_metrics in metrics_list]
            elif key == 'recall':
                values = [run_metrics['per_class_recall'][i] for run_metrics in metrics_list]
            elif key == 'f1':
                values = [run_metrics['per_class_f1'][i] for run_metrics in metrics_list]
            
            per_class_stats[class_name][key] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    # Calculate mean and std for confusion matrices
    confusion_matrices = [run_metrics['confusion_matrix'] for run_metrics in metrics_list]
    confusion_mean = np.mean(confusion_matrices, axis=0)
    confusion_std = np.std(confusion_matrices, axis=0)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write("--- Final Metrics (Mean ± Std over {} runs) ---\n\n".format(len(metrics_list)))
        
        f.write("--- Overall Performance ---\n")
        for key in overall_keys:
            mean_val = overall_stats[key]['mean']
            std_val = overall_stats[key]['std']
            f.write(f"{key.capitalize():<18}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write("\n--- Per-class Metrics ---\n")
        for class_name in class_names:
            f.write(f"  Class: {class_name}\n")
            for key in ['precision', 'recall', 'f1']:
                mean_val = per_class_stats[class_name][key]['mean']
                std_val = per_class_stats[class_name][key]['std']
                f.write(f"    {key.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write("\n--- Confusion Matrix ---\n")
        f.write("Mean:\n")
        f.write(str(confusion_mean) + "\n\n")
        f.write("Std Dev:\n")
        f.write(str(confusion_std) + "\n")
    
    # Also print to console
    print("\n--- Final Metrics (Mean ± Std over {} runs) ---\n".format(len(metrics_list)))
    
    print("--- Overall Performance ---")
    for key in overall_keys:
        mean_val = overall_stats[key]['mean']
        std_val = overall_stats[key]['std']
        print(f"{key.capitalize():<18}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n--- Per-class Metrics ---")
    for class_name in class_names:
        print(f"  Class: {class_name}")
    
    print("\n--- Confusion Matrix ---")
    print("Mean:")
    print(confusion_mean)
    print("\nStd Dev:")
    print(confusion_std)


def plot_average_learning_curves(history, result_dir):
    """Plot average learning curves across all runs"""
    
    # Calculate averages
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and Validation Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', alpha=0.8)
    ax1.set_title('Model Loss (Average across runs)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training and Validation Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', alpha=0.8)
    ax2.set_title('Model Accuracy (Average across runs)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning Rate Schedule
    if 'lr' in history:
        ax3.plot(epochs, history['lr'], 'g-', label='Learning Rate', alpha=0.8)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Loss Components (if available)
    if 'focal_loss' in history and 'smooth_loss' in history:
        ax4.plot(epochs, history['focal_loss'], 'orange', label='Focal Loss', alpha=0.8)
        ax4.plot(epochs, history['smooth_loss'], 'purple', label='Smooth Loss', alpha=0.8)
        ax4.set_title('Loss Components')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # --- Enhanced Transformations ---
    train_transform_dict = {
        "RandomHorizontalFlip": {"p": 0.7},  # Increased probability
        "RandomRotation": {"degrees": 12},  # Increased rotation range
        "RandomResizedCrop": {"size": (224, 224), "scale": (0.7, 1.0)},  # More aggressive cropping
        "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.6},  # Enhanced color variations
        "RandomAffine": {"degrees": 10, "translate": (0.1, 0.1), "scale": (0.85, 1.15), "p": 0.5},  # More geometric variations
        "GaussianBlur": {"kernel_size": 5, "p": 0.4},  # Increased blur probability
        "RandomErasing": {"p": 0.2, "scale": (0.02, 0.2), "ratio": (0.3, 3.3)},  # More aggressive erasing
        "RandomPerspective": {"distortion_scale": 0.3, "p": 0.4},  # Enhanced perspective transformation
        "ToTensor": {},
        "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    }
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=train_transform_dict["RandomHorizontalFlip"]["p"]),
        transforms.RandomRotation(train_transform_dict["RandomRotation"]["degrees"]),
        transforms.RandomResizedCrop(size=train_transform_dict["RandomResizedCrop"]["size"], 
                                   scale=train_transform_dict["RandomResizedCrop"]["scale"]),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=train_transform_dict["ColorJitter"]["brightness"],
            contrast=train_transform_dict["ColorJitter"]["contrast"],
            saturation=train_transform_dict["ColorJitter"]["saturation"],
            hue=train_transform_dict["ColorJitter"]["hue"]
        )], p=train_transform_dict["ColorJitter"]["p"]),
        transforms.RandomApply([transforms.RandomAffine(
            degrees=train_transform_dict["RandomAffine"]["degrees"],
            translate=train_transform_dict["RandomAffine"]["translate"],
            scale=train_transform_dict["RandomAffine"]["scale"]
        )], p=train_transform_dict["RandomAffine"]["p"]),
        transforms.RandomApply([transforms.GaussianBlur(
            kernel_size=train_transform_dict["GaussianBlur"]["kernel_size"]
        )], p=train_transform_dict["GaussianBlur"]["p"]),
        transforms.RandomPerspective(
            distortion_scale=train_transform_dict["RandomPerspective"]["distortion_scale"], 
            p=train_transform_dict["RandomPerspective"]["p"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_transform_dict["Normalize"]["mean"], 
                           std=train_transform_dict["Normalize"]["std"]),
        transforms.RandomErasing(
            p=train_transform_dict["RandomErasing"]["p"],
            scale=train_transform_dict["RandomErasing"]["scale"],
            ratio=train_transform_dict["RandomErasing"]["ratio"]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Save Enhanced Configuration ---
    config = {
        'RESULT_DIR': RESULT_DIR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MIXUP_ALPHA': MIXUP_ALPHA,
        'CUTMIX_ALPHA': CUTMIX_ALPHA,
        'AUG_PROB': AUG_PROB,
        'MODEL': 'Enhanced ResNet-34 with Multi-Scale Attention, MixUp/CutMix, Adaptive Label Smoothing',
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
    data_parser = SMICDataParser(RAW_VIDEO_DIR, EMOTION_MAP)
    all_subjects = data_parser.get_all_subjects()
    all_samples = data_parser.get_samples()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_runs_metrics = []
    all_runs_histories = []

    # --- LOSO Cross-Validation over Multiple Enhanced Runs ---
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n--- Starting Enhanced Run {run_idx + 1}/{len(REPRODUCIBILITY_SEEDS)} with Seed: {seed} ---")
        set_seed(seed)

        run_true_labels = []
        run_pred_labels = []
        
        run_log_file = os.path.join(RESULT_DIR, f'enhanced_run_{run_idx+1}_seed_{seed}_log.txt')

        with open(run_log_file, 'w') as log_f:
            log_f.write(f"Enhanced Run {run_idx + 1} with Seed {seed}\n\n")

            for fold_idx, subject_to_leave_out in enumerate(all_subjects):
                log_f.write(f"--- Fold {fold_idx+1}/{len(all_subjects)}: Leaving out subject {subject_to_leave_out} ---\n")
                print(f"--- Fold {fold_idx+1}/{len(all_subjects)}: Leaving out subject {subject_to_leave_out} ---")

                train_samples = [s for s in all_samples if s['subject'] != subject_to_leave_out]
                val_samples = [s for s in all_samples if s['subject'] == subject_to_leave_out]

                train_dataset = CASME2Dataset(train_samples, transform=train_transform)
                val_dataset = CASME2Dataset(val_samples, transform=val_transform)

                g = torch.Generator()
                g.manual_seed(seed)
                
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                        num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                      num_workers=0, pin_memory=True)

                if len(val_loader) == 0:
                    warning_msg = f"WARNING: Skipping fold {fold_idx+1} for subject {subject_to_leave_out} due to empty validation set.\n"
                    print(warning_msg.strip())
                    log_f.write(warning_msg)
                    continue

                num_classes = len(CLASS_NAMES)
                # Use enhanced model
                model = resnet34_enhanced(num_classes=num_classes, in_channels=3, dropout_p=0.4).to(device)
                
                if run_idx == 0 and fold_idx == 0:
                     print(f"Initial enhanced model weights hash: {get_model_weights_hash(model)}")

                # Enhanced loss function combination
                focal_criterion = FocalLoss(gamma=2.5, alpha=None)  # Increased gamma
                smooth_criterion = AdaptiveLabelSmoothingCrossEntropy(initial_smoothing=0.2, min_smoothing=0.05)
                ce_criterion = nn.CrossEntropyLoss(reduction='none')
                
                # MixUp and CutMix
                mixup = MixUp(alpha=MIXUP_ALPHA)
                cutmix = CutMix(alpha=CUTMIX_ALPHA)
                
                def enhanced_combined_loss(outputs, targets, aux_outputs=None, epoch=0):
                    # Update adaptive smoothing
                    smooth_criterion.update_smoothing(epoch, NUM_EPOCHS)
                    
                    focal_loss = focal_criterion(outputs, targets)
                    smooth_loss = smooth_criterion(outputs, targets)
                    ce_loss = ce_criterion(outputs, targets).mean()
                    
                    # Dynamic loss weighting based on training progress
                    progress = epoch / NUM_EPOCHS
                    focal_weight = 0.6 - 0.2 * progress  # Reduce focal loss weight over time
                    smooth_weight = 0.3 + 0.1 * progress  # Increase smooth loss weight
                    ce_weight = 0.1 + 0.1 * progress     # Increase CE loss weight
                    
                    main_loss = focal_weight * focal_loss + smooth_weight * smooth_loss + ce_weight * ce_loss
                    
                    # Add auxiliary loss if available
                    if aux_outputs is not None:
                        aux_loss = F.cross_entropy(aux_outputs, targets)
                        main_loss += 0.3 * aux_loss
                    
                    return main_loss
                
                criterion = enhanced_combined_loss
                
                # Enhanced optimizer with different parameter groups
                backbone_params = []
                attention_params = []
                classifier_params = []
                
                for name, param in model.named_parameters():
                    if 'classifier' in name or 'fc' in name:
                        classifier_params.append(param)
                    elif 'cbam' in name or 'se' in name or 'attention' in name:
                        attention_params.append(param)
                    else:
                        backbone_params.append(param)
                
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': LEARNING_RATE * 0.8, 'weight_decay': WEIGHT_DECAY},
                    {'params': attention_params, 'lr': LEARNING_RATE * 1.2, 'weight_decay': WEIGHT_DECAY * 0.5},
                    {'params': classifier_params, 'lr': LEARNING_RATE * 1.5, 'weight_decay': WEIGHT_DECAY * 2}
                ], betas=(0.9, 0.999), eps=1e-8)
                
                # Enhanced learning rate scheduling with warm restarts
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
                
                early_stopping = EarlyStopping(patience=40, verbose=True, 
                                             path=os.path.join(RESULT_DIR, f'enhanced_run_{run_idx+1}_fold_{fold_idx+1}_best.pt'))
                
                fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

                # Enhanced class weight calculation
                train_labels = [sample['emotion'] for sample in train_samples]
                class_counts = np.bincount([EMOTION_MAP[label] for label in train_labels], minlength=num_classes)
                total_samples = len(train_samples)
                
                # Use effective number of samples with higher beta
                beta = 0.99999
                effective_num = 1.0 - np.power(beta, class_counts)
                class_weights = (1.0 - beta) / np.array(effective_num)
                
                # Enhanced weight processing
                class_weights = class_weights / class_weights.sum() * num_classes
                class_weights = np.sqrt(class_weights)  # Square root for gentler weighting
                class_weights = class_weights / class_weights.sum()
                
                class_weights_tensor = torch.FloatTensor(class_weights).to(device)
                focal_criterion.alpha = class_weights_tensor
                
                print(f"Class counts: {class_counts}")
                print(f"Enhanced class weights: {class_weights}")

                for epoch in range(NUM_EPOCHS):
                    model.train()
                    train_loss, train_correct, train_total = 0, 0, 0
                    
                    for batch_idx, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(device), labels.to(device)

                        # Apply MixUp or CutMix randomly
                        use_mixup = np.random.rand() < AUG_PROB
                        use_cutmix = np.random.rand() < AUG_PROB and not use_mixup
                        
                        optimizer.zero_grad()
                        
                        if use_mixup:
                            mixed_images, y_a, y_b, lam = mixup(images, labels)
                            outputs = model(mixed_images, return_aux=True)
                            if isinstance(outputs, tuple):
                                outputs, aux_outputs = outputs
                                loss = mixup_criterion(lambda o, t: criterion(o, t, aux_outputs, epoch), 
                                                     outputs, y_a, y_b, lam)
                            else:
                                loss = mixup_criterion(lambda o, t: criterion(o, t, None, epoch), 
                                                     outputs, y_a, y_b, lam)
                        elif use_cutmix:
                            mixed_images, y_a, y_b, lam = cutmix(images, labels)
                            outputs = model(mixed_images, return_aux=True)
                            if isinstance(outputs, tuple):
                                outputs, aux_outputs = outputs
                                loss = mixup_criterion(lambda o, t: criterion(o, t, aux_outputs, epoch), 
                                                     outputs, y_a, y_b, lam)
                            else:
                                loss = mixup_criterion(lambda o, t: criterion(o, t, None, epoch), 
                                                     outputs, y_a, y_b, lam)
                        else:
                            outputs = model(images, return_aux=True)
                            if isinstance(outputs, tuple):
                                outputs, aux_outputs = outputs
                                loss = criterion(outputs, labels, aux_outputs, epoch)
                            else:
                                loss = criterion(outputs, labels, None, epoch)
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                        optimizer.step()
                        
                        train_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += labels.size(0)
                        
                        if use_mixup or use_cutmix:
                            # For mixed samples, use the dominant label for accuracy calculation
                            if lam > 0.5:
                                train_correct += (predicted == y_a).sum().item()
                            else:
                                train_correct += (predicted == y_b).sum().item()
                        else:
                            train_correct += (predicted == labels).sum().item()

                    # Validation phase
                    model.eval()
                    val_loss, val_correct, val_total = 0, 0, 0
                    val_predictions, val_true_labels = [], []
                    
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]  # Take main output only
                            
                            loss = F.cross_entropy(outputs, labels)
                            val_loss += loss.item() * images.size(0)
                            
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                            
                            val_predictions.extend(predicted.cpu().numpy())
                            val_true_labels.extend(labels.cpu().numpy())

                    # Calculate metrics
                    train_loss /= train_total
                    val_loss /= val_total
                    train_acc = train_correct / train_total
                    val_acc = val_correct / val_total
                    
                    # Update learning rate
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Store history
                    fold_history['train_loss'].append(train_loss)
                    fold_history['val_loss'].append(val_loss)
                    fold_history['train_acc'].append(train_acc)
                    fold_history['val_acc'].append(val_acc)
                    fold_history['lr'].append(current_lr)

                    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
                    
                    log_f.write(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                               f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}\n')
                    log_f.flush()

                    # Early stopping
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        log_f.write(f"Early stopping at epoch {epoch+1}\n")
                        break

                # Load best model and get final predictions
                model.load_state_dict(torch.load(early_stopping.path))
                model.eval()
                
                fold_predictions = []
                fold_true_labels = []
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        _, predicted = torch.max(outputs.data, 1)
                        fold_predictions.extend(predicted.cpu().numpy())
                        fold_true_labels.extend(labels.cpu().numpy())

                run_pred_labels.extend(fold_predictions)
                run_true_labels.extend(fold_true_labels)
                
                # Clean up model file
                if os.path.exists(early_stopping.path):
                    os.remove(early_stopping.path)

        # Calculate metrics for this run
        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, CLASS_NAMES)
        all_runs_metrics.append(run_metrics)
        
        print(f"\n--- Run {run_idx + 1} Results ---")
        print(f"Accuracy: {run_metrics['accuracy']:.4f}")
        print(f"F1 Score: {run_metrics['f1_score']:.4f}")
        print(f"UAR: {run_metrics['uar']:.4f}")

    # Save final aggregated results
    final_metrics_file = os.path.join(RESULT_DIR, 'enhanced_final_metrics.txt')
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, final_metrics_file)
    
    # Save detailed results
    with open(os.path.join(RESULT_DIR, 'enhanced_detailed_results.json'), 'w') as f:
        json.dump({
            'all_runs_metrics': all_runs_metrics,
            'config': config
        }, f, indent=4, default=str)
    
    print(f"\nEnhanced training completed! Results saved to {RESULT_DIR}")


if __name__ == '__main__':
    main()