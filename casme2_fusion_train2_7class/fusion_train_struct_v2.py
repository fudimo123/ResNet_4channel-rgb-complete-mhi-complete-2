import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import json
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler

from casme2_data_parser import CASME2DataParser
from fusion_dataset_struct_v2 import CASME2FusionDataset
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping
from center_loss import CenterLoss
from fusion_model_spp_struct_v2 import create_fusion_model

RAW_VIDEO_DIR = '../data/CASME2_RAW_selected/CASME2_RAW_selected'
ANNOTATION_FILE = '../data/CASME2_RAW_selected/CASME2-coding-20140508.xlsx'
DYNAMIC_IMAGE_DIR = './dynamic_data'
RESULT_DIR = 'fusion_result'
NUM_EPOCHS = 90
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 12
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]
LOGIT_TAU = 0.3
CENTER_LOSS_WEIGHT = 0.02

EMOTION_MAP = {
    'happiness': 0,
    'disgust': 1,
    'repression': 2,
    'sadness': 3,
    'fear': 4,
    'surprise': 5,
    'others': 6
}
CLASS_NAMES = ['happiness', 'disgust', 'repression', 'sadness', 'fear', 'surprise', 'others']

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
    import numpy as np
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

def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    rgb_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    rgb_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dynamic_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dynamic_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    config = {
        'RESULT_DIR': RESULT_DIR,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'ResNet-18 + SPP (1,2,3,4) + CBAM + Gated MLP + CenterLoss + AMP',
        'fusion_architecture': 'Late fusion with gating; 1024→512→7; UAR early stop; fixed alpha; logit adjustment; weighted sampler'
    }
    with open(os.path.join(RESULT_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    data_parser = CASME2DataParser(ANNOTATION_FILE, RAW_VIDEO_DIR, EMOTION_MAP)
    all_samples = data_parser.get_samples()
    all_subjects = data_parser.get_all_subjects()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_runs_metrics = []
    all_runs_histories = []
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        set_seed(seed)
        run_true_labels, run_pred_labels = [], []
        run_histories = []
        for fold_idx, test_subject in enumerate(all_subjects):
            train_subjects = [s for s in all_subjects if s != test_subject]
            train_samples = [s for s in all_samples if s['subject'] in train_subjects]
            val_samples = [s for s in all_samples if s['subject'] == test_subject]
            if len(val_samples) == 0:
                continue
            train_dataset = CASME2FusionDataset(train_samples, rgb_transform=rgb_train_transform, dynamic_transform=dynamic_train_transform, dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            val_dataset = CASME2FusionDataset(val_samples, rgb_transform=rgb_val_transform, dynamic_transform=dynamic_val_transform, dynamic_image_dir=DYNAMIC_IMAGE_DIR)
            g = torch.Generator(); g.manual_seed(seed)
            counts = np.bincount([s['label'] for s in train_samples], minlength=len(CLASS_NAMES)).astype(np.float32)
            sample_weights = [1.0 / max(counts[int(s['label'])], 1.0) for s in train_samples]
            sample_weights = np.array(sample_weights, dtype=np.float32)
            sample_weights = sample_weights / sample_weights.sum()
            sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sample_weights), num_samples=len(train_samples), replacement=True, generator=g)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            model = create_fusion_model(num_classes=len(CLASS_NAMES), dropout_p=0.5, pretrained=True).to(device)
            criterion = FocalLoss(gamma=2, label_smoothing=0.1)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            center_loss = CenterLoss(num_classes=len(CLASS_NAMES), feat_dim=512).to(device)
            center_optimizer = optim.SGD(center_loss.parameters(), lr=0.05)
            scaler = GradScaler(enabled=torch.cuda.is_available())
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            early_stopping = EarlyStopping(patience=45, verbose=True, path=os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt'))
            inv = 1.0 / np.where(counts > 0, counts, 1.0); inv = inv / inv.sum()
            alpha = torch.tensor(inv, dtype=torch.float32, device=device)
            criterion.alpha = alpha
            prior = counts / max(counts.sum(), 1.0)
            prior_log = torch.tensor(np.log(np.where(prior>0, prior, 1e-8)), dtype=torch.float32, device=device)
            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                for rgb_images, dynamic_images, labels in train_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any():
                        continue
                    rgb_images = rgb_images[valid_mask]
                    dynamic_images = dynamic_images[valid_mask]
                    labels = labels[valid_mask]
                    rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.long().to(device)
                    optimizer.zero_grad()
                    center_optimizer.zero_grad()
                    with autocast(enabled=torch.cuda.is_available()):
                        logits, feats = model(rgb_images, dynamic_images, return_feat=True)
                        logits = logits + LOGIT_TAU * prior_log
                        loss_cls = criterion(logits, labels)
                        loss_center = center_loss(feats, labels)
                        loss = loss_cls + CENTER_LOSS_WEIGHT * loss_center
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.step(center_optimizer)
                    scaler.update()
                    train_loss += loss.item() * rgb_images.size(0)
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()
                model.eval()
                val_loss, val_correct, val_total = 0, 0, 0
                epoch_true_labels, epoch_pred_labels = [], []
                with torch.no_grad():
                    for rgb_images, dynamic_images, labels in val_loader:
                        valid_mask = labels != -1
                        if not valid_mask.any():
                            continue
                        rgb_images = rgb_images[valid_mask]
                        dynamic_images = dynamic_images[valid_mask]
                        labels = labels[valid_mask]
                        rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.long().to(device)
                        with autocast(enabled=torch.cuda.is_available()):
                            logits, _ = model(rgb_images, dynamic_images, return_feat=True)
                            logits = logits + LOGIT_TAU * prior_log
                            loss = criterion(logits, labels)
                        val_loss += loss.item() * rgb_images.size(0)
                        _, predicted = torch.max(logits.data, 1)
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
                metrics_epoch = calculate_metrics(epoch_true_labels, epoch_pred_labels, labels=list(range(len(CLASS_NAMES))), class_names=CLASS_NAMES)
                early_stopping(metrics_epoch['uar'], model)
                if early_stopping.early_stop:
                    break
            run_histories.append(fold_history)
            model.load_state_dict(torch.load(os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt')))
            model.eval()
            with torch.no_grad():
                for rgb_images, dynamic_images, labels in val_loader:
                    rgb_images, dynamic_images, labels = rgb_images.to(device), dynamic_images.to(device), labels.to(device)
                    logits, _ = model(rgb_images, dynamic_images, return_feat=True)
                    logits = logits + LOGIT_TAU * prior_log
                    _, predicted = torch.max(logits.data, 1)
                    run_true_labels.extend(labels.cpu().numpy())
                    run_pred_labels.extend(predicted.cpu().numpy())
        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))), class_names=CLASS_NAMES)
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)
        plot_confusion_matrix(run_metrics['confusion_matrix'], CLASS_NAMES, os.path.join(RESULT_DIR, f'run_{run_idx+1}_confusion_matrix.png'))
    save_metrics_with_std(all_runs_metrics, CLASS_NAMES, os.path.join(RESULT_DIR, 'final_metrics.txt'))
    import matplotlib.pyplot as plt
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)

if __name__ == '__main__':
    main()

