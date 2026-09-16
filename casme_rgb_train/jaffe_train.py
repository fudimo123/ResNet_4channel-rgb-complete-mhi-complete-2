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

# 导入 ResNet-18（复用 CASME2 RGB 模型实现）
try:
    if __package__ is None or __package__ == '':
        # 直接运行脚本模式：从同目录下导入
        from model import resnet18
        from jaffe_dataset import JAFFEDataset
    else:
        # 作为包模块运行：使用相对导入
        from .model import resnet18
        from .jaffe_dataset import JAFFEDataset
except ImportError:
    # 回退方案：把项目根目录加入路径后使用绝对包名导入
    PROJECT_ROOT_FALLBACK = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT_FALLBACK not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_FALLBACK)
    from casme_rgb_train.model import resnet18
    from casme_rgb_train.jaffe_dataset import JAFFEDataset

# --- Configuration ---
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, 'data', 'jaffe'))
RESULT_DIR = os.path.join(os.path.dirname(__file__), 'jaffe_rgb')
RESULT_NAME = 'result_jaffe_loso_resnet18'
NUM_EPOCHS = 40
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]

# Map JAFFE emotions to 3 classes
EMOTION_MAP = {
    'happiness': 0,  # positive
    'anger': 1,      # negative
    'disgust': 1,    # negative
    'fear': 1,       # negative
    'sadness': 1,    # negative
    'surprise': 2    # surprise
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

def get_data_hash(samples):
    m = hashlib.md5()
    for s in samples:
        m.update(s['path'].encode('utf-8'))
        m.update(str(s['subject']).encode('utf-8'))
        m.update(str(s['label']).encode('utf-8'))
    return m.hexdigest()

# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Training and Eval ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, total_correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, total_correct / total

# --- Plotting ---
def plot_average_learning_curves(history, result_dir):
    """Plots and saves the average learning curves for loss and accuracy."""
    if len(history) == 0:
        return
    avg_history = {}
    for key in history[0].keys():
        max_len = max(len(h[key]) for h in history)
        padded_histories = [np.pad(h[key], (0, max_len - len(h[key])), 'edge') for h in history]
        avg_history[key] = np.mean(padded_histories, axis=0)

    epochs = range(1, len(avg_history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Average Learning Curves', fontsize=16)

    ax1.plot(epochs, avg_history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, avg_history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.minorticks_on()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

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
    # 自动创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, RESULT_NAME), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build samples
    samples = JAFFEDataset.build_samples(DATA_DIR, EMOTION_MAP)
    subjects = JAFFEDataset.get_all_subjects(samples)
    print(f"Total samples: {len(samples)}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Subjects: {subjects}")

    # Save config
    config = {
        'DATA_DIR': DATA_DIR,
        'RESULT_DIR': RESULT_NAME,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'WARMUP_EPOCHS': WARMUP_EPOCHS,
        'CLIP_GRAD_NORM': CLIP_GRAD_NORM,
        'MODEL': 'ResNet-18',
        'EMOTION_MAP': EMOTION_MAP,
        'CLASS_NAMES': CLASS_NAMES,
        'train_transform': 'Resize(224), HFlip(0.5), Rotation(5), ColorJitter(0.2,0.2), Normalize Imagenet',
        'val_transform': 'Resize(224), Normalize Imagenet'
    }
    with open(os.path.join(RESULT_DIR, RESULT_NAME, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    run_avg_accs = []
    all_runs_histories = []

    # LOSO over multiple runs
    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS, start=1):
        print(f"\n=== Run {run_idx}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)

        # Prepare per-run logging directory
        run_dir = os.path.join(RESULT_DIR, RESULT_NAME)
        log_path = os.path.join(run_dir, f"run_{run_idx}_seed_{seed}_log.txt")

        # Warmup scheduler setup
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
            return 1.0

        fold_best_accs = []

        # LOSO folds
        for fold_idx, test_subject in enumerate(subjects, start=1):
            print(f"\n--- Fold {fold_idx}/{len(subjects)}: Test Subject {test_subject} ---")
            with open(log_path, 'a') as lf:
                lf.write(f"\n--- Fold {fold_idx}/{len(subjects)}: Test Subject {test_subject} ---\n")

            train_subjects = [s for s in subjects if s != test_subject]
            train_samples = [s for s in samples if s['subject'] in train_subjects]
            val_samples = [s for s in samples if s['subject'] == test_subject]
            print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
            with open(log_path, 'a') as lf:
                lf.write(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}\n")
            if len(val_samples) == 0:
                print(f"No validation samples for subject {test_subject}, skipping...")
                with open(log_path, 'a') as lf:
                    lf.write(f"No validation samples for subject {test_subject}, skipping...\n")
                continue

            # Datasets and loaders
            train_dataset = JAFFEDataset(train_samples, transform=train_transform)
            val_dataset = JAFFEDataset(val_samples, transform=val_transform)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

            # Model, criterion, optimizer, schedulers
            num_classes = len(CLASS_NAMES)
            model = resnet18(num_classes=num_classes, pretrained=True, in_channels=3).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - WARMUP_EPOCHS))
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            # History for this fold
            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            best_acc, best_path = 0.0, None
            for epoch in range(NUM_EPOCHS):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                # Adjust LR after optimizer.step(), following best practice
                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                # Record history
                fold_history['train_loss'].append(train_loss)
                fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc)
                fold_history['val_acc'].append(val_acc)

                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}")
                with open(log_path, 'a') as lf:
                    lf.write(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f}\n")

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_path = os.path.join(run_dir, f"run_{run_idx}_fold_{fold_idx}_best.pt")
                    torch.save(model.state_dict(), best_path)

            # Append this fold's history
            all_runs_histories.append(fold_history)

            fold_best_accs.append(best_acc)
            print(f"Fold {fold_idx} best Val Acc: {best_acc:.4f}")
            with open(log_path, 'a') as lf:
                lf.write(f"Fold {fold_idx} best Val Acc: {best_acc:.4f}\n")

        # Final training on all samples（可选，用于迁移学习初始化）
        all_dataset = JAFFEDataset(samples, transform=train_transform)
        all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        final_model = resnet18(num_classes=len(CLASS_NAMES), pretrained=True, in_channels=3).to(device)
        final_optim = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        final_scheduler = CosineAnnealingLR(final_optim, T_max=NUM_EPOCHS)
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(final_model, all_loader, nn.CrossEntropyLoss(), final_optim, device)
            # Adjust LR after optimizer.step()
            final_scheduler.step()
        final_model_path = os.path.join(run_dir, 'resnet18_jaffe_final_model_for_transfer.pth')
        torch.save(final_model.state_dict(), final_model_path)
        # 同时保存一个用于 GUI/调用的同名风格权重文件，保持与 result9 一致
        gui_model_path = os.path.join(run_dir, 'resnet18_jaffe_final_model_for_gui.pth')
        torch.save(final_model.state_dict(), gui_model_path)
        print(f"Saved final model to {final_model_path} and {gui_model_path}")
        with open(log_path, 'a') as lf:
            lf.write(f"Saved final model to {final_model_path} and {gui_model_path}\n")

        # 每个 run 的 LOSO 平均最佳准确率
        if len(fold_best_accs) > 0:
            run_avg_acc = sum(fold_best_accs) / len(fold_best_accs)
        else:
            run_avg_acc = 0.0
        run_avg_accs.append(run_avg_acc)
        print(f"Run {run_idx} LOSO Avg Best Val Acc: {run_avg_acc:.4f}")
        with open(log_path, 'a') as lf:
            lf.write(f"Run {run_idx} LOSO Avg Best Val Acc: {run_avg_acc:.4f}\n")

    # 保存综合指标
    final_metrics_path = os.path.join(RESULT_DIR, RESULT_NAME, 'final_metrics.txt')
    overall_avg = (sum(run_avg_accs) / len(run_avg_accs)) if len(run_avg_accs) > 0 else 0.0
    with open(final_metrics_path, 'w') as f:
        f.write("JAFFE LOSO ResNet-18 Results\n")
        for i, acc in enumerate(run_avg_accs, start=1):
            f.write(f"Run {i} Avg Best Val Acc: {acc:.4f}\n")
        f.write(f"Overall Avg Across Runs: {overall_avg:.4f}\n")
    print(f"Saved final metrics to {final_metrics_path}")

    # 绘制并保存学习曲线图，保持与 result9 一致
    plot_average_learning_curves(all_runs_histories, os.path.join(RESULT_DIR, RESULT_NAME))

if __name__ == '__main__':
    main()