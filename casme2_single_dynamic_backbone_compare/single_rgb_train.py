import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms

from casme2_processed_parser import CASME2ProcessedDataParser
from early_stopping import EarlyStopping
from focal_loss_ls import FocalLossLabelSmoothing
from metrics import calculate_metrics, plot_confusion_matrix
from single_rgb_dataset import CASME2SingleRGBDataset
from single_rgb_models import create_single_rgb_model


RESULT_DIR = "rgb_single_result"
MODEL_NAME = "plain_resnet18"
TRANSFER_WEIGHTS_PATH = None
PROCESSED_RGB_DIR = "./casme2_aligned_rgb_offline"
MANIFEST_FILE = "./casme2_single_rgb_manifest.csv"
NUM_EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8
CLIP_GRAD_NORM = 1.0
REPRODUCIBILITY_SEEDS = [42, 43, 44]
LABEL_SMOOTHING = 0.1

CLASS_NAMES = ["positive", "negative", "surprise"]


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
    mean_metrics = {}
    std_metrics = {}

    overall_metric_keys = ["accuracy", "precision", "recall", "f1_score", "uar", "uf1"]
    overall_metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision (weighted)",
        "recall": "Recall (weighted)",
        "f1_score": "F1 score (weighted)",
        "uar": "UAR",
        "uf1": "UF1",
    }
    per_class_metric_keys = ["precision", "recall", "f1-score"]

    for key in overall_metric_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)

    mean_metrics["per_class_metrics"] = {c: {} for c in class_names}
    std_metrics["per_class_metrics"] = {c: {} for c in class_names}
    for class_name in class_names:
        for metric_key in per_class_metric_keys:
            values = [
                m["per_class_metrics"][class_name][metric_key]
                for m in metrics_list
                if "per_class_metrics" in m
                and class_name in m["per_class_metrics"]
                and metric_key in m["per_class_metrics"][class_name]
            ]
            if values:
                mean_metrics["per_class_metrics"][class_name][metric_key] = np.mean(values)
                std_metrics["per_class_metrics"][class_name][metric_key] = np.std(values)

    cm_list = [m["confusion_matrix"] for m in metrics_list if "confusion_matrix" in m]
    if cm_list:
        cm_stack = np.stack(cm_list, axis=0)
        mean_metrics["confusion_matrix"] = np.mean(cm_stack, axis=0)
        std_metrics["confusion_matrix"] = np.std(cm_stack, axis=0)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"--- Final Metrics (Mean ± Std over {len(metrics_list)} runs) ---\n\n")
        f.write("--- Overall Performance ---\n")
        for key in overall_metric_keys:
            if key in mean_metrics:
                f.write(f"{overall_metric_labels[key]:<22}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
        f.write("\n")

        f.write("--- Per-class Metrics ---\n")
        for class_name in class_names:
            f.write(f"  Class: {class_name}\n")
            for metric_key in per_class_metric_keys:
                if metric_key in mean_metrics["per_class_metrics"][class_name]:
                    mean_val = mean_metrics["per_class_metrics"][class_name][metric_key]
                    std_val = std_metrics["per_class_metrics"][class_name][metric_key]
                    f.write(f"    {metric_key.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

        f.write("--- Confusion Matrix ---\n")
        f.write("Mean:\n")
        f.write(np.array2string(mean_metrics["confusion_matrix"], formatter={"float_kind": lambda x: "%.2f" % x}))
        f.write("\n\nStd Dev:\n")
        f.write(np.array2string(std_metrics["confusion_matrix"], formatter={"float_kind": lambda x: "%.2f" % x}))
        f.write("\n")


def plot_average_learning_curves(history, result_dir):
    avg_history = {}
    for key in history[0]:
        max_len = max(len(h[key]) for h in history)
        padded = [h[key] + [h[key][-1]] * (max_len - len(h[key])) for h in history]
        avg_history[key] = np.mean(padded, axis=0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(avg_history["train_loss"], label="Train Loss")
    plt.plot(avg_history["val_loss"], label="Validation Loss")
    plt.title("Average Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(avg_history["train_acc"], label="Train Accuracy")
    plt.plot(avg_history["val_acc"], label="Validation Accuracy")
    plt.title("Average Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "average_learning_curves.png"))
    plt.close()


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def create_model(num_classes, dropout_p, pretrained):
    return create_single_rgb_model(
        model_name=MODEL_NAME,
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
        transfer_weights_path=TRANSFER_WEIGHTS_PATH,
    )


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    train_transform, val_transform = build_transforms()
    data_parser = CASME2ProcessedDataParser(MANIFEST_FILE)
    all_samples = data_parser.get_samples()
    all_subjects = data_parser.get_all_subjects()

    print(f"Starting CASME2 RGB single-channel training: {MODEL_NAME}")
    print(f"Strategy: Train Random Frame RGB + Val/Test Middle Frame RGB + Label Smoothing ({LABEL_SMOOTHING})")
    print(f"Results will be saved to: {RESULT_DIR}")
    print(f"Using offline RGB root: {PROCESSED_RGB_DIR}")
    print(f"Using manifest file: {MANIFEST_FILE}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Total subjects: {len(all_subjects)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_runs_metrics = []
    all_runs_histories = []

    for run_idx, seed in enumerate(REPRODUCIBILITY_SEEDS):
        print(f"\n=== Run {run_idx + 1}/{len(REPRODUCIBILITY_SEEDS)} (seed={seed}) ===")
        set_seed(seed)

        run_true_labels, run_pred_labels = [], []
        run_histories = []

        for fold_idx, test_subject in enumerate(all_subjects):
            print(f"\n--- Fold {fold_idx + 1}/{len(all_subjects)}: Test Subject {test_subject:02d} ---")
            train_subjects = [s for s in all_subjects if s != test_subject]
            train_samples = [s for s in all_samples if s["subject"] in train_subjects]
            val_samples = [s for s in all_samples if s["subject"] == test_subject]

            train_dataset = CASME2SingleRGBDataset(
                train_samples,
                transform=train_transform,
                frame_selection="random",
                processed_rgb_root=PROCESSED_RGB_DIR,
            )
            val_dataset = CASME2SingleRGBDataset(
                val_samples,
                transform=val_transform,
                frame_selection="middle",
                processed_rgb_root=PROCESSED_RGB_DIR,
            )

            g = torch.Generator()
            g.manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            model = create_model(num_classes=len(CLASS_NAMES), dropout_p=0.5, pretrained=True).to(device)
            criterion = FocalLossLabelSmoothing(gamma=2, smoothing=LABEL_SMOOTHING)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            warmup_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: float(epoch) / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1,
            )
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
            early_stopping = EarlyStopping(patience=30, verbose=True)

            fold_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in train_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any():
                        continue

                    images = images[valid_mask].to(device)
                    labels = labels[valid_mask].long().to(device)

                    class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
                    class_weights = 1.0 / torch.where(
                        class_counts > 0, class_counts, torch.ones_like(class_counts)
                    ).to(device)
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

                if epoch < WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    main_scheduler.step()

                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                epoch_true_labels = []
                epoch_pred_labels = []
                with torch.no_grad():
                    for images, labels in val_loader:
                        valid_mask = labels != -1
                        if not valid_mask.any():
                            continue

                        images = images[valid_mask].to(device)
                        labels = labels[valid_mask].long().to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                        val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        epoch_true_labels.extend(labels.cpu().numpy())
                        epoch_pred_labels.extend(predicted.cpu().numpy())

                train_loss /= max(train_total, 1)
                train_acc = train_correct / max(train_total, 1)
                val_loss /= max(val_total, 1)
                val_acc = val_correct / max(val_total, 1)

                fold_history["train_loss"].append(train_loss)
                fold_history["val_loss"].append(val_loss)
                fold_history["train_acc"].append(train_acc)
                fold_history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                early_stopping(val_acc, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if early_stopping.best_model_state is not None:
                model.load_state_dict(early_stopping.best_model_state)

            model.eval()
            fold_true_labels = []
            fold_pred_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    valid_mask = labels != -1
                    if not valid_mask.any():
                        continue

                    images = images[valid_mask].to(device)
                    labels = labels[valid_mask].long().to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    fold_true_labels.extend(labels.cpu().numpy())
                    fold_pred_labels.extend(predicted.cpu().numpy())

            run_true_labels.extend(fold_true_labels)
            run_pred_labels.extend(fold_pred_labels)
            run_histories.append(fold_history)

        run_metrics = calculate_metrics(
            np.array(run_true_labels),
            np.array(run_pred_labels),
            labels=list(range(len(CLASS_NAMES))),
            class_names=CLASS_NAMES,
        )
        all_runs_metrics.append(run_metrics)
        all_runs_histories.extend(run_histories)

        print(
            f"Run {run_idx + 1} metrics | "
            f"Accuracy: {run_metrics['accuracy']:.4f} | "
            f"UAR: {run_metrics['uar']:.4f} | "
            f"UF1: {run_metrics['uf1']:.4f}"
        )

        plot_confusion_matrix(
            run_metrics["confusion_matrix"],
            CLASS_NAMES,
            os.path.join(RESULT_DIR, f"run_{run_idx + 1}_confusion_matrix.png"),
        )

    save_metrics_with_std(
        all_runs_metrics,
        CLASS_NAMES,
        os.path.join(RESULT_DIR, "final_metrics.txt"),
    )
    plot_average_learning_curves(all_runs_histories, RESULT_DIR)
    print(f"\nTraining completed. Final results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
