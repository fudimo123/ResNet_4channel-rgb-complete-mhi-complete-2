import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import hashlib
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from PIL import Image

from casme2_data_parser import CASME2DataParser
from fusion_dataset import CASME2FusionDataset
from fusion_model import create_fusion_model
from metrics import calculate_metrics, plot_confusion_matrix
from focal_loss import FocalLoss
from early_stopping import EarlyStopping

# --- Configuration ---
RAW_VIDEO_DIR = '../data/CASME2_RAW_selected/CASME2_RAW_selected'
ANNOTATION_FILE = '../data/CASME2_RAW_selected/CASME2-coding-20140508.xlsx'
RESULT_DIR = 'fusion_result2'  # Changed to fusion_result2
NUM_EPOCHS = 40  # Reduced from 60 to 40
BATCH_SIZE = 32  # Increased from 16 to 32
LEARNING_RATE = 0.0005  # Reduced learning rate
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 8  # Longer warmup
MHI_SAVE_DIR = 'casme2_mhi'  # Changed to specified directory
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

    # Save to file
    with open(filepath, 'w') as f:
        f.write("=== FINAL METRICS (Mean ± Std over multiple runs) ===\n\n")
        
        f.write("Overall Metrics:\n")
        for key in overall_metric_keys:
            if key in mean_metrics:
                f.write(f"{key.upper()}: {mean_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n")
        
        f.write("\nPer-Class Metrics:\n")
        for c_name in class_names:
            f.write(f"\n{c_name.upper()}:\n")
            for pc_key in per_class_metric_keys:
                if pc_key in mean_metrics['per_class_metrics'][c_name]:
                    f.write(f"  {pc_key}: {mean_metrics['per_class_metrics'][c_name][pc_key]:.4f} ± {std_metrics['per_class_metrics'][c_name][pc_key]:.4f}\n")

def plot_average_learning_curves(history, result_dir):
    """Plot average learning curves across all folds and runs."""
    if not history:
        return
    
    # Aggregate all histories
    all_train_loss = []
    all_val_loss = []
    all_train_acc = []
    all_val_acc = []
    
    max_epochs = max(len(h['train_loss']) for h in history)
    
    for epoch in range(max_epochs):
        epoch_train_loss = [h['train_loss'][epoch] for h in history if epoch < len(h['train_loss'])]
        epoch_val_loss = [h['val_loss'][epoch] for h in history if epoch < len(h['val_loss'])]
        epoch_train_acc = [h['train_acc'][epoch] for h in history if epoch < len(h['train_acc'])]
        epoch_val_acc = [h['val_acc'][epoch] for h in history if epoch < len(h['val_acc'])]
        
        if epoch_train_loss:
            all_train_loss.append(np.mean(epoch_train_loss))
            all_val_loss.append(np.mean(epoch_val_loss))
            all_train_acc.append(np.mean(epoch_train_acc))
            all_val_acc.append(np.mean(epoch_val_acc))
    
    epochs = range(1, len(all_train_loss) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, all_train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, all_val_loss, 'r-', label='Validation Loss')
    plt.title('Average Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, all_train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, all_val_acc, 'r-', label='Validation Accuracy')
    plt.title('Average Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'average_learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def pregenerate_mhi_data(data_parser, mhi_save_dir):
    """Pre-generate all MHI data and save to disk."""
    print("\n=== Pre-generating MHI data ===")
    
    # Import MHI generator
    import sys
    sys.path.append('../casme2_mhi_train')
    from mhi_generator import MHIGenerator
    
    mhi_generator = MHIGenerator(duration=40)
    all_samples = data_parser.get_samples()
    
    os.makedirs(mhi_save_dir, exist_ok=True)
    
    generated_count = 0
    skipped_count = 0
    
    for i, sample in enumerate(all_samples):
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']
        subject = sample.get('subject', 'unknown')
        
        # Create subject directory
        subject_dir = os.path.join(mhi_save_dir, f"sub{subject:02d}" if isinstance(subject, int) else str(subject))
        os.makedirs(subject_dir, exist_ok=True)
        
        # Generate filename based on sequence info
        sequence_name = os.path.basename(sequence_path)
        mhi_filename = f"{sequence_name}_onset{onset}_offset{offset}_mhi.png"
        mhi_path = os.path.join(subject_dir, mhi_filename)
        
        # Skip if MHI already exists
        if os.path.exists(mhi_path):
            skipped_count += 1
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(all_samples)} samples processed (Generated: {generated_count}, Skipped: {skipped_count})")
            continue
        
        try:
            # Generate MHI image
            mhi_image = mhi_generator.generate_mhi_from_sequence(sequence_path, onset, offset)
            
            # Save MHI image
            mhi_image.save(mhi_path)
            generated_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{len(all_samples)} samples processed (Generated: {generated_count}, Skipped: {skipped_count})")
                
        except Exception as e:
            print(f"Error generating MHI for {sequence_path}: {e}")
            continue
    
    print(f"\nMHI pre-generation complete!")
    print(f"Generated: {generated_count} new MHI images")
    print(f"Skipped: {skipped_count} existing MHI images")
    print(f"Total samples: {len(all_samples)}")
    print(f"MHI data saved to: {mhi_save_dir}")

class OptimizedCASME2FusionDataset(Dataset):
    """Optimized dataset class that loads pre-generated MHI images."""
    
    def __init__(self, samples, rgb_transform=None, mhi_transform=None, mhi_save_dir=None):
        self.samples = samples
        self.rgb_transform = rgb_transform
        self.mhi_transform = mhi_transform
        self.mhi_save_dir = mhi_save_dir
        
        # Initialize MTCNN for face detection (for RGB images)
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=False, device='cpu')
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_face_roi(self, image):
        """Extract face ROI using MTCNN, similar to RGB training."""
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Add some margin around the face
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.width, x2 + margin)
            y2 = min(image.height, y2 + margin)
            
            # Crop the face region
            face_image = image.crop((x1, y1, x2, y2))
            return face_image
        else:
            # If no face detected, return the original image
            return image
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        sequence_path = sample['raw_video_path']
        onset = sample['onset']
        offset = sample['offset']
        subject = sample.get('subject', 'unknown')
        
        try:
            frame_files = sorted(os.listdir(sequence_path))
        except FileNotFoundError:
            print(f"Directory not found: {sequence_path}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # Find valid frames within onset-offset range
        valid_frames = []
        for f in frame_files:
            if f.startswith('img') and f.endswith('.jpg'):
                try:
                    frame_num = int(f[3:-4])
                    if onset <= frame_num <= offset:
                        valid_frames.append(f)
                except ValueError:
                    continue
        
        if not valid_frames:
            print(f"No valid frames found for {sequence_path} between {onset} and {offset}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # --- Load RGB Image ---
        selected_frame_file = random.choice(valid_frames)
        rgb_image_path = os.path.join(sequence_path, selected_frame_file)
        
        try:
            rgb_image = Image.open(rgb_image_path).convert('RGB')
            rgb_image = self._extract_face_roi(rgb_image)
        except FileNotFoundError:
            print(f"RGB image file not found: {rgb_image_path}")
            return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224), -1
        
        # --- Load Pre-generated MHI Image ---
        try:
            # Generate MHI filename
            subject_dir = os.path.join(self.mhi_save_dir, f"sub{subject:02d}" if isinstance(subject, int) else str(subject))
            sequence_name = os.path.basename(sequence_path)
            mhi_filename = f"{sequence_name}_onset{onset}_offset{offset}_mhi.png"
            mhi_path = os.path.join(subject_dir, mhi_filename)
            
            # Load pre-generated MHI image
            if os.path.exists(mhi_path):
                mhi_image = Image.open(mhi_path).convert('L')  # Convert to grayscale
            else:
                print(f"Pre-generated MHI not found: {mhi_path}")
                # Fallback: create a blank MHI image
                mhi_image = Image.new('L', (224, 224), 0)
                
        except Exception as e:
            print(f"Error loading MHI for {sequence_path}: {e}")
            mhi_image = Image.new('L', (224, 224), 0)
        
        # Apply transforms
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
        
        if self.mhi_transform:
            mhi_image = self.mhi_transform(mhi_image)
        
        return rgb_image, mhi_image, label

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

    # --- Transformations for MHI (single channel) ---
    mhi_train_transform_dict = {
        "Resize": {"size": (224, 224)},
        "RandomRotation": {"degrees": 3},
        "ToTensor": {},
        "Normalize": {"mean": [0.5], "std": [0.5]}
    }
    mhi_train_transform = transforms.Compose([
        transforms.Resize(mhi_train_transform_dict["Resize"]["size"]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mhi_train_transform_dict["Normalize"]["mean"], 
                           std=mhi_train_transform_dict["Normalize"]["std"])
    ])

    mhi_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
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
        'MODEL': 'Optimized Late Fusion ResNet-18 with Pre-generated MHI',
        'rgb_train_transform': rgb_train_transform_dict,
        'mhi_train_transform': mhi_train_transform_dict,
        'fusion_architecture': 'Late fusion at feature level (RGB: 512 + MHI: 512 -> 1024 -> 3 classes)',
        'optimizations': {
            'pre_generated_mhi': True,
            'reduced_epochs': f'{NUM_EPOCHS} (from 60)',
            'increased_batch_size': f'{BATCH_SIZE} (from 16)',
            'early_stopping_patience': '15 (from 30)'
        }
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

    # --- Pre-generate MHI Data ---
    pregenerate_mhi_data(data_parser, MHI_SAVE_DIR)

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
            
            # Create optimized datasets (using pre-generated MHI)
            train_dataset = OptimizedCASME2FusionDataset(train_samples, 
                                                      rgb_transform=rgb_train_transform, 
                                                      mhi_transform=mhi_train_transform,
                                                      mhi_save_dir=MHI_SAVE_DIR)
            val_dataset = OptimizedCASME2FusionDataset(val_samples, 
                                                    rgb_transform=rgb_val_transform, 
                                                    mhi_transform=mhi_val_transform,
                                                    mhi_save_dir=MHI_SAVE_DIR)
            
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
            
            # Updated early stopping with patience=15
            early_stopping = EarlyStopping(patience=15, verbose=True, 
                                         path=os.path.join(RESULT_DIR, f'run_{run_idx+1}_fold_{fold_idx+1}_best.pt'))

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            for epoch in range(NUM_EPOCHS):
                model.train()
                train_loss, train_correct, train_total = 0, 0, 0
                
                # Dynamic alpha for Focal Loss based on current batch
                for rgb_images, mhi_images, labels in train_loader:
                    rgb_images, mhi_images, labels = rgb_images.to(device), mhi_images.to(device), labels.to(device)
                    
                    class_counts = torch.bincount(labels, minlength=len(CLASS_NAMES)).float()
                    class_weights = 1. / torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts)).to(device)
                    criterion.alpha = class_weights / class_weights.sum()

                    optimizer.zero_grad()
                    outputs = model(rgb_images, mhi_images)
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
                    for rgb_images, mhi_images, labels in val_loader:
                        rgb_images, mhi_images, labels = rgb_images.to(device), mhi_images.to(device), labels.to(device)
                        outputs = model(rgb_images, mhi_images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * rgb_images.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        epoch_true_labels.extend(labels.cpu().numpy())
                        epoch_pred_labels.extend(predicted.cpu().numpy())

                train_loss /= len(train_dataset)
                val_loss /= len(val_dataset)
                train_acc = train_correct / train_total
                val_acc = val_correct / val_total
                
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
                for rgb_images, mhi_images, labels in val_loader:
                    rgb_images, mhi_images, labels = rgb_images.to(device), mhi_images.to(device), labels.to(device)
                    outputs = model(rgb_images, mhi_images)
                    _, predicted = torch.max(outputs.data, 1)
                    run_true_labels.extend(labels.cpu().numpy())
                    run_pred_labels.extend(predicted.cpu().numpy())

        run_metrics = calculate_metrics(run_true_labels, run_pred_labels, labels=list(range(len(CLASS_NAMES))))
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
    
    # Use optimized dataset for final training (no MHI generation needed)
    final_train_dataset = OptimizedCASME2FusionDataset(all_samples, 
                                                    rgb_transform=rgb_train_transform, 
                                                    mhi_transform=mhi_train_transform,
                                                    mhi_save_dir=MHI_SAVE_DIR)
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

    final_model_path = os.path.join(RESULT_DIR, 'fusion_resnet18_casme2_final_model_optimized.pth')
    torch.save(final_model.state_dict(), final_model_path)
    print(f"--- Final optimized fusion model saved to {final_model_path} ---")


if __name__ == '__main__':
    main()