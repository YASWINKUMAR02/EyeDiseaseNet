"""
train_idrid_all.py — Fine-tune using ALL IDRiD Parts (A + B + C) + MESSIDOR
=============================================================================
Combines ALL available data for maximum model performance:

Part B (Disease Grading):   413 train + 103 test images with DR grade labels
Part A (Segmentation):      54 train + 27 test images → DR grades inferred from lesion masks
Part C (Localization):      Same 516 images as Part B (already included)
MESSIDOR (existing):        Original training data for combined training

Lesion-to-Grade Mapping (for Part A):
  - No lesions detected           → Grade 0 (No DR)
  - Microaneurysms only           → Grade 1 (Mild NPDR)
  - Haemorrhages OR Hard Exudates → Grade 2 (Moderate NPDR)
  - Multiple lesion types present → Grade 3 (Severe NPDR)
  - All lesion types present      → Grade 4 (Proliferative DR)

Usage:
    python train_idrid_all.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, Subset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
import time
import copy
import csv
from collections import Counter

# ==========================================
# Configuration
# ==========================================
BASE = r"C:\DR_CP"

# Part B: Disease Grading
IDRID_B_BASE      = os.path.join(BASE, "B. Disease Grading", "B. Disease Grading")
IDRID_B_TRAIN_CSV = os.path.join(IDRID_B_BASE, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
IDRID_B_TEST_CSV  = os.path.join(IDRID_B_BASE, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
IDRID_B_TRAIN_IMG = os.path.join(IDRID_B_BASE, "1. Original Images", "a. Training Set")
IDRID_B_TEST_IMG  = os.path.join(IDRID_B_BASE, "1. Original Images", "b. Testing Set")

# Part A: Segmentation
IDRID_A_BASE       = os.path.join(BASE, "A. Segmentation", "A. Segmentation")
IDRID_A_TRAIN_IMG  = os.path.join(IDRID_A_BASE, "1. Original Images", "a. Training Set")
IDRID_A_TEST_IMG   = os.path.join(IDRID_A_BASE, "1. Original Images", "b. Testing Set")
IDRID_A_MASKS_TRAIN = os.path.join(IDRID_A_BASE, "2. All Segmentation Groundtruths", "a. Training Set")
IDRID_A_MASKS_TEST  = os.path.join(IDRID_A_BASE, "2. All Segmentation Groundtruths", "b. Testing Set")

# MESSIDOR (existing data)
MESSIDOR_DIR = os.path.join(BASE, "DE_MESSIDOR_EX", "augmented_resized_V2")

# Model paths
PRETRAINED_MODEL = os.path.join(BASE, "efficientnet_retinal.pth")
OUTPUT_MODEL     = os.path.join(BASE, "efficientnet_retinal_idrid.pth")

NUM_CLASSES    = 5
BATCH_SIZE     = 32
NUM_EPOCHS     = 15
LEARNING_RATE  = 0.0001
VAL_SPLIT      = 0.15


# ==========================================
# Dataset: Part B (Disease Grading from CSV)
# ==========================================
class IDRiDGradingDataset(Dataset):
    """Loads Part B images with DR grade labels from CSV."""
    
    def __init__(self, csv_path, images_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['Image name'].strip()
                dr_grade = int(row['Retinopathy grade'].strip())
                img_path = os.path.join(images_dir, f"{img_name}.jpg")
                if os.path.exists(img_path):
                    self.samples.append((img_path, dr_grade))
        
        dist = Counter(g for _, g in self.samples)
        print(f"  [Part B] Loaded {len(self.samples)} images from {os.path.basename(images_dir)}")
        for k in sorted(dist.keys()):
            print(f"    Grade {k}: {dist[k]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# Dataset: Part A (Segmentation → Inferred DR Grade)
# ==========================================
class IDRiDSegmentationAsGrading(Dataset):
    """
    Loads Part A images and INFERS DR grade from lesion masks.
    
    Logic:
      - Check which lesion masks exist AND have non-zero pixels
      - MA = Microaneurysms, HE = Haemorrhages, EX = Hard Exudates, SE = Soft Exudates
      
      Grade assignment:
        0 = No lesion masks have pixels       (No DR)
        1 = Only MA present                   (Mild NPDR)
        2 = HE or EX present (with or without MA)  (Moderate NPDR)
        3 = 3+ lesion types present           (Severe NPDR)
        4 = All 4 lesion types present        (Proliferative DR)
    """
    
    LESION_DIRS = {
        'MA': '1. Microaneurysms',
        'HE': '2. Haemorrhages',
        'EX': '3. Hard Exudates',
        'SE': '4. Soft Exudates',
    }
    LESION_SUFFIXES = {'MA': '_MA', 'HE': '_HE', 'EX': '_EX', 'SE': '_SE'}
    
    def __init__(self, images_dir, masks_base_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Get all image names
        image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
        
        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]  # e.g., "IDRiD_01"
            img_path = os.path.join(images_dir, img_file)
            
            # Check which lesions are present
            lesions_present = []
            for lesion_key, dir_name in self.LESION_DIRS.items():
                suffix = self.LESION_SUFFIXES[lesion_key]
                mask_path = os.path.join(masks_base_dir, dir_name, f"{img_name}{suffix}.tif")
                
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path).convert('L')
                        mask_arr = np.array(mask)
                        if mask_arr.max() > 0:  # Has actual lesion pixels
                            lesions_present.append(lesion_key)
                    except Exception:
                        pass
            
            # Infer DR grade from lesions
            n_lesions = len(lesions_present)
            if n_lesions == 0:
                grade = 0  # No DR
            elif n_lesions == 1 and 'MA' in lesions_present:
                grade = 1  # Mild — only microaneurysms
            elif n_lesions <= 2:
                grade = 2  # Moderate — haemorrhages or exudates
            elif n_lesions == 3:
                grade = 3  # Severe — multiple lesion types
            else:
                grade = 4  # Proliferative — all lesion types
            
            self.samples.append((img_path, grade, lesions_present))
        
        dist = Counter(g for _, g, _ in self.samples)
        print(f"  [Part A] Inferred {len(self.samples)} images from segmentation masks")
        for k in sorted(dist.keys()):
            print(f"    Grade {k}: {dist[k]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ==========================================
# Training Function
# ==========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=15):
    since = time.time()
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"  {phase}"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=(phase == 'train')):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            print(f'  {phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  --> New best! Val Acc: {best_acc:.4f}")

    elapsed = time.time() - since
    print(f'\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 65)
    print("  COMBINED TRAINING: MESSIDOR + IDRiD (All Parts A + B + C)")
    print("=" * 65)

    # ── Transforms ─────────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── Load ALL Datasets ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  LOADING ALL DATASETS")
    print("=" * 65)

    # 1. MESSIDOR (existing large dataset)
    print("\n--- MESSIDOR Dataset ---")
    messidor_train = datasets.ImageFolder(os.path.join(MESSIDOR_DIR, 'train'), train_transforms)
    messidor_val   = datasets.ImageFolder(os.path.join(MESSIDOR_DIR, 'val'),   val_transforms)
    print(f"  [MESSIDOR] Train: {len(messidor_train)}, Val: {len(messidor_val)}")

    # 2. IDRiD Part B — Disease Grading (CSV labels)
    print("\n--- IDRiD Part B: Disease Grading ---")
    idrid_b_train = IDRiDGradingDataset(IDRID_B_TRAIN_CSV, IDRID_B_TRAIN_IMG, train_transforms)
    idrid_b_test  = IDRiDGradingDataset(IDRID_B_TEST_CSV,  IDRID_B_TEST_IMG,  val_transforms)

    # 3. IDRiD Part A — Segmentation (inferred grades from masks)
    print("\n--- IDRiD Part A: Segmentation (inferring DR grades from masks) ---")
    idrid_a_train = IDRiDSegmentationAsGrading(IDRID_A_TRAIN_IMG, IDRID_A_MASKS_TRAIN, train_transforms)
    idrid_a_test  = IDRiDSegmentationAsGrading(IDRID_A_TEST_IMG,  IDRID_A_MASKS_TEST,  val_transforms)

    # 4. Part C is the same 516 images as Part B — already included!
    print("\n--- IDRiD Part C: Localization ---")
    print("  [Part C] Same 516 images as Part B — already included in training")

    # ── Combine All Training Data ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMBINING ALL DATA")
    print("=" * 65)

    combined_train = ConcatDataset([messidor_train, idrid_b_train, idrid_a_train])
    combined_val   = ConcatDataset([messidor_val, idrid_a_test])

    total_train = len(combined_train)
    total_val   = len(combined_val)

    print(f"\n  Combined Training Set:")
    print(f"    MESSIDOR train:    {len(messidor_train):>6d} images")
    print(f"    IDRiD Part B train: {len(idrid_b_train):>5d} images")
    print(f"    IDRiD Part A train: {len(idrid_a_train):>5d} images")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL TRAIN:       {total_train:>6d} images")
    print(f"\n  Combined Validation Set:")
    print(f"    MESSIDOR val:      {len(messidor_val):>6d} images")
    print(f"    IDRiD Part A test: {len(idrid_a_test):>5d} images")
    print(f"    ─────────────────────────────")
    print(f"    TOTAL VAL:         {total_val:>6d} images")
    print(f"\n  Separate Test Set:")
    print(f"    IDRiD Part B test: {len(idrid_b_test):>5d} images (held out for final evaluation)")

    # ── DataLoaders ────────────────────────────────────────────────
    dataloaders = {
        'train': DataLoader(combined_train, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True),
        'val':   DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True),
    }
    dataset_sizes = {'train': total_train, 'val': total_val}

    # ── Build Model ────────────────────────────────────────────────
    print("\n--- Loading Pre-trained Model ---")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    if os.path.exists(PRETRAINED_MODEL):
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
        print(f"  Loaded weights from: {PRETRAINED_MODEL}")
    else:
        print("  No pre-trained model found, using ImageNet weights")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    model = model.to(device)

    # ── Loss, Optimizer, Scheduler ─────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # Differential learning rates
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STARTING COMBINED TRAINING: {NUM_EPOCHS} epochs")
    print(f"  Total training images: {total_train}")
    print(f"{'='*65}")

    model, history = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, device, num_epochs=NUM_EPOCHS
    )

    # ── Save Model ─────────────────────────────────────────────────
    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"\nModel saved to: {OUTPUT_MODEL}")

    # ── Plot Training History ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train', color='#4f9cf9', linewidth=2)
    ax1.plot(history['val_loss'], label='Val', color='#f87171', linewidth=2)
    ax1.set_title('Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(history['train_acc'], label='Train', color='#4f9cf9', linewidth=2)
    ax2.plot(history['val_acc'], label='Val', color='#34d399', linewidth=2)
    ax2.set_title('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    hist_path = os.path.join(BASE, "idrid_all_training_history.png")
    plt.savefig(hist_path, dpi=150); plt.close()
    print(f"Training history saved: {hist_path}")

    # ── Evaluate on IDRiD Part B Test Set ──────────────────────────
    print(f"\n{'='*65}")
    print("  EVALUATING ON IDRiD TEST SET (103 images)")
    print(f"{'='*65}")

    test_loader = DataLoader(idrid_b_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_names = ['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative']
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Combined Model — IDRiD Test Confusion Matrix', fontsize=13)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(BASE, "confusion_matrix_idrid.png")
    plt.savefig(cm_path, dpi=150); plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # ── Final Summary ──────────────────────────────────────────────
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE — ALL IDRiD PARTS USED!")
    print(f"  Data used:")
    print(f"    MESSIDOR:             {len(messidor_train)} train images")
    print(f"    IDRiD Part B (grades): {len(idrid_b_train)} train images")
    print(f"    IDRiD Part A (masks):  {len(idrid_a_train)} train images (grades inferred)")
    print(f"    IDRiD Part C (locs):   same as Part B (included)")
    print(f"  IDRiD Test Accuracy:    {test_acc:.2f}%")
    print(f"  Model: {OUTPUT_MODEL}")
    print(f"{'='*65}")
    print(f"\nRun the app:  streamlit run app.py")
