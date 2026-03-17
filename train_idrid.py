"""
train_idrid.py — Fine-tune EfficientNet-B0 on IDRiD Disease Grading Dataset
=============================================================================
Loads the pre-trained MESSIDOR model and fine-tunes it on the IDRiD dataset
for better DR severity classification. The IDRiD dataset has expert-graded
fundus images with 5 DR severity levels (0-4).

Usage:
    python train_idrid.py

Features:
    - Loads pre-trained weights from efficientnet_retinal.pth
    - Fine-tunes with lower learning rate to preserve learned features
    - Uses weighted loss to handle class imbalance in IDRiD
    - Generates confusion matrix and classification report
    - Saves improved model as efficientnet_retinal_idrid.pth
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
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import copy

# Import our custom IDRiD dataset
from idrid_dataset import IDRiDDiseaseGrading

# ==========================================
# Configuration
# ==========================================
# IDRiD dataset paths (double-nested folder structure)
IDRID_BASE = r"C:\DR_CP\B. Disease Grading\B. Disease Grading"
IDRID_TRAIN_CSV = os.path.join(IDRID_BASE, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
IDRID_TEST_CSV  = os.path.join(IDRID_BASE, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
IDRID_TRAIN_IMG = os.path.join(IDRID_BASE, "1. Original Images", "a. Training Set")
IDRID_TEST_IMG  = os.path.join(IDRID_BASE, "1. Original Images", "b. Testing Set")

# Model paths
PRETRAINED_MODEL = r"C:\DR_CP\efficientnet_retinal.pth"      # Pre-trained on MESSIDOR
FINETUNED_MODEL  = r"C:\DR_CP\efficientnet_retinal_idrid.pth" # Output: fine-tuned model

NUM_CLASSES    = 5
BATCH_SIZE     = 16        # Smaller batch since IDRiD has fewer images
NUM_EPOCHS     = 20        # More epochs since dataset is small
LEARNING_RATE  = 0.0001    # Lower LR for fine-tuning (10x smaller than initial training)
VAL_SPLIT      = 0.2       # 20% of IDRiD training data for validation


# ==========================================
# Training Function (with AMP)
# ==========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=20):
    since = time.time()
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 40)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

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
                print(f"  --> New best model! Val Acc: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history


# ==========================================
# Plot Training History
# ==========================================
def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', color='#4f9cf9', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='#f87171', linewidth=2)
    ax1.set_title('Loss per Epoch', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', color='#4f9cf9', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', color='#34d399', linewidth=2)
    ax2.set_title('Accuracy per Epoch', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")
    plt.close()


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("  IDRiD Fine-tuning — EfficientNet-B0")
    print("=" * 60)

    # ── Data Transforms ────────────────────────────────────────────
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

    # ── Load IDRiD Dataset ─────────────────────────────────────────
    print("\n--- Loading IDRiD Disease Grading Dataset ---")

    # Full training set (will be split into train/val)
    full_train = IDRiDDiseaseGrading(
        csv_path=IDRID_TRAIN_CSV,
        images_dir=IDRID_TRAIN_IMG,
        transform=train_transforms,
        task='dr'
    )

    # Test set
    test_dataset = IDRiDDiseaseGrading(
        csv_path=IDRID_TEST_CSV,
        images_dir=IDRID_TEST_IMG,
        transform=val_transforms,
        task='dr'
    )

    # ── Split training into train/val ──────────────────────────────
    val_size = int(len(full_train) * VAL_SPLIT)
    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Override val transforms (random_split shares parent transforms)
    # We create a separate val dataset with val transforms
    val_dataset_clean = IDRiDDiseaseGrading(
        csv_path=IDRID_TRAIN_CSV,
        images_dir=IDRID_TRAIN_IMG,
        transform=val_transforms,
        task='dr'
    )
    # Use the same indices as the val_dataset split
    val_indices = val_dataset.indices
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset_clean, val_indices)

    print(f"\nDataset split:")
    print(f"  Train: {train_size} images")
    print(f"  Val:   {val_size} images")
    print(f"  Test:  {len(test_dataset)} images")

    # ── DataLoaders ────────────────────────────────────────────────
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, persistent_workers=True),
        'val':   DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True),
    }
    dataset_sizes = {'train': train_size, 'val': val_size}

    # ── Build Model ────────────────────────────────────────────────
    print("\n--- Loading Pre-trained MESSIDOR Model ---")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    if os.path.exists(PRETRAINED_MODEL):
        model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
        print(f"  Loaded pre-trained weights from: {PRETRAINED_MODEL}")
        print("  Fine-tuning will START from MESSIDOR-trained weights!")
    else:
        print(f"  WARNING: Pre-trained model not found at {PRETRAINED_MODEL}")
        print("  Training from ImageNet weights instead...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

    model = model.to(device)

    # ── Weighted Loss for Class Imbalance ──────────────────────────
    # IDRiD is imbalanced (many Grade 0 and Grade 2, fewer Grade 1)
    train_dist = full_train.get_class_distribution()
    total_samples = sum(train_dist.values())
    class_weights = []
    for i in range(NUM_CLASSES):
        count = train_dist.get(i, 1)
        weight = total_samples / (NUM_CLASSES * count)
        class_weights.append(weight)
        print(f"  Class {i} weight: {weight:.3f} (count: {count})")

    weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # ── Optimizer & Scheduler ──────────────────────────────────────
    # Use different learning rates: lower for backbone, higher for classifier
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},    # 1e-5 for backbone
        {'params': classifier_params, 'lr': LEARNING_RATE},        # 1e-4 for classifier
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting Fine-tuning: {NUM_EPOCHS} epochs, LR={LEARNING_RATE}")
    print(f"{'='*60}")

    model, history = train_model(
        model, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, device, num_epochs=NUM_EPOCHS
    )

    # ── Save Model ─────────────────────────────────────────────────
    torch.save(model.state_dict(), FINETUNED_MODEL)
    print(f"\nFine-tuned model saved to: {FINETUNED_MODEL}")

    # ── Plot Training History ──────────────────────────────────────
    history_path = r"C:\DR_CP\idrid_training_history.png"
    plot_history(history, history_path)

    # ── Evaluate on IDRiD Test Set ─────────────────────────────────
    print("\n--- Evaluating on IDRiD Test Set (103 images) ---")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    label_names = ['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative']
    print("\n--- Classification Report (IDRiD Test Set) ---")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('IDRiD Test Set — Confusion Matrix (Fine-tuned Model)', fontsize=13)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = r"C:\DR_CP\confusion_matrix_idrid.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")
    plt.close()

    # ── Summary ────────────────────────────────────────────────────
    test_acc = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels) * 100
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPLETE!")
    print(f"  IDRiD Test Accuracy: {test_acc:.2f}%")
    print(f"  Model saved: {FINETUNED_MODEL}")
    print(f"  Confusion Matrix: {cm_path}")
    print(f"  Training History: {history_path}")
    print(f"{'='*60}")
    print(f"\nTo use this model in the app:")
    print(f"  1. Update MODEL_PATH in app.py to: {FINETUNED_MODEL}")
    print(f"  2. Or run: streamlit run app.py")
