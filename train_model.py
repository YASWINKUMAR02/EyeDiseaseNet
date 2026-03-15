import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for scripts)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import copy

# ==========================================
# Configuration
# ==========================================
DATA_DIR = r"C:\DR_CP\DE_MESSIDOR_EX\augmented_resized_V2"
MODEL_SAVE_PATH = r"C:\DR_CP\efficientnet_retinal.pth"
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# ==========================================
# Training Function (with AMP for speed)
# ==========================================
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=10):
    since = time.time()
    scaler = GradScaler()  # Mixed precision scaler

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # Use autocast for mixed precision
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

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("--> Saved new best model weights!")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# MAIN GUARD - Required for Windows multiprocessing (num_workers > 0)
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms),
        'val':   datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   val_transforms),
        'test':  datasets.ImageFolder(os.path.join(DATA_DIR, 'test'),  val_transforms)
    }

    # Use num_workers=4 (safe on Windows), pin_memory for faster GPU transfer
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True),
        'val':   DataLoader(image_datasets['val'],   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True),
        'test':  DataLoader(image_datasets['test'],  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")

    # Model
    print("Initializing EfficientNet-B0 model...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train
    model = train_model(model, criterion, optimizer, scheduler,
                        dataloaders, dataset_sizes, device, num_epochs=NUM_EPOCHS)

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")

    # ==========================================
    # Confusion Matrix on Test Set
    # ==========================================
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test'], desc="Test evaluation"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    label_names = ['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative']
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Retinal Disease Diagnosis', fontsize=14)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_save_path = r"C:\DR_CP\confusion_matrix.png"
    plt.savefig(cm_save_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_save_path}")

