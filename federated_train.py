"""
federated_train.py — Federated Learning Simulation for Retinal Disease Diagnosis
==================================================================================
Simulates N=5 hospital clients training EfficientNet-B0 locally, with a central
server aggregating updates using Federated Averaging (FedAvg) over R=10 rounds.

Architecture:
  - Server  : holds global model, runs FedAvg, evaluates on val set each round
  - Clients : each gets 1/N of training data, trains for E local epochs per round
  - No real networking — everything is in-memory (valid for academic simulation)

Usage:
  python federated_train.py
"""

import os, copy, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR          = r"C:\DR_CP\DE_MESSIDOR_EX\augmented_resized_V2"
MODEL_SAVE_PATH   = r"C:\DR_CP\efficientnet_retinal_federated.pth"
CM_SAVE_PATH      = r"C:\DR_CP\confusion_matrix_federated.png"
ACC_PLOT_PATH     = r"C:\DR_CP\federated_accuracy_curve.png"

NUM_CLASSES       = 5
BATCH_SIZE        = 32       # smaller batch per client
NUM_CLIENTS       = 5        # simulated hospitals
NUM_ROUNDS        = 10       # global communication rounds
LOCAL_EPOCHS      = 2        # each client trains for 2 epochs per round
LEARNING_RATE     = 0.001


# ══════════════════════════════════════════════════════════════════════════════
# DATA TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_model(device):
    """Load pretrained EfficientNet-B0 and replace the classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════════════
# PARTITION DATA INTO N CLIENT SHARDS
# ══════════════════════════════════════════════════════════════════════════════
def partition_data(full_dataset, num_clients):
    """
    Split training dataset indices evenly across clients.
    Returns a list of Subset objects, one per client.
    IID split: shuffle indices first so each client sees all classes.
    """
    indices = np.random.permutation(len(full_dataset))
    shard_size = len(full_dataset) // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * shard_size
        # Last client gets any remaining samples
        end   = (i + 1) * shard_size if i < num_clients - 1 else len(full_dataset)
        client_datasets.append(Subset(full_dataset, indices[start:end]))
    return client_datasets


# ══════════════════════════════════════════════════════════════════════════════
# CLIENT UPDATE — Local Training
# ══════════════════════════════════════════════════════════════════════════════
def client_update(global_weights, client_dataset, device, client_id):
    """
    Simulate one client's local training:
      1. Load global weights into a local model copy
      2. Train for LOCAL_EPOCHS epochs on client's own data
      3. Return updated weight dict + dataset size
    """
    # Build local model and load global weights
    local_model = build_model(device)
    local_model.load_state_dict(copy.deepcopy(global_weights))
    local_model.train()

    loader = DataLoader(
        client_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

    total_loss = 0.0
    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss
        print(f"    Client {client_id+1} | Epoch {epoch+1}/{LOCAL_EPOCHS} | "
              f"Loss: {epoch_loss/len(loader):.4f}")

    return local_model.state_dict(), len(client_dataset)


# ══════════════════════════════════════════════════════════════════════════════
# FEDAVG — Server Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def fedavg(client_weights_list, client_sizes):
    """
    Federated Averaging:
    Compute weighted average of client model weights.
    Each client contributes proportionally to its dataset size.
    
    Args:
        client_weights_list : list of state_dict from each client
        client_sizes        : list of dataset sizes per client
    Returns:
        averaged_weights    : new global state_dict
    """
    total_samples = sum(client_sizes)
    averaged_weights = copy.deepcopy(client_weights_list[0])

    for key in averaged_weights:
        # Start from zeros
        averaged_weights[key] = torch.zeros_like(averaged_weights[key], dtype=torch.float32)
        # Weighted sum
        for i, state in enumerate(client_weights_list):
            weight = client_sizes[i] / total_samples
            averaged_weights[key] += state[key].float() * weight

    return averaged_weights


# ══════════════════════════════════════════════════════════════════════════════
# SERVER EVALUATION — Val Accuracy
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, dataloader, device, desc="Validation"):
    """Evaluate model on a dataloader. Returns accuracy."""
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Federated Training Loop
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Federated Learning — Retinal Disease Diagnosis")
    print(f"  Clients: {NUM_CLIENTS} | Rounds: {NUM_ROUNDS} | Local Epochs: {LOCAL_EPOCHS}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── 1. Load datasets ────────────────────────────────────────────────────
    print("Loading datasets...")
    full_train = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),  val_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), val_transforms)

    val_loader  = DataLoader(val_dataset,  batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    class_names = full_train.classes
    print(f"  Total training samples : {len(full_train)}")
    print(f"  Val samples            : {len(val_dataset)}")
    print(f"  Test samples           : {len(test_dataset)}")
    print(f"  Classes                : {class_names}\n")

    # ── 2. Partition training data across clients ───────────────────────────
    print(f"Partitioning data into {NUM_CLIENTS} client shards...")
    client_datasets = partition_data(full_train, NUM_CLIENTS)
    for i, cd in enumerate(client_datasets):
        print(f"  Hospital {i+1} — {len(cd)} samples")
    print()

    # ── 3. Initialize global model on server ───────────────────────────────
    print("Initializing global model (EfficientNet-B0)...")
    global_model = build_model(device)
    global_weights = global_model.state_dict()

    # ── 4. Federated Training Rounds ────────────────────────────────────────
    round_accuracies = []
    best_acc = 0.0
    best_weights = copy.deepcopy(global_weights)

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'─'*60}")
        print(f"  🌐 GLOBAL ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'─'*60}")
        start = time.time()

        # ── Each client trains locally ──
        client_weights_list = []
        client_sizes        = []

        for cid, client_data in enumerate(client_datasets):
            print(f"\n  ➤ Hospital {cid+1} training...")
            weights, size = client_update(global_weights, client_data, device, cid)
            client_weights_list.append(weights)
            client_sizes.append(size)

        # ── Server aggregates (FedAvg) ──
        print(f"\n  🔄 Server aggregating weights (FedAvg)...")
        global_weights = fedavg(client_weights_list, client_sizes)

        # ── Load averaged weights into global model ──
        global_model.load_state_dict(global_weights)

        # ── Evaluate global model on val set ──
        val_acc = evaluate(global_model, val_loader, device, desc="Val Eval")
        round_accuracies.append(val_acc)

        elapsed = time.time() - start
        print(f"\n  ✅ Round {round_num} complete | Val Acc: {val_acc*100:.2f}% | "
              f"Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")

        # ── Save best model ──
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(global_weights)
            print(f"  ⭐ New best model saved! (Val Acc: {best_acc*100:.2f}%)")

    # ── 5. Load best global model ───────────────────────────────────────────
    global_model.load_state_dict(best_weights)

    # ── 6. Save federated model ─────────────────────────────────────────────
    torch.save(global_model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Federated model saved → {MODEL_SAVE_PATH}")
    print(f"   Best Val Accuracy: {best_acc*100:.2f}%")

    # ── 7. Plot accuracy curve ──────────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, NUM_ROUNDS + 1), [a * 100 for a in round_accuracies],
             marker='o', linewidth=2.5, color='#4f9cf9', markerfacecolor='#a78bfa',
             markersize=8, markeredgewidth=2)
    plt.title('Federated Learning — Global Model Accuracy per Round', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, NUM_ROUNDS + 1))
    plt.tight_layout()
    plt.savefig(ACC_PLOT_PATH, dpi=150)
    print(f"   Accuracy curve saved → {ACC_PLOT_PATH}")

    # ── 8. Final Test Evaluation ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Final Test Set Evaluation")
    print("="*60)

    global_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Evaluation"):
            inputs = inputs.to(device)
            outputs = global_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    label_names = ['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative']
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # ── Confusion Matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix — Federated Model', fontsize=14)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=150)
    print(f"\nConfusion matrix saved → {CM_SAVE_PATH}")
    print("\n🎉 Federated Training Complete!")
