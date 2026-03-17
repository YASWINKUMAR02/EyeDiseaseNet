"""
train_multitask.py — Multi-Task Training: DR + DME + Localization + Segmentation
==================================================================================
Trains a comprehensive retinal analysis system using ALL IDRiD parts:

Tasks trained:
  1. DR Grading (Part B):     5-class classification (Grade 0-4)
  2. DME Grading (Part B):    3-class classification (Risk 0-2)
  3. Localization (Part C):   Optic disc & fovea center regression (4 values: OD_x, OD_y, F_x, F_y)
  4. Segmentation (Part A):   U-Net for pixel-level lesion masks (4 channels: MA, HE, EX, SE)

Architecture:
  - EfficientNet-B0 backbone (shared features)
  - Task-specific heads for DR, DME, and localization
  - Separate lightweight U-Net for segmentation (trained independently)

Usage:
    python train_multitask.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
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

# Part B paths
B_BASE      = os.path.join(BASE, "B. Disease Grading", "B. Disease Grading")
B_TRAIN_CSV = os.path.join(B_BASE, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
B_TEST_CSV  = os.path.join(B_BASE, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
B_TRAIN_IMG = os.path.join(B_BASE, "1. Original Images", "a. Training Set")
B_TEST_IMG  = os.path.join(B_BASE, "1. Original Images", "b. Testing Set")

# Part C paths
C_BASE     = os.path.join(BASE, "C. Localization", "C. Localization")
OD_TRAIN   = os.path.join(C_BASE, "2. Groundtruths", "1. Optic Disc Center Location", "a. IDRiD_OD_Center_Training Set_Markups.csv")
OD_TEST    = os.path.join(C_BASE, "2. Groundtruths", "1. Optic Disc Center Location", "b. IDRiD_OD_Center_Testing Set_Markups.csv")
FOV_TRAIN  = os.path.join(C_BASE, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Training Set_Markups.csv")
FOV_TEST   = os.path.join(C_BASE, "2. Groundtruths", "2. Fovea Center Location", "IDRiD_Fovea_Center_Testing Set_Markups.csv")

# Part A paths
A_BASE        = os.path.join(BASE, "A. Segmentation", "A. Segmentation")
A_TRAIN_IMG   = os.path.join(A_BASE, "1. Original Images", "a. Training Set")
A_TEST_IMG    = os.path.join(A_BASE, "1. Original Images", "b. Testing Set")
A_MASKS_TRAIN = os.path.join(A_BASE, "2. All Segmentation Groundtruths", "a. Training Set")
A_MASKS_TEST  = os.path.join(A_BASE, "2. All Segmentation Groundtruths", "b. Testing Set")

# Model paths
PRETRAINED    = os.path.join(BASE, "efficientnet_retinal.pth")
OUTPUT_MULTI  = os.path.join(BASE, "efficientnet_multitask.pth")
OUTPUT_UNET   = os.path.join(BASE, "unet_segmentation.pth")

DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS_MULTI = 20
EPOCHS_SEG   = 30
LR = 0.0001

# Image dimensions (original IDRiD images are ~4288x2848, we normalize coords to 0-1)
ORIG_W, ORIG_H = 4288, 2848


# ==========================================
# Dataset: Multi-task (DR + DME + Localization)
# ==========================================
class MultiTaskDataset(Dataset):
    """Combined dataset: image → (DR_grade, DME_risk, OD_x, OD_y, Fovea_x, Fovea_y)"""
    
    def __init__(self, grade_csv, od_csv, fovea_csv, images_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load DR + DME grades
        grades = {}
        with open(grade_csv, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                name = row['Image name'].strip()
                grades[name] = {
                    'dr': int(row['Retinopathy grade'].strip()),
                    'dme': int(row['Risk of macular edema '].strip()),
                }
        
        # Load OD coordinates
        od_coords = {}
        with open(od_csv, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                name = row['Image No'].strip()
                if not name:
                    continue
                try:
                    od_coords[name] = (float(row['X- Coordinate'].strip()), float(row['Y - Coordinate'].strip()))
                except (ValueError, KeyError):
                    pass
        
        # Load Fovea coordinates
        fov_coords = {}
        with open(fovea_csv, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                name = row['Image No'].strip()
                if not name:
                    continue
                try:
                    fov_coords[name] = (float(row['X- Coordinate'].strip()), float(row['Y - Coordinate'].strip()))
                except (ValueError, KeyError):
                    pass
        
        # Merge — only keep images that have ALL labels
        for name in grades:
            img_path = os.path.join(images_dir, f"{name}.jpg")
            if os.path.exists(img_path) and name in od_coords and name in fov_coords:
                od_x, od_y = od_coords[name]
                f_x, f_y = fov_coords[name]
                self.samples.append({
                    'path': img_path,
                    'dr': grades[name]['dr'],
                    'dme': grades[name]['dme'],
                    'od_x': od_x / ORIG_W,   # Normalize to 0-1
                    'od_y': od_y / ORIG_H,
                    'fov_x': f_x / ORIG_W,
                    'fov_y': f_y / ORIG_H,
                })
        
        print(f"  [MultiTask] Loaded {len(self.samples)} images with DR+DME+Localization labels")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        dr_label = s['dr']
        dme_label = s['dme']
        coords = torch.tensor([s['od_x'], s['od_y'], s['fov_x'], s['fov_y']], dtype=torch.float32)
        
        return image, dr_label, dme_label, coords


# ==========================================
# Dataset: Segmentation (Part A)
# ==========================================
class SegmentationDataset(Dataset):
    """Part A: image → 4-channel lesion mask (MA, HE, EX, SE)"""
    
    LESIONS = [
        ('1. Microaneurysms', '_MA'),
        ('2. Haemorrhages', '_HE'),
        ('3. Hard Exudates', '_EX'),
        ('4. Soft Exudates', '_SE'),
    ]
    
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        
        self.image_names = sorted([
            os.path.splitext(f)[0] for f in os.listdir(images_dir)
            if f.lower().endswith('.jpg')
        ])
        print(f"  [Segmentation] Loaded {len(self.image_names)} images")
        
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        
        # Load image
        img = Image.open(os.path.join(self.images_dir, f"{name}.jpg")).convert('RGB')
        img = self.img_transform(img)
        
        # Load 4 lesion masks → stack into 4-channel tensor
        masks = []
        for dir_name, suffix in self.LESIONS:
            mask_path = os.path.join(self.masks_dir, dir_name, f"{name}{suffix}.tif")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()  # Binary
            else:
                mask = torch.zeros(1, self.img_size, self.img_size)
            masks.append(mask)
        
        masks = torch.cat(masks, dim=0)  # Shape: (4, H, W)
        return img, masks


# ==========================================
# Model: Multi-Task EfficientNet
# ==========================================
class MultiTaskEfficientNet(nn.Module):
    """
    EfficientNet-B0 with 3 task heads:
      1. DR Classification (5 classes)
      2. DME Classification (3 classes)
      3. Localization Regression (4 coordinates: OD_x, OD_y, Fovea_x, Fovea_y)
    """
    
    def __init__(self, num_dr_classes=5, num_dme_classes=3, num_coords=4):
        super().__init__()
        
        # Shared backbone
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        in_features = 1280  # EfficientNet-B0 output features
        
        # Task heads
        self.dr_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_dr_classes)
        )
        
        self.dme_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_dme_classes)
        )
        
        self.loc_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_coords),
            nn.Sigmoid()  # Coords normalized to 0-1
        )
    
    def forward(self, x):
        features = self.features(x)
        features = self.avgpool(features)
        features = features.flatten(1)
        
        dr_out  = self.dr_head(features)
        dme_out = self.dme_head(features)
        loc_out = self.loc_head(features)
        
        return dr_out, dme_out, loc_out


# ==========================================
# Model: Lightweight U-Net for Segmentation
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class LightUNet(nn.Module):
    """Lightweight U-Net for 4-class lesion segmentation."""
    
    def __init__(self, in_channels=3, out_channels=4):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        self.final = nn.Conv2d(32, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.final(d1)


# ==========================================
# Training: Multi-Task
# ==========================================
def train_multitask(model, train_loader, val_loader, train_size, val_size, device, epochs=20):
    print(f"\n{'='*65}")
    print(f"  MULTI-TASK TRAINING: DR + DME + Localization ({epochs} epochs)")
    print(f"{'='*65}")
    
    dr_criterion  = nn.CrossEntropyLoss()
    dme_criterion = nn.CrossEntropyLoss()
    loc_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for phase, loader, size in [('train', train_loader, train_size), ('val', val_loader, val_size)]:
            model.train() if phase == 'train' else model.eval()
            
            r_loss = 0.0
            dr_correct = 0
            dme_correct = 0
            loc_error = 0.0
            
            for imgs, dr_labels, dme_labels, coords in tqdm(loader, desc=f"  {phase}"):
                imgs = imgs.to(device)
                dr_labels = dr_labels.to(device)
                dme_labels = dme_labels.to(device)
                coords = coords.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast(enabled=(phase == 'train')):
                        dr_out, dme_out, loc_out = model(imgs)
                        
                        loss_dr  = dr_criterion(dr_out, dr_labels)
                        loss_dme = dme_criterion(dme_out, dme_labels)
                        loss_loc = loc_criterion(loc_out, coords)
                        
                        # Combined loss (weighted)
                        loss = loss_dr + 0.5 * loss_dme + 2.0 * loss_loc
                    
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                r_loss += loss.item() * imgs.size(0)
                dr_correct += (dr_out.argmax(1) == dr_labels).sum().item()
                dme_correct += (dme_out.argmax(1) == dme_labels).sum().item()
                loc_error += loss_loc.item() * imgs.size(0)
            
            if phase == 'train':
                scheduler.step()
            
            avg_loss = r_loss / size
            dr_acc = dr_correct / size * 100
            dme_acc = dme_correct / size * 100
            avg_loc = loc_error / size
            
            print(f"  {phase}: Loss={avg_loss:.4f}  DR_Acc={dr_acc:.1f}%  DME_Acc={dme_acc:.1f}%  Loc_MSE={avg_loc:.6f}")
            
            if phase == 'val' and dr_acc > best_acc:
                best_acc = dr_acc
                best_wts = copy.deepcopy(model.state_dict())
                print(f"  --> New best! DR Val Acc: {best_acc:.1f}%")
    
    model.load_state_dict(best_wts)
    return model


# ==========================================
# Training: Segmentation (U-Net)
# ==========================================
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train_segmentation(model, train_loader, device, epochs=30):
    print(f"\n{'='*65}")
    print(f"  SEGMENTATION TRAINING: U-Net ({epochs} epochs)")
    print(f"{'='*65}")
    
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for imgs, masks in tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            loss = bce_criterion(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_wts = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f} (best={best_loss:.4f})")
    
    model.load_state_dict(best_wts)
    return model


# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print("=" * 65)
    print("  MULTI-TASK RETINAL ANALYSIS — FULL IDRiD TRAINING")
    print("  Tasks: DR Grading + DME Grading + Localization + Segmentation")
    print("=" * 65)
    
    # ── Transforms ─────────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # ═══════════════════════════════════════════════════════════════
    # TASK 1-3: Multi-Task (DR + DME + Localization)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Loading Multi-Task Dataset (Parts B + C) ---")
    train_mt = MultiTaskDataset(B_TRAIN_CSV, OD_TRAIN, FOV_TRAIN, B_TRAIN_IMG, train_tf)
    test_mt  = MultiTaskDataset(B_TEST_CSV,  OD_TEST,  FOV_TEST,  B_TEST_IMG,  val_tf)
    
    # 85/15 train/val split
    val_size = int(len(train_mt) * 0.15)
    train_size = len(train_mt) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        train_mt, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Build multi-task model
    mt_model = MultiTaskEfficientNet().to(DEVICE)
    
    # Load pre-trained backbone weights
    if os.path.exists(PRETRAINED):
        pretrained_dict = torch.load(PRETRAINED, map_location=DEVICE)
        model_dict = mt_model.state_dict()
        # Only load matching feature weights
        pretrained_features = {k: v for k, v in pretrained_dict.items()
                               if k.startswith('features.') and k in model_dict
                               and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_features)
        mt_model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_features)} pre-trained backbone layers")
    
    # Train multi-task model
    mt_model = train_multitask(mt_model, train_loader, val_loader, train_size, val_size, DEVICE, EPOCHS_MULTI)
    
    # Save
    torch.save(mt_model.state_dict(), OUTPUT_MULTI)
    print(f"\nMulti-task model saved: {OUTPUT_MULTI}")
    
    # ── Evaluate multi-task on test set ────────────────────────────
    print("\n--- Multi-Task Test Evaluation ---")
    test_loader = DataLoader(test_mt, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    mt_model.eval()
    
    all_dr_preds, all_dr_labels = [], []
    all_dme_preds, all_dme_labels = [], []
    total_loc_error = 0.0
    n_test = 0
    
    with torch.no_grad():
        for imgs, dr_labels, dme_labels, coords in test_loader:
            imgs = imgs.to(DEVICE)
            dr_out, dme_out, loc_out = mt_model(imgs)
            
            all_dr_preds.extend(dr_out.argmax(1).cpu().numpy())
            all_dr_labels.extend(dr_labels.numpy())
            all_dme_preds.extend(dme_out.argmax(1).cpu().numpy())
            all_dme_labels.extend(dme_labels.numpy())
            
            loc_err = ((loc_out.cpu() - coords) ** 2).mean().item()
            total_loc_error += loc_err * imgs.size(0)
            n_test += imgs.size(0)
    
    dr_names = ['0-No DR', '1-Mild', '2-Moderate', '3-Severe', '4-Proliferative']
    dme_names = ['0-No Risk', '1-Moderate', '2-High Risk']
    
    print("\n--- DR Classification Report ---")
    print(classification_report(all_dr_labels, all_dr_preds, target_names=dr_names, zero_division=0))
    
    print("\n--- DME Classification Report ---")
    print(classification_report(all_dme_labels, all_dme_preds, target_names=dme_names, zero_division=0))
    
    dr_acc = np.mean(np.array(all_dr_preds) == np.array(all_dr_labels)) * 100
    dme_acc = np.mean(np.array(all_dme_preds) == np.array(all_dme_labels)) * 100
    avg_loc = total_loc_error / n_test
    print(f"  DR Test Accuracy:  {dr_acc:.1f}%")
    print(f"  DME Test Accuracy: {dme_acc:.1f}%")
    print(f"  Localization MSE:  {avg_loc:.6f}")
    
    # ═══════════════════════════════════════════════════════════════
    # TASK 4: Segmentation (U-Net)
    # ═══════════════════════════════════════════════════════════════
    print("\n--- Loading Segmentation Dataset (Part A) ---")
    seg_train = SegmentationDataset(A_TRAIN_IMG, A_MASKS_TRAIN, img_size=256)
    seg_test  = SegmentationDataset(A_TEST_IMG, A_MASKS_TEST, img_size=256)
    
    seg_train_loader = DataLoader(seg_train, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    
    unet = LightUNet(in_channels=3, out_channels=4).to(DEVICE)
    unet = train_segmentation(unet, seg_train_loader, DEVICE, EPOCHS_SEG)
    
    # Save
    torch.save(unet.state_dict(), OUTPUT_UNET)
    print(f"\nU-Net segmentation model saved: {OUTPUT_UNET}")
    
    # ── Evaluate segmentation ─────────────────────────────────────
    print("\n--- Segmentation Test Evaluation ---")
    seg_test_loader = DataLoader(seg_test, batch_size=4, shuffle=False)
    unet.eval()
    
    lesion_names = ['Microaneurysms', 'Haemorrhages', 'Hard Exudates', 'Soft Exudates']
    dice_scores = [[] for _ in range(4)]
    
    with torch.no_grad():
        for imgs, masks in seg_test_loader:
            imgs = imgs.to(DEVICE)
            preds = torch.sigmoid(unet(imgs)).cpu()
            preds_bin = (preds > 0.5).float()
            
            for c in range(4):
                for b in range(preds_bin.size(0)):
                    intersection = (preds_bin[b, c] * masks[b, c]).sum()
                    union = preds_bin[b, c].sum() + masks[b, c].sum()
                    if union > 0:
                        dice = (2 * intersection / union).item()
                        dice_scores[c].append(dice)
    
    print("\n  Segmentation Dice Scores:")
    for i, name in enumerate(lesion_names):
        if dice_scores[i]:
            avg = np.mean(dice_scores[i])
            print(f"    {name}: {avg:.4f}")
        else:
            print(f"    {name}: N/A (no positive samples)")
    
    # ═══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  ALL TRAINING COMPLETE!")
    print(f"  Multi-task model: {OUTPUT_MULTI}")
    print(f"    - DR Grading (5 classes):  {dr_acc:.1f}% test accuracy")
    print(f"    - DME Grading (3 classes): {dme_acc:.1f}% test accuracy")
    print(f"    - Localization (OD+Fovea): MSE={avg_loc:.6f}")
    print(f"  Segmentation model: {OUTPUT_UNET}")
    print(f"    - 4 lesion types: MA, HE, EX, SE")
    print(f"{'='*65}")
    print(f"\n  Run the app: streamlit run app.py")
