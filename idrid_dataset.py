"""
idrid_dataset.py — Custom PyTorch Dataset for IDRiD Disease Grading
====================================================================
Loads fundus images and DR severity labels from the IDRiD CSV groundtruth files.
Supports both the Training and Testing sets.

CSV columns: "Image name", "Retinopathy grade", "Risk of macular edema "
DR grades: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative)
DME risk:  0 (No risk), 1 (Moderate risk), 2 (High risk)
"""

import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class IDRiDDiseaseGrading(Dataset):
    """
    Custom Dataset for IDRiD Disease Grading (Part B).
    
    Args:
        csv_path (str): Path to the CSV file with labels.
        images_dir (str): Path to the directory containing JPG images.
        transform (callable, optional): Transform to apply to images.
        task (str): 'dr' for Retinopathy grade, 'dme' for DME risk.
    """
    
    def __init__(self, csv_path, images_dir, transform=None, task='dr'):
        self.images_dir = images_dir
        self.transform = transform
        self.task = task
        
        # Parse CSV
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row['Image name'].strip()
                dr_grade = int(row['Retinopathy grade'].strip())
                dme_risk = int(row['Risk of macular edema '].strip())
                
                # Check if image file exists
                img_path = os.path.join(images_dir, f"{img_name}.jpg")
                if os.path.exists(img_path):
                    self.samples.append({
                        'image_path': img_path,
                        'image_name': img_name,
                        'dr_grade': dr_grade,
                        'dme_risk': dme_risk,
                    })
        
        print(f"[IDRiD] Loaded {len(self.samples)} images from {images_dir}")
        
        # Print class distribution
        from collections import Counter
        if task == 'dr':
            dist = Counter(s['dr_grade'] for s in self.samples)
            labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
        else:
            dist = Counter(s['dme_risk'] for s in self.samples)
            labels = {0: 'No Risk', 1: 'Moderate Risk', 2: 'High Risk'}
        
        for k in sorted(dist.keys()):
            print(f"  Grade {k} ({labels.get(k, '?')}): {dist[k]} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = sample['dr_grade'] if self.task == 'dr' else sample['dme_risk']
        return image, label
    
    def get_class_distribution(self):
        """Returns a dict of {grade: count}."""
        from collections import Counter
        key = 'dr_grade' if self.task == 'dr' else 'dme_risk'
        return dict(Counter(s[key] for s in self.samples))


class IDRiDSegmentation(Dataset):
    """
    Custom Dataset for IDRiD Segmentation (Part A).
    Loads original images + binary masks for specified lesion types.
    
    Args:
        images_dir (str): Path to original images directory.
        masks_dirs (dict): Mapping of lesion name -> directory with TIF masks.
        transform (callable, optional): Transform to apply to images.
        mask_transform (callable, optional): Transform to apply to masks.
    """
    
    def __init__(self, images_dir, masks_dirs, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dirs = masks_dirs
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get list of available images
        self.image_names = sorted([
            os.path.splitext(f)[0] for f in os.listdir(images_dir)
            if f.lower().endswith('.jpg')
        ])
        print(f"[IDRiD Segmentation] Found {len(self.image_names)} images")
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{name}.jpg")
        image = Image.open(img_path).convert('RGB')
        
        # Load masks for each lesion type
        masks = {}
        for lesion_name, masks_dir in self.masks_dirs.items():
            mask_path = os.path.join(masks_dir, f"{name}_{lesion_name}.tif")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')  # Grayscale
                if self.mask_transform:
                    mask = self.mask_transform(mask)
                masks[lesion_name] = mask
        
        if self.transform:
            image = self.transform(image)
        
        return image, masks
