"""
prepare_idrid.py — IDRiD Dataset Verification & Statistics
===========================================================
Run this BEFORE training to make sure all files are in place.

Usage:
    python prepare_idrid.py
"""

import os
import csv
from collections import Counter

# ==========================================
# IDRiD Paths (double-nested folder structure)
# ==========================================
BASE = r"C:\DR_CP"

PARTS = {
    "B. Disease Grading": {
        "base": os.path.join(BASE, "B. Disease Grading", "B. Disease Grading"),
        "train_images": os.path.join(BASE, "B. Disease Grading", "B. Disease Grading", "1. Original Images", "a. Training Set"),
        "test_images":  os.path.join(BASE, "B. Disease Grading", "B. Disease Grading", "1. Original Images", "b. Testing Set"),
        "train_csv":    os.path.join(BASE, "B. Disease Grading", "B. Disease Grading", "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv"),
        "test_csv":     os.path.join(BASE, "B. Disease Grading", "B. Disease Grading", "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    },
    "A. Segmentation": {
        "base": os.path.join(BASE, "A. Segmentation", "A. Segmentation"),
        "train_images": os.path.join(BASE, "A. Segmentation", "A. Segmentation", "1. Original Images", "a. Training Set"),
        "test_images":  os.path.join(BASE, "A. Segmentation", "A. Segmentation", "1. Original Images", "b. Testing Set"),
        "masks": os.path.join(BASE, "A. Segmentation", "A. Segmentation", "2. All Segmentation Groundtruths"),
    },
    "C. Localization": {
        "base": os.path.join(BASE, "C. Localization", "C. Localization"),
        "train_images": os.path.join(BASE, "C. Localization", "C. Localization", "1. Original Images", "a. Training Set") if os.path.exists(os.path.join(BASE, "C. Localization", "C. Localization", "1. Original Images")) else None,
        "od_csv_dir":   os.path.join(BASE, "C. Localization", "C. Localization", "2. Groundtruths", "1. Optic Disc Center Location"),
        "fovea_csv_dir": os.path.join(BASE, "C. Localization", "C. Localization", "2. Groundtruths", "2. Fovea Center Location"),
    }
}


def count_files(directory, extension='.jpg'):
    """Count files with given extension in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(extension)])


def analyze_csv(csv_path, label_col='Retinopathy grade'):
    """Analyze a CSV file and return class distribution."""
    if not os.path.exists(csv_path):
        return None, 0
    
    labels = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append(int(row[label_col].strip()))
            except (KeyError, ValueError):
                pass
    
    return Counter(labels), len(labels)


def main():
    print("=" * 65)
    print("  IDRiD Dataset Verification")
    print("=" * 65)
    
    all_ok = True
    
    # ── Part B: Disease Grading (MOST IMPORTANT) ─────────────────
    print("\n" + "─" * 65)
    print("  PART B: Disease Grading  [Used for fine-tuning your DR model]")
    print("─" * 65)
    
    b = PARTS["B. Disease Grading"]
    
    # Check base exists
    if os.path.exists(b["base"]):
        print("  ✓ Base directory found")
    else:
        print("  ✗ Base directory NOT FOUND:", b["base"])
        all_ok = False
    
    # Count images
    train_count = count_files(b["train_images"])
    test_count = count_files(b["test_images"])
    print(f"  Training images: {train_count}")
    print(f"  Testing images:  {test_count}")
    
    if train_count == 0:
        print("  ✗ No training images found!")
        all_ok = False
    
    # Analyze training CSV
    if os.path.exists(b["train_csv"]):
        print(f"  ✓ Training CSV found")
        dist, total = analyze_csv(b["train_csv"])
        print(f"    Total labels: {total}")
        dr_labels = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
        for grade in sorted(dist.keys()):
            pct = dist[grade] / total * 100
            bar = "█" * int(pct / 2)
            print(f"    Grade {grade} ({dr_labels.get(grade, '?'):>14s}): {dist[grade]:>4d} ({pct:5.1f}%) {bar}")
    else:
        print(f"  ✗ Training CSV NOT FOUND: {b['train_csv']}")
        all_ok = False
    
    # Analyze testing CSV
    if os.path.exists(b["test_csv"]):
        print(f"  ✓ Testing CSV found")
        dist, total = analyze_csv(b["test_csv"])
        print(f"    Total labels: {total}")
        for grade in sorted(dist.keys()):
            pct = dist[grade] / total * 100
            print(f"    Grade {grade} ({dr_labels.get(grade, '?'):>14s}): {dist[grade]:>4d} ({pct:5.1f}%)")
    else:
        print(f"  ✗ Testing CSV NOT FOUND")
        all_ok = False
    
    # ── Part A: Segmentation ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PART A: Segmentation  [Good for Grad-CAM validation]")
    print("─" * 65)
    
    a = PARTS["A. Segmentation"]
    if os.path.exists(a["base"]):
        print("  ✓ Base directory found")
        train_count = count_files(a["train_images"])
        test_count = count_files(a["test_images"])
        print(f"  Training images: {train_count}")
        print(f"  Testing images:  {test_count}")
        
        # Check mask directories
        masks_base = a["masks"]
        for split in ["a. Training Set", "b. Testing Set"]:
            split_path = os.path.join(masks_base, split)
            if os.path.exists(split_path):
                lesions = os.listdir(split_path)
                print(f"  Masks ({split}):")
                for lesion in sorted(lesions):
                    lesion_path = os.path.join(split_path, lesion)
                    if os.path.isdir(lesion_path):
                        tif_count = count_files(lesion_path, '.tif')
                        print(f"    {lesion}: {tif_count} masks")
    else:
        print("  ✗ Segmentation directory not found")
    
    # ── Part C: Localization ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("  PART C: Localization  [Optic Disc & Fovea coordinates]")
    print("─" * 65)
    
    c = PARTS["C. Localization"]
    if os.path.exists(c["base"]):
        print("  ✓ Base directory found")
        
        for name, path in [("Optic Disc CSVs", c["od_csv_dir"]), ("Fovea CSVs", c["fovea_csv_dir"])]:
            if os.path.exists(path):
                csvs = [f for f in os.listdir(path) if f.endswith('.csv')]
                print(f"  {name}: {len(csvs)} files")
    else:
        print("  ✗ Localization directory not found")
    
    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_ok:
        print("  ✓ ALL CHECKS PASSED — Dataset is ready for training!")
        print(f"\n  Next step: python train_idrid.py")
    else:
        print("  ✗ SOME CHECKS FAILED — Please fix the issues above.")
    print("=" * 65)


if __name__ == '__main__':
    main()
