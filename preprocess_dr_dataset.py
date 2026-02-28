"""
Diabetic Retinopathy Image Preprocessing Pipeline
Preprocesses images and organizes them for training
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split


class DRPreprocessor:
    """Preprocess Diabetic Retinopathy fundus images"""
    
    def __init__(self, source_dir, output_dir, target_size=224):
        """
        Args:
            source_dir: Path to original dataset (Downloads/dataset/Dataset)
            output_dir: Path to save preprocessed data
            target_size: Target image size (224 for EfficientNet-B0, 300 for B1, etc.)
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Class names as per your dataset
        self.classes = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def preprocess_image(self, img_path, apply_clahe=True, crop_circle=True):
        """
        Preprocess a single fundus image
        
        Args:
            img_path: Path to input image
            apply_clahe: Apply CLAHE for contrast enhancement
            crop_circle: Crop to circular fundus region
        
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Step 1: Resize maintaining aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h = int(self.target_size * 1.2)
                new_w = int(w * (new_h / h))
            else:
                new_w = int(self.target_size * 1.2)
                new_h = int(h * (new_w / w))
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Step 2: Crop to circle (remove black borders in fundus images)
            if crop_circle:
                img = self.crop_fundus_circle(img)
            
            # Step 3: Apply CLAHE for contrast enhancement
            if apply_clahe:
                img = self.apply_clahe(img)
            
            # Step 4: Final resize to target size
            img = cv2.resize(img, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_LANCZOS4)
            
            # Step 5: Normalize
            img = self.normalize_image(img)
            
            return img
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None
    
    def crop_fundus_circle(self, img):
        """Crop fundus image to circular region, removing black borders"""
        # Convert to grayscale for circle detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Threshold to find non-black regions
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (should be the fundus circle)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small margin
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            # Crop
            img = img[y:y+h, x:x+w]
        
        return img
    
    def apply_clahe(self, img):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return img
    
    def normalize_image(self, img):
        """Normalize image pixel values"""
        # Simple min-max normalization
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        return img
    
    def process_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Process entire dataset and split into train/val/test
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        print("=" * 70)
        print("DIABETIC RETINOPATHY IMAGE PREPROCESSING")
        print("=" * 70)
        print(f"Source: {self.source_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Target size: {self.target_size}x{self.target_size}")
        print(f"Split: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
        print("=" * 70)
        
        # Create output directory structure
        self.create_output_structure()
        
        # Statistics
        stats = {
            'total_processed': 0,
            'total_failed': 0,
            'class_counts': {cls: 0 for cls in self.classes},
            'split_info': {}
        }
        
        # Process each class
        all_file_info = []
        
        for class_name in self.classes:
            print(f"\n📁 Processing class: {class_name}")
            class_path = self.source_dir / class_name
            
            if not class_path.exists():
                print(f"   ⚠️  Folder not found: {class_path}")
                continue
            
            # Get all image files
            image_files = list(class_path.glob('*.png')) + \
                         list(class_path.glob('*.jpg')) + \
                         list(class_path.glob('*.jpeg'))
            
            print(f"   Found {len(image_files)} images")
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"   Processing {class_name}"):
                processed_img = self.preprocess_image(img_path)
                
                if processed_img is not None:
                    # Store file info for splitting later
                    all_file_info.append({
                        'original_path': str(img_path),
                        'filename': img_path.name,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'processed_img': processed_img
                    })
                    stats['total_processed'] += 1
                    stats['class_counts'][class_name] += 1
                else:
                    stats['total_failed'] += 1
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Processed: {stats['total_processed']} images")
        print(f"   Failed: {stats['total_failed']} images")
        
        # Split dataset
        print(f"\n📊 Splitting dataset...")
        split_data = self.split_and_save(all_file_info, train_ratio, val_ratio, test_ratio)
        stats['split_info'] = split_data['split_counts']
        
        # Save statistics
        self.save_statistics(stats)
        
        print("\n" + "=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)
        for split_name, counts in split_data['split_counts'].items():
            print(f"\n{split_name.upper()} SET:")
            total = sum(counts.values())
            print(f"  Total: {total} images")
            for cls, count in counts.items():
                pct = (count / total * 100) if total > 0 else 0
                print(f"    {cls:20s}: {count:4d} ({pct:5.2f}%)")
        
        print("\n✨ All done! Your preprocessed dataset is ready.")
        print(f"📂 Location: {self.output_dir}")
        
        return stats
    
    def create_output_structure(self):
        """Create output directory structure"""
        # Main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for train/val/test
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Create class subdirectories
            for class_name in self.classes:
                class_dir = split_dir / class_name
                class_dir.mkdir(exist_ok=True)
    
    def split_and_save(self, all_file_info, train_ratio, val_ratio, test_ratio):
        """Split data and save to respective directories"""
        
        # Group by class for stratified splitting
        class_groups = {cls: [] for cls in self.classes}
        for info in all_file_info:
            class_groups[info['class_name']].append(info)
        
        split_counts = {'train': {}, 'val': {}, 'test': {}}
        split_metadata = {'train': [], 'val': [], 'test': []}
        
        # Split each class separately (stratified)
        for class_name, files in class_groups.items():
            if len(files) == 0:
                continue
            
            # First split: train vs (val+test)
            train_files, temp_files = train_test_split(
                files, 
                train_size=train_ratio, 
                random_state=42
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_files, test_files = train_test_split(
                temp_files,
                train_size=val_size,
                random_state=42
            )
            
            # Save files
            for split_name, file_list in [('train', train_files), 
                                          ('val', val_files), 
                                          ('test', test_files)]:
                split_counts[split_name][class_name] = len(file_list)
                
                for i, file_info in enumerate(file_list):
                    # Create new filename
                    new_filename = f"{class_name}_{i:04d}.png"
                    output_path = self.output_dir / split_name / class_name / new_filename
                    
                    # Save image
                    img = file_info['processed_img']
                    img_pil = Image.fromarray(img)
                    img_pil.save(output_path)
                    
                    # Store metadata
                    split_metadata[split_name].append({
                        'filename': new_filename,
                        'class_name': class_name,
                        'class_idx': file_info['class_idx'],
                        'original_filename': file_info['filename']
                    })
        
        # Save metadata to JSON
        for split_name in ['train', 'val', 'test']:
            metadata_file = self.output_dir / f"{split_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(split_metadata[split_name], f, indent=2)
        
        return {'split_counts': split_counts, 'split_metadata': split_metadata}
    
    def save_statistics(self, stats):
        """Save preprocessing statistics to JSON"""
        stats_file = self.output_dir / 'preprocessing_stats.json'
        
        # Remove non-serializable items
        stats_copy = stats.copy()
        
        with open(stats_file, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        print(f"\n📊 Statistics saved to: {stats_file}")


def main():
    """Main execution"""
    
    # CONFIGURATION - UPDATE THESE PATHS
    SOURCE_DIR = r"C:\CapstoneProject\dataset\Dataset"  # Your original dataset
    OUTPUT_DIR = r"C:\CapstoneProject\preprocessed_dr_dataset"  # Where to save processed data
    TARGET_SIZE = 224  # 224 for EfficientNet-B0, 300 for B1, 380 for B2, etc.
    
    print("\n🏥 Diabetic Retinopathy Preprocessing Pipeline\n")
    
    # Check if source directory exists
    if not os.path.exists(SOURCE_DIR):
        print("❌ ERROR: Source directory not found!")
        print(f"   Path: {SOURCE_DIR}")
        print("\n📝 Please update SOURCE_DIR in this script to your actual dataset path")
        print("   Based on your screenshot, it should be something like:")
        print("   Windows: C:/Users/YourName/Downloads/dataset/Dataset")
        print("   Linux/Mac: /home/yourname/Downloads/dataset/Dataset")
        return
    
    # Create preprocessor
    preprocessor = DRPreprocessor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE
    )
    
    # Process dataset
    stats = preprocessor.process_dataset(
        train_ratio=0.7,   # 70% for training
        val_ratio=0.15,    # 15% for validation
        test_ratio=0.15    # 15% for testing
    )
    
    print("\n✅ Preprocessing complete!")
    print(f"✨ Your preprocessed dataset is ready at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
