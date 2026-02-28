"""
Setup and Environment Check Script
Verify all dependencies are installed correctly
"""

import sys
import importlib


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✅ GPU is available!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"\n⚠️  GPU is NOT available. Training will use CPU (slower)")
            return False
    except:
        print(f"\n❌ Cannot check GPU status")
        return False


def main():
    """Main setup check"""
    print("=" * 70)
    print("ENVIRONMENT SETUP CHECK")
    print("=" * 70)
    
    print(f"\nPython version: {sys.version}")
    
    print("\n📦 Checking required packages...\n")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('timm', 'timm'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    # Check GPU
    check_gpu()
    
    print("\n" + "=" * 70)
    if all_installed:
        print("✅ ALL DEPENDENCIES ARE INSTALLED!")
        print("=" * 70)
        print("\n🚀 You're ready to start!")
        print("\nNext steps:")
        print("1. Update paths in preprocess_dr_dataset.py")
        print("2. Run: python preprocess_dr_dataset.py")
        print("3. Run: python train_dr_model.py")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING")
        print("=" * 70)
        print("\n📝 To install missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\nOr install individually:")
        print("   pip install torch torchvision")
        print("   pip install timm opencv-python pillow")
        print("   pip install numpy pandas scikit-learn")
        print("   pip install matplotlib seaborn tqdm")


if __name__ == "__main__":
    main()
