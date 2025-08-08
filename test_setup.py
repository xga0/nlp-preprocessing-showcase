#!/usr/bin/env python3
"""
Test script to verify the setup and package installations.
This script checks if all required packages are installed and working correctly.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchtext', 'TorchText'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'TQDM'),
        ('lightlemma', 'LightLemma'),
        ('emoticon_fix', 'Emoticon Fix'),
        ('contraction_fix', 'Contraction Fix'),
        ('transformers', 'Transformers')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"SUCCESS: {name} imported successfully")
        except ImportError as e:
            print(f"FAILED: {name} import failed: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_package_functionality():
    """Test if the three main packages work correctly."""
    print("\nTesting package functionality...")
    
    try:
        # Test lightlemma
        from lightlemma import text_to_lemmas, text_to_stems
        test_text = "The cats are running faster than dogs"
        lemmas = text_to_lemmas(test_text)
        stems = text_to_stems(test_text)
        print(f"SUCCESS: LightLemma working: {lemmas[:3]}... (lemmas), {stems[:3]}... (stems)")
    except Exception as e:
        print(f"FAILED: LightLemma test failed: {e}")
        return False
    
    try:
        # Test emoticon_fix
        from emoticon_fix import replace_emoticons as fix_emoticons
        test_text = "I'm happy! 😊"
        fixed = fix_emoticons(test_text)
        print(f"SUCCESS: Emoticon Fix working: {test_text} → {fixed}")
    except Exception as e:
        print(f"FAILED: Emoticon Fix test failed: {e}")
        return False
    
    try:
        # Test contraction_fix
        from contraction_fix import fix as fix_contractions
        test_text = "I don't like this"
        fixed = fix_contractions(test_text)
        print(f"SUCCESS: Contraction Fix working: {test_text} → {fixed}")
    except Exception as e:
        print(f"FAILED: Contraction Fix test failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test if the dataset files exist."""
    print("\nTesting data files...")
    
    required_files = [
        'twitter_training.csv',
        'twitter_validation.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            print(f"SUCCESS: {file} exists ({size:.1f} MB)")
        else:
            print(f"FAILED: {file} missing")
            missing_files.append(file)
    
    return missing_files

def test_custom_modules():
    """Test if our custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        from data_processor import TwitterDataProcessor
        print("SUCCESS: data_processor.py imported successfully")
    except Exception as e:
        print(f"FAILED: data_processor.py import failed: {e}")
        return False
    
    try:
        from model import TwitterSentimentModel
        print("SUCCESS: model.py imported successfully")
    except Exception as e:
        print(f"FAILED: model.py import failed: {e}")
        return False
    
    return True

def test_pytorch_setup():
    """Test PyTorch setup and device availability."""
    print("\nTesting PyTorch setup...")
    
    try:
        import torch
        print(f"SUCCESS: PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"SUCCESS: CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"SUCCESS: CUDA version: {torch.version.cuda}")
        else:
            print("WARNING: CUDA not available, will use CPU")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print("SUCCESS: Basic tensor operations working")
        
    except Exception as e:
        print(f"FAILED: PyTorch test failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("="*80)
    print("SETUP TESTING")
    print("="*80)
    
    all_tests_passed = True
    
    # Test 1: Package imports
    failed_imports = test_imports()
    if failed_imports:
        print(f"\nFAILED: Failed imports: {failed_imports}")
        print("Please install missing packages:")
        print("pip install -r requirements.txt")
        all_tests_passed = False
    
    # Test 2: Package functionality
    if not test_package_functionality():
        all_tests_passed = False
    
    # Test 3: Data files
    missing_files = test_data_files()
    if missing_files:
        print(f"\nFAILED: Missing files: {missing_files}")
        print("Please download the dataset files from Kaggle")
        all_tests_passed = False
    
    # Test 4: Custom modules
    if not test_custom_modules():
        all_tests_passed = False
    
    # Test 5: PyTorch setup
    if not test_pytorch_setup():
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*80)
    if all_tests_passed:
        print("SUCCESS: ALL TESTS PASSED!")
        print("Your setup is ready for training and inference.")
        print("\nNext steps:")
        print("1. Run 'python demo.py' to see the packages in action")
        print("2. Run 'python train.py' to train the model")
        print("3. Run 'python inference.py' to make predictions")
    else:
        print("FAILED: SOME TESTS FAILED!")
        print("Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Download dataset files from Kaggle")
        print("3. Check Python version compatibility")
    
    print("="*80)

if __name__ == "__main__":
    main()
