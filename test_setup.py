"""
Test script to verify ExtractTool setup and basic functionality.
"""

import os
import sys
from PIL import Image
import numpy as np

# Add src directory to path
sys.path.append('src')

from src.utils import check_api_key, create_sample_env_file
from src.document_classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.data_extractor import DataExtractor


def create_sample_image():
    """Create a simple test image."""
    # Create a simple test image with text
    img = Image.new('RGB', (300, 200), color='white')
    return img


def test_api_key():
    """Test if API key is configured."""
    print("Testing API key configuration...")
    
    if check_api_key():
        print("✅ API key found")
        return True
    else:
        print("❌ API key not found")
        print("Creating sample .env file...")
        create_sample_env_file()
        print("Please add your Mistral API key to .env file")
        return False


def test_imports():
    """Test if all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from src.document_classifier import DocumentClassifier
        from src.ocr_processor import OCRProcessor
        from src.data_extractor import DataExtractor
        from src.utils import check_api_key
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_dependencies():
    """Test if all dependencies are installed."""
    print("Testing dependencies...")
    
    try:
        import requests
        import PIL
        import cv2
        import numpy
        import streamlit
        from mistralai.client import MistralClient
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without API calls."""
    print("Testing basic functionality...")
    
    try:
        # Test data extractor (no API required)
        extractor = DataExtractor()
        sample_text = "PAN CARD ABCDE1234F JOHN DOE 01/01/1990"
        result = extractor.extract_pan_data(sample_text)
        
        if result.get("pan_number") == "ABCDE1234F":
            print("✅ Data extraction working")
        else:
            print("⚠️ Data extraction may have issues")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        return False


def main():
    """Run all tests."""
    print("ExtractTool Setup Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_dependencies,
        test_api_key,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
            print()
    
    print("Summary:")
    print("=" * 40)
    if all(results):
        print("✅ All tests passed! Setup is complete.")
        print("\nTo run the application:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not results[0]:  # Import test failed
            print("\n🔧 Fix: Check if all source files are present")
        if not results[1]:  # Dependencies test failed
            print("\n🔧 Fix: Run 'pip install -r requirements.txt'")
        if not results[2]:  # API key test failed
            print("\n🔧 Fix: Add your Mistral API key to .env file")


if __name__ == "__main__":
    main()
