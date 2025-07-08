"""
Demo script showing how to use ExtractTool modules programmatically.
"""

import os
import sys
from PIL import Image

# Add src directory to path
sys.path.append('src')

from src.document_classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.data_extractor import DataExtractor
from src.utils import load_image_from_path, format_extracted_data


def demo_with_sample_image():
    """
    Demo function showing the complete OCR pipeline.
    Note: This requires a valid Mistral API key in .env file.
    """
    print("ExtractTool Demo")
    print("=" * 50)
    
    # Check if API key is configured
    from src.utils import check_api_key
    if not check_api_key():
        print("❌ Please configure your Mistral API key in .env file first!")
        print("Run: python test_setup.py for setup verification")
        return
    
    # Create a sample image (in real usage, load from file)
    print("📸 Creating sample image...")
    sample_image = Image.new('RGB', (400, 300), color='white')
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        classifier = DocumentClassifier()
        ocr_processor = OCRProcessor()
        data_extractor = DataExtractor()
        
        # Step 1: Classify document
        print("🔍 Classifying document...")
        classification_result = classifier.classify_document_sync(sample_image)
        
        doc_type = classification_result.get("document_type", "Unknown")
        confidence = classification_result.get("confidence", 0)
        
        print(f"📋 Document Type: {doc_type}")
        print(f"📊 Confidence: {confidence:.2f}")
        
        if doc_type == "Unknown":
            print("⚠️ Could not classify document (expected with sample image)")
            return
        
        # Step 2: Extract text
        print("📝 Extracting text...")
        ocr_result = ocr_processor.extract_structured_text(sample_image, doc_type)
        
        if ocr_result.get("success"):
            extracted_text = ocr_result.get("structured_text", "")
            print(f"📄 Extracted Text: {extracted_text[:100]}...")
        else:
            print("❌ OCR extraction failed")
            return
        
        # Step 3: Extract data fields
        print("🎯 Extracting data fields...")
        extracted_data = data_extractor.extract_data(extracted_text, doc_type)
        
        # Format and display results
        print("\n📊 Results:")
        print("-" * 30)
        formatted_data = format_extracted_data(extracted_data)
        print(formatted_data)
        
        # Validate extracted data
        validation_result = data_extractor.validate_extracted_data(extracted_data)
        print(f"\n✅ Data Valid: {validation_result.get('is_valid')}")
        
        if validation_result.get("warnings"):
            print("⚠️ Warnings:")
            for warning in validation_result["warnings"]:
                print(f"  • {warning}")
        
        if validation_result.get("errors"):
            print("❌ Errors:")
            for error in validation_result["errors"]:
                print(f"  • {error}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_data_extraction():
    """
    Demo function showing LLM-based data extraction without API calls for sample text.
    """
    print("LLM-Based Data Extraction Demo")
    print("=" * 50)
    
    # Check if API key is available
    from src.utils import check_api_key
    if not check_api_key():
        print("❌ Please configure your Mistral API key in .env file first!")
        print("This demo requires API access for LLM-based extraction.")
        return
    
    data_extractor = DataExtractor()
    
    # Sample document texts for different types
    sample_documents = {
        "PAN Card": """
        INCOME TAX DEPARTMENT
        PERMANENT ACCOUNT NUMBER CARD
        PAN: ABCDE1234F
        Name: JOHN DOE
        Father's Name: RICHARD DOE
        Date of Birth: 01/01/1990
        Signature
        """,
        
        "AADHAR Card": """
        भारत सरकार Government of India
        आधार AADHAAR
        Name: JANE SMITH
        Date of Birth: 15/08/1985
        Female
        AADHAR: 1234 5678 9012
        Address: 123 Main Street, Mumbai 400001
        """,
        
        "Birth Certificate": """
        REGISTRAR OF BIRTHS AND DEATHS
        BIRTH CERTIFICATE
        Certificate No: BC/2020/12345
        Name of Child: ALEX KUMAR
        Date of Birth: 25/12/2020
        Place of Birth: Mumbai, Maharashtra
        Father's Name: RAVI KUMAR
        Mother's Name: PRIYA KUMAR
        Registration Date: 30/12/2020
        Registrar Signature
        """,
        
        "Electricity Bill": """
        MUMBAI ELECTRICITY BOARD
        ELECTRICITY BILL
        Consumer No: 123456789
        Consumer Name: RAMESH SHARMA
        Service Address: 789 Park Lane, Mumbai 400020
        Bill Period: Jan 2024 to Feb 2024
        Units Consumed: 150
        Amount Due: Rs. 2,450
        Due Date: 15/03/2024
        """,
        
        "Passport": """
        REPUBLIC OF INDIA
        PASSPORT
        Passport No: Z1234567
        Type: P
        Country Code: IND
        Surname: PATEL
        Given Names: ARJUN KUMAR
        Date of Birth: 10/05/1988
        Place of Birth: AHMEDABAD
        Date of Issue: 15/06/2018
        Date of Expiry: 14/06/2028
        """
    }
    
    for doc_type, sample_text in sample_documents.items():
        print(f"\n📋 Testing {doc_type} extraction:")
        print("-" * 40)
        
        try:
            # Use LLM-based extraction
            extracted_data = data_extractor.extract_data(sample_text, doc_type)
            
            # Format and display results
            formatted_data = format_extracted_data(extracted_data)
            print(formatted_data)
            
            # Show validation results
            validation = data_extractor.validate_extracted_data(extracted_data)
            print(f"\n✅ Extraction Valid: {validation.get('is_valid')}")
            
            if validation.get("warnings"):
                print("⚠️ Warnings:")
                for warning in validation["warnings"]:
                    print(f"  • {warning}")
            
            if validation.get("errors"):
                print("❌ Errors:")
                for error in validation["errors"]:
                    print(f"  • {error}")
                    
        except Exception as e:
            print(f"❌ Extraction failed for {doc_type}: {e}")
        
        print("\n" + "=" * 50)
    
    print("\n" + "=" * 50)
    
    # Sample Passport text
    print("📋 Testing Passport extraction (General Document):")
    passport_text = """
    PASSPORT
    Republic of India
    Passport No: Z1234567
    Name: PRIYA SHARMA
    Nationality: IND
    Date of Birth: 25/03/1988
    Place of Birth: NEW DELHI
    Issue Date: 10/01/2020
    Expiry Date: 09/01/2030
    """
    
    passport_data = data_extractor.extract_general_document_data(passport_text, "Passport")
    print(format_extracted_data(passport_data))
    
    print("\n" + "=" * 50)
    
    # Sample Electricity Bill text
    print("📋 Testing Electricity Bill extraction (General Document):")
    bill_text = """
    STATE ELECTRICITY BOARD
    ELECTRICITY BILL
    Consumer No: 1234567890
    Name: RAJESH KUMAR
    Address: 789 Green Street, Delhi 110001
    Bill Date: 15/11/2024
    Due Date: 30/11/2024
    Total Amount: ₹2,450.00
    """
    
    bill_data = data_extractor.extract_general_document_data(bill_text, "Electricity Bill")
    print(format_extracted_data(bill_data))


def main():
    """Main demo function."""
    print("Choose demo mode:")
    print("1. Full OCR Demo (requires API key)")
    print("2. LLM-based Data Extraction Demo (requires API key)")
    
    try:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            demo_with_sample_image()
        elif choice == "2":
            demo_data_extraction()
        else:
            print("Invalid choice. Running LLM extraction demo...")
            demo_data_extraction()
            
    except KeyboardInterrupt:
        print("\n\nDemo cancelled.")
    except Exception as e:
        print(f"\nDemo error: {e}")


if __name__ == "__main__":
    main()
