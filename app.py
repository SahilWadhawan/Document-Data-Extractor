"""
ExtractTool - Document Classification and Data Extraction Application

Streamlit web interface for OCR-based document processing using Mistral APIs.
"""

import streamlit as st
import os
from PIL import Image
import io
from dotenv import load_dotenv

# Import our custom modules
from src.document_classifier import DocumentClassifier
from src.ocr_processor import OCRProcessor
from src.data_extractor import DataExtractor
from src.utils import (
    load_image_from_bytes, resize_image, validate_image_format,
    get_image_info, format_extracted_data, check_api_key,
    validate_file_type, create_sample_env_file, log_processing_step
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ExtractTool - Document Processing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'classification_result' not in st.session_state:
        st.session_state.classification_result = None
    if 'ocr_result' not in st.session_state:
        st.session_state.ocr_result = None
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None

def check_prerequisites():
    """Check if all prerequisites are met."""
    # Check API key
    if not check_api_key():
        st.error("❌ Mistral API key not found!")
        st.info("Please set your MISTRAL_API_KEY in the .env file.")
        
        if st.button("Create sample .env file"):
            create_sample_env_file()
            st.success("Created sample .env file. Please add your API key and restart the app.")
        
        st.stop()
    
    st.success("✅ Mistral API key found!")

def display_header():
    """Display the application header."""
    st.title("📄 ExtractTool - Document Processing")
    st.markdown("""
    **Automatically classify and extract data from any official document**
    
    **Primary Support**: Driving License, PAN Card, AADHAR Card  
    **Extended Support**: Passport, Voter ID, Certificates, Bills, and other official documents
    """)
    
    st.divider()

def display_sidebar():
    """Display the sidebar with information and settings."""
    st.sidebar.header("📋 Application Info")
    
    st.sidebar.markdown("""
    ### Primary Documents
    - 🚗 **Driving License**
    - 💳 **PAN Card** 
    - 🆔 **AADHAR Card**
    
    ### Extended Support
    - 📘 **Passport**
    - 🗳️ **Voter ID**
    - 📜 **Certificates**
    - 💰 **Bills & Statements**
    - 📋 **Other Official Documents**
    
    ### Supported Formats
    - JPG/JPEG
    - PNG
    - TIFF
    - BMP
    
    ### Processing Steps
    1. Upload document image
    2. Auto-classify document type
    3. Extract text using OCR
    4. Extract relevant data fields
    """)
    
    st.sidebar.divider()
    
    st.sidebar.header("⚙️ Settings")
    
    # Image preprocessing option
    preprocess = st.sidebar.checkbox(
        "Enable image preprocessing",
        value=True,
        help="Applies denoising and thresholding to improve OCR accuracy"
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Classification confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimum confidence required for document classification"
    )
    
    return {
        "preprocess": preprocess,
        "confidence_threshold": confidence_threshold
    }

def process_uploaded_image(uploaded_file, settings):
    """Process the uploaded image through the OCR pipeline."""
    try:
        # Load and validate image
        image_bytes = uploaded_file.read()
        image = load_image_from_bytes(image_bytes)
        
        if image is None:
            st.error("Failed to load the image. Please try a different file.")
            return
        
        if not validate_image_format(image):
            st.error("Unsupported image format. Please use JPG, PNG, TIFF, or BMP.")
            return
        
        # Display image info
        img_info = get_image_info(image)
        st.info(f"📊 Image Info: {img_info['width']}x{img_info['height']} pixels, Format: {img_info['format']}")
        
        # Resize if too large
        image = resize_image(image, max_size=(1024, 1024))
        
        # Display the uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📤 Uploaded Image")
            st.image(image, caption="Document Image", use_column_width=True)
        
        with col2:
            st.subheader("🔄 Processing Status")
            
            # Initialize processors
            classifier = DocumentClassifier()
            ocr_processor = OCRProcessor()
            data_extractor = DataExtractor()
            
            # Step 1: Classification
            with st.spinner("🔍 Classifying document..."):
                log_processing_step("Document Classification", "Starting classification")
                classification_result = classifier.classify_document_sync(image)
                st.session_state.classification_result = classification_result
            
            if classification_result.get("confidence", 0) < settings["confidence_threshold"]:
                st.warning(f"⚠️ Low classification confidence: {classification_result.get('confidence', 0):.2f}")
                st.warning("Classification may not be accurate. Please try a clearer image.")
            
            # Display classification result
            doc_type = classification_result.get("document_type", "Unknown")
            confidence = classification_result.get("confidence", 0)
            
            if doc_type != "Unknown":
                st.success(f"✅ **Document Type:** {doc_type}")
                st.info(f"📊 **Confidence:** {confidence:.2f}")
            else:
                st.error("❌ Could not classify document type")
                if "error" in classification_result:
                    st.error(f"Error: {classification_result['error']}")
                return
            
            # Step 2: OCR Processing
            with st.spinner("📝 Extracting text..."):
                log_processing_step("OCR Processing", f"Processing {doc_type}")
                ocr_result = ocr_processor.extract_structured_text(image, doc_type)
                st.session_state.ocr_result = ocr_result
            
            if ocr_result.get("success"):
                st.success("✅ Text extraction completed")
            else:
                st.error("❌ Text extraction failed")
                if "error" in ocr_result:
                    st.error(f"Error: {ocr_result['error']}")
                return
            
            # Step 3: Data Extraction
            with st.spinner("🎯 Extracting data fields..."):
                log_processing_step("Data Extraction", f"Extracting fields for {doc_type}")
                raw_text = ocr_result.get("structured_text", "")
                extracted_data = data_extractor.extract_data(raw_text, doc_type)
                
                # Validate extracted data
                validation_result = data_extractor.validate_extracted_data(extracted_data)
                
                st.session_state.extracted_data = extracted_data
                st.session_state.validation_result = validation_result
            
            st.success("✅ Data extraction completed")
            st.session_state.processing_complete = True
            
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        log_processing_step("Error", str(e))

def display_results():
    """Display the processing results."""
    if not st.session_state.processing_complete:
        return
    
    st.divider()
    st.header("📊 Processing Results")
    
    # Create tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Extracted Data", "📝 Raw OCR Text", "🔍 Classification Details", "✅ Validation"])
    
    with tab1:
        st.subheader("Extracted Data Fields")
        
        if st.session_state.extracted_data:
            # Check if we have LLM-based extracted information
            if st.session_state.extracted_data.get("extraction_method") == "LLM":
                extracted_info = st.session_state.extracted_data.get("extracted_information", "")
                if extracted_info:
                    # Display as markdown for better formatting
                    st.markdown("### 📋 Extracted Information")
                    st.markdown(extracted_info)
                else:
                    st.info("No structured information could be extracted from the document")
            else:
                # Fall back to formatted text display for older format
                formatted_data = format_extracted_data(st.session_state.extracted_data)
                st.text_area(
                    "Extracted Information",
                    value=formatted_data,
                    height=300,
                    disabled=True
                )
            
            # Show additional extraction details if available
            if st.session_state.extracted_data.get("extraction_quality"):
                quality = st.session_state.extracted_data["extraction_quality"]
                if quality == "high":
                    st.success("✅ High quality extraction")
                elif quality == "medium":
                    st.warning("⚠️ Medium quality extraction - some information may be unclear")
                else:
                    st.error("❌ Low quality extraction - document may be unclear or damaged")
            
            # Download button for extracted data
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download as JSON"):
                    import json
                    json_data = json.dumps(st.session_state.extracted_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name="extracted_data.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📄 Download as Text"):
                    if st.session_state.extracted_data.get("extraction_method") == "LLM":
                        text_data = st.session_state.extracted_data.get("extracted_information", "")
                    else:
                        text_data = format_extracted_data(st.session_state.extracted_data)
                    
                    st.download_button(
                        label="Download Text",
                        data=text_data,
                        file_name="extracted_data.txt",
                        mime="text/plain"
                    )
        else:
            st.info("No data extracted")
    
    with tab2:
        st.subheader("Raw OCR Text")
        
        if st.session_state.ocr_result and st.session_state.ocr_result.get("success"):
            raw_text = st.session_state.ocr_result.get("structured_text", "")
            st.text_area(
                "OCR Output",
                value=raw_text,
                height=300,
                disabled=True
            )
        else:
            st.info("No OCR text available")
    
    with tab3:
        st.subheader("Classification Details")
        
        if st.session_state.classification_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Document Type",
                    st.session_state.classification_result.get("document_type", "Unknown")
                )
            
            with col2:
                confidence = st.session_state.classification_result.get("confidence", 0)
                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}",
                    delta=f"{confidence - 0.5:.2f}" if confidence > 0.5 else f"{confidence - 0.5:.2f}"
                )
            
            # Raw classification response
            if "raw_response" in st.session_state.classification_result:
                with st.expander("Raw Classification Response"):
                    st.text(st.session_state.classification_result["raw_response"])
        else:
            st.info("No classification results available")
    
    with tab4:
        st.subheader("Data Validation & Quality Assessment")
        
        if hasattr(st.session_state, 'validation_result'):
            validation = st.session_state.validation_result
            
            # Overall validation status
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if validation.get("is_valid"):
                    st.success("✅ Data Valid")
                else:
                    st.error("❌ Issues Found")
            
            with col2:
                extraction_method = st.session_state.extracted_data.get("extraction_method", "Unknown")
                st.info(f"🔧 Method: {extraction_method}")
            
            with col3:
                if st.session_state.extracted_data.get("success"):
                    st.success("✅ Extraction OK")
                else:
                    st.error("❌ Extraction Failed")
            
            # Display warnings
            if validation.get("warnings"):
                st.warning("⚠️ **Warnings:**")
                for warning in validation["warnings"]:
                    st.markdown(f"• {warning}")
            
            # Display errors
            if validation.get("errors"):
                st.error("❌ **Errors:**")
                for error in validation["errors"]:
                    st.markdown(f"• {error}")
            
            # Additional quality information
            if not validation.get("warnings") and not validation.get("errors"):
                st.success("🎉 **Perfect!** No issues found with the extracted data.")
                
            # Show processing statistics if available
            if st.session_state.extracted_data.get("processing_time"):
                processing_time = st.session_state.extracted_data["processing_time"]
                st.info(f"⏱️ Processing completed in {processing_time:.2f} seconds")
        else:
            st.info("No validation results available")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Check prerequisites
    check_prerequisites()
    
    # Display sidebar and get settings
    settings = display_sidebar()
    
    # File upload section
    st.header("📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document image...",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Upload an image of any official document (ID cards, certificates, bills, etc.)"
    )
    
    if uploaded_file is not None:
        # Validate file type
        if not validate_file_type(uploaded_file.name):
            st.error("❌ Unsupported file type. Please upload JPG, PNG, TIFF, or BMP files.")
            return
        
        # Process the uploaded image
        if st.button("🚀 Start Processing", type="primary"):
            # Reset session state for new processing
            st.session_state.processing_complete = False
            st.session_state.classification_result = None
            st.session_state.ocr_result = None
            st.session_state.extracted_data = None
            
            process_uploaded_image(uploaded_file, settings)
    
    # Display results if processing is complete
    display_results()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>ExtractTool v1.0 - Powered by Mistral AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
