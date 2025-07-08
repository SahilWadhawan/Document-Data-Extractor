"""
Utility functions for the ExtractTool project.
"""

import os
import io
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np


def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from a file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Load an image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        
    Returns:
        PIL Image object or None if loading fails
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        print(f"Error loading image from bytes: {e}")
        return None


def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimensions (width, height)
        
    Returns:
        Resized PIL Image object
    """
    try:
        # Calculate new size maintaining aspect ratio
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image


def validate_image_format(image: Image.Image) -> bool:
    """
    Validate if image format is supported.
    
    Args:
        image: PIL Image object
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = ['JPEG', 'PNG', 'TIFF', 'BMP']
    return image.format in supported_formats


def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary containing image information
    """
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "width": image.width,
        "height": image.height
    }


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = " ".join(text.split())
    
    # Remove common OCR artifacts
    artifacts = ["@", "#", "$", "%", "^", "&", "*", "~", "`"]
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")
    
    return cleaned.strip()


def format_extracted_data(data: dict) -> str:
    """
    Format extracted data for display.
    
    Args:
        data: Dictionary containing extracted data
        
    Returns:
        Formatted string representation
    """
    if not data:
        return "No data extracted"
    
    formatted_lines = []
    doc_type = data.get("document_type", "Unknown")
    extraction_method = data.get("extraction_method", "Legacy")
    
    formatted_lines.append(f"Document Type: {doc_type}")
    formatted_lines.append(f"Extraction Method: {extraction_method}")
    formatted_lines.append("-" * 50)
    
    # Check if this is LLM-based extraction
    if extraction_method == "LLM":
        # Check extraction success
        if not data.get("success", True):
            formatted_lines.append(f"❌ Extraction Failed: {data.get('error', 'Unknown error')}")
            return "\n".join(formatted_lines)
        
        # Display the structured LLM response
        if data.get("extracted_information"):
            formatted_lines.append("EXTRACTED INFORMATION:")
            formatted_lines.append(data["extracted_information"])
            formatted_lines.append("")
        
        # Display key information summary
        key_info = data.get("key_information", {})
        if key_info:
            formatted_lines.append("KEY INFORMATION SUMMARY:")
            for key, value in key_info.items():
                formatted_lines.append(f"{key.replace('_', ' ').title()}: {value}")
            formatted_lines.append("")
        
        # Display structured data sections
        structured_data = data.get("structured_data", {})
        if structured_data:
            formatted_lines.append("STRUCTURED DATA BREAKDOWN:")
            for section, items in structured_data.items():
                if items:  # Only show non-empty sections
                    section_title = section.replace('_', ' ').title()
                    formatted_lines.append(f"\n{section_title}:")
                    for item in items:
                        formatted_lines.append(f"  • {item}")
        
        # Add parsing errors if any
        if data.get("parsing_error"):
            formatted_lines.append(f"\n⚠️ Parsing Note: {data['parsing_error']}")
    
    else:
        # Legacy format for backward compatibility
        if doc_type == "PAN Card":
            fields = ["pan_number", "name", "father_name", "date_of_birth"]
            labels = ["PAN Number", "Name", "Father's Name", "Date of Birth"]
        elif doc_type == "Driving License":
            fields = ["license_number", "name", "father_name", "date_of_birth", 
                     "address", "issue_date", "valid_till", "vehicle_classes"]
            labels = ["License Number", "Name", "Father's Name", "Date of Birth",
                     "Address", "Issue Date", "Valid Till", "Vehicle Classes"]
        elif doc_type == "AADHAR Card":
            fields = ["aadhar_number", "name", "date_of_birth", "year_of_birth", 
                     "gender", "address"]
            labels = ["AADHAR Number", "Name", "Date of Birth", "Year of Birth",
                     "Gender", "Address"]
        else:
            # For other documents, show all available fields
            fields = [k for k in data.keys() if k not in ["document_type", "raw_text", "error", "extraction_method"]]
            labels = [k.replace("_", " ").title() for k in fields]
        
        for field, label in zip(fields, labels):
            value = data.get(field)
            if value:
                formatted_lines.append(f"{label}: {value}")
    
    # Add any errors
    if data.get("error"):
        formatted_lines.append(f"\n❌ Error: {data['error']}")
    
    return "\n".join(formatted_lines)


def check_api_key() -> bool:
    """
    Check if Mistral API key is available.
    
    Returns:
        True if API key is found, False otherwise
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    return api_key is not None and len(api_key.strip()) > 0


def get_supported_file_types() -> List[str]:
    """
    Get list of supported file types for upload.
    
    Returns:
        List of supported file extensions
    """
    return ["jpg", "jpeg", "png", "tiff", "bmp", "pdf"]


def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not filename:
        return False
    
    extension = filename.lower().split(".")[-1]
    return extension in get_supported_file_types()


def create_sample_env_file():
    """
    Create a sample .env file if it doesn't exist.
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("MISTRAL_API_KEY=your_mistral_api_key_here\n")
        print(f"Created sample {env_path} file. Please add your Mistral API key.")


def log_processing_step(step: str, details: str = ""):
    """
    Log processing steps for debugging.
    
    Args:
        step: Processing step name
        details: Additional details
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {step}"
    if details:
        log_message += f": {details}"
    print(log_message)
