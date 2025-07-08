"""
OCR Processor Module

Handles OCR text extraction from images using Mistral API.
"""

import base64
import io
from typing import Optional, Dict, Any, List
from PIL import Image
import cv2
import numpy as np
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv

load_dotenv()


class OCRProcessor:
    """
    Extracts text from images using Mistral OCR capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OCRProcessor.
        
        Args:
            api_key: Mistral API key. If None, will read from environment variable.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = MistralClient(api_key=self.api_key)
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image object
        """
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding to improve text clarity
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL
        processed_image = Image.fromarray(thresh)
        
        return processed_image
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def extract_text(self, image: Image.Image, preprocess: bool = True) -> Dict[str, Any]:
        """
        Extract text from image using Mistral OCR.
        
        Args:
            image: PIL Image object
            preprocess: Whether to preprocess the image for better OCR
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Preprocess image if requested
            if preprocess:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(processed_image)
            
            # Create the prompt for OCR
            prompt = """
            Extract ALL text from this document image. Please provide:
            1. The complete text content exactly as it appears
            2. Preserve the original formatting and structure as much as possible
            3. Include all numbers, dates, and special characters
            4. If text is unclear, make your best attempt and note uncertainty
            
            Return the extracted text in a clear, readable format.
            """
            
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat(
                model="pixtral-12b-2409",
                messages=messages,
                max_tokens=1000
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "text": extracted_text,
                "preprocessed": preprocess,
                "model_used": "pixtral-12b-2409"
            }
            
        except Exception as e:
            return {
                "success": False,
                "text": "",
                "error": str(e),
                "preprocessed": preprocess
            }
    
    def extract_structured_text(self, image: Image.Image, document_type: str) -> Dict[str, Any]:
        """
        Extract text with structure awareness based on document type.
        
        Args:
            image: PIL Image object
            document_type: Type of document
            
        Returns:
            Dictionary containing structured text extraction
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            base64_image = self._encode_image_to_base64(processed_image)
            
            # Create document-specific prompt
            if document_type == "Driving License":
                structure_prompt = """
                Extract text from this Driving License and organize it into these categories:
                - License Number
                - Name
                - Father's/Husband's Name
                - Date of Birth
                - Address
                - Issue Date
                - Valid Till
                - Vehicle Classes
                - Any other visible text
                """
            elif document_type == "PAN Card":
                structure_prompt = """
                Extract text from this PAN Card and organize it into these categories:
                - PAN Number
                - Name
                - Father's Name
                - Date of Birth
                - Signature
                - Any other visible text
                """
            elif document_type == "AADHAR Card":
                structure_prompt = """
                Extract text from this AADHAR Card and organize it into these categories:
                - AADHAR Number (12 digits)
                - Name
                - Date of Birth / Year of Birth
                - Gender
                - Address
                - Any other visible text
                """
            elif "passport" in document_type.lower():
                structure_prompt = """
                Extract text from this Passport and organize it into these categories:
                - Passport Number
                - Name
                - Nationality
                - Date of Birth
                - Place of Birth
                - Issue Date
                - Expiry Date
                - Issuing Authority
                - Any other visible text
                """
            elif "voter" in document_type.lower():
                structure_prompt = """
                Extract text from this Voter ID and organize it into these categories:
                - Voter ID Number / EPIC Number
                - Name
                - Father's/Husband's Name
                - Date of Birth
                - Address
                - Constituency
                - Any other visible text
                """
            elif "certificate" in document_type.lower():
                structure_prompt = """
                Extract text from this Certificate and organize it into these categories:
                - Certificate Number
                - Name of Holder
                - Type of Certificate
                - Issue Date
                - Issuing Authority
                - Validity Period (if any)
                - Any other visible text
                """
            elif "bill" in document_type.lower():
                structure_prompt = """
                Extract text from this Bill and organize it into these categories:
                - Account/Consumer Number
                - Customer Name
                - Address
                - Bill Date
                - Due Date
                - Amount
                - Service Provider
                - Any other visible text
                """
            else:
                # Generic prompt for any other official document
                structure_prompt = f"""
                Extract text from this {document_type} and organize it into these categories:
                - Document Number/Reference Number (if any)
                - Name(s) of person(s) mentioned
                - Dates (issue date, validity, birth date, etc.)
                - Address (if present)
                - Issuing Authority/Organization
                - Any identification numbers
                - Any other important information
                
                Please extract all visible text accurately and organize it clearly under appropriate headings.
                """
            
            prompt = f"""
            {structure_prompt}
            
            Please extract the text accurately and organize it clearly under the appropriate headings.
            If any field is not clearly visible or missing, indicate it as "Not found" or "Unclear".
            Preserve original formatting and spacing where important.
            """
            
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            # Make API call
            response = self.client.chat(
                model="pixtral-12b-2409",
                messages=messages,
                max_tokens=1000
            )
            
            structured_text = response.choices[0].message.content.strip()
            
            return {
                "success": True,
                "structured_text": structured_text,
                "document_type": document_type,
                "model_used": "pixtral-12b-2409"
            }
            
        except Exception as e:
            return {
                "success": False,
                "structured_text": "",
                "error": str(e),
                "document_type": document_type
            }
