"""
Document Classifier Module

Uses Mistral API to classify documents into Driving License, PAN, or AADHAR types.
"""

import base64
import io
from typing import Optional, Dict, Any
from PIL import Image
import requests
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv

load_dotenv()


class DocumentClassifier:
    """
    Classifies documents into predefined categories using Mistral API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DocumentClassifier.
        
        Args:
            api_key: Mistral API key. If None, will read from environment variable.
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = MistralClient(api_key=self.api_key)
        # Primary supported document types with full data extraction
        self.primary_document_types = ["Driving License", "PAN Card", "AADHAR Card"]
        # Additional document types that can be classified
        self.supported_document_types = [
            "Driving License", "PAN Card", "AADHAR Card", "Passport", "Voter ID", 
            "Birth Certificate", "Death Certificate", "Marriage Certificate",
            "Education Certificate", "Mark Sheet", "Degree Certificate",
            "Income Certificate", "Caste Certificate", "Domicile Certificate",
            "Ration Card", "Electricity Bill", "Water Bill", "Gas Bill",
            "Bank Statement", "Salary Slip", "Insurance Policy", "Property Document"
        ]
    
    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def classify_document_sync(self, image: Image.Image) -> Dict[str, Any]:
        """
        Synchronous version of classify_document.
        
        Args:
            image: PIL Image object of the document
            
        Returns:
            Dictionary containing classification result and confidence
        """
        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image)
            
            # Create the prompt for classification
            prompt = """
            Analyze this document image and classify it into the most appropriate official document type.
            
            Common Indian official documents include:
            - Identity Documents: Driving License, PAN Card, AADHAR Card, Passport, Voter ID
            - Certificates: Birth Certificate, Death Certificate, Marriage Certificate, Education Certificate, Mark Sheet, Degree Certificate
            - Income/Social Documents: Income Certificate, Caste Certificate, Domicile Certificate
            - Utility Documents: Ration Card, Electricity Bill, Water Bill, Gas Bill
            - Financial Documents: Bank Statement, Salary Slip, Insurance Policy
            - Property Documents: Property Document, Land Records
            
            Look for distinctive features such as:
            - Government logos, seals, and official headers
            - Document titles and headings
            - Specific numbering formats (PAN: ABCDE1234F, AADHAR: 12 digits, etc.)
            - Issuing authority names
            - Official stamps and signatures
            
            Respond with the most specific document type you can identify.
            If you cannot determine the specific type, respond with a general category like "Official Document" or "Identity Document".
            If it's clearly not an official document, respond with "Non-Official Document".
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
                max_tokens=50
            )
            
            classification = response.choices[0].message.content.strip()
            
            # Validate classification result
            confidence = 0.0
            
            # Check if it's one of our primary supported types (highest confidence)
            if classification in self.primary_document_types:
                confidence = 0.95
            # Check if it's in our extended supported types
            elif classification in self.supported_document_types:
                confidence = 0.9
            # Check for partial matches in primary types
            elif any(doc_type.lower() in classification.lower() for doc_type in self.primary_document_types):
                for doc_type in self.primary_document_types:
                    if doc_type.lower() in classification.lower():
                        classification = doc_type
                        confidence = 0.8
                        break
            # Check for partial matches in supported types
            elif any(doc_type.lower() in classification.lower() for doc_type in self.supported_document_types):
                for doc_type in self.supported_document_types:
                    if doc_type.lower() in classification.lower():
                        classification = doc_type
                        confidence = 0.75
                        break
            # Check for general official document indicators
            elif any(keyword in classification.lower() for keyword in ["certificate", "card", "license", "bill", "document", "statement"]):
                confidence = 0.6
            # Unknown or non-official document
            elif classification.lower() in ["unknown", "non-official document"]:
                confidence = 0.0
            else:
                # Some classification was made, give it moderate confidence
                confidence = 0.5
            
            return {
                "document_type": classification,
                "confidence": confidence,
                "raw_response": response.choices[0].message.content
            }
            
        except Exception as e:
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "error": str(e)
            }
