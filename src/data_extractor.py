"""
Data Extractor Module

Extracts specific data fields from OCR text using LLM-based intelligent extraction.
"""

import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from mistralai.client import MistralClient
import os
from dotenv import load_dotenv

load_dotenv()


class DataExtractor:
    """
    Extracts specific data fields from OCR text using intelligent LLM-based extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the DataExtractor with LLM client."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable.")
        
        self.client = MistralClient(api_key=self.api_key)
    
    def extract_data_with_llm(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        Extract relevant data from document text using LLM.
        
        Args:
            text: OCR extracted text
            document_type: Type of document
            
        Returns:
            Dictionary containing extracted data
        """
        try:
            # Create a comprehensive prompt for data extraction
            prompt = f"""
            You are an expert document analysis AI. Analyze the following OCR-extracted text from a "{document_type}" document and extract all relevant information.

            DOCUMENT TEXT:
            {text}

            INSTRUCTIONS:
            1. Identify and extract ALL relevant information from this {document_type}
            2. Organize the information into clear categories
            3. For each piece of information, provide the exact text as it appears in the document
            4. Include common fields like names, numbers, dates, addresses, and any document-specific information
            5. If certain information is unclear or partially visible, note it as "Unclear" or "Partial"
            6. Do not make up or infer information that is not clearly visible in the text

            EXPECTED INFORMATION TO LOOK FOR (extract only what is present):

            For Identity Documents (PAN, AADHAR, Driving License, Passport, Voter ID):
            - Document number/ID
            - Full name
            - Father's/Husband's name
            - Date of birth
            - Address
            - Gender (if applicable)
            - Issue date and expiry date
            - Any other identifying information

            For Certificates (Birth, Death, Marriage, Education, etc.):
            - Certificate number
            - Names of individuals
            - Dates (birth, death, marriage, graduation, etc.)
            - Places (birth place, institution, etc.)
            - Issuing authority
            - Registration details

            For Bills/Statements:
            - Account/Consumer number
            - Customer name
            - Service address
            - Bill amount
            - Due date
            - Service period
            - Provider information

            For Other Documents:
            - Any reference numbers
            - Names mentioned
            - Important dates
            - Addresses
            - Official stamps or signatures mentioned
            - Key information specific to the document type

            RESPONSE FORMAT:
            Provide a structured response using proper markdown formatting:

            ## Document Analysis Results

            ### Primary Identifiers
            - **Document Number:** [number if found]
            - **Reference ID:** [any other IDs]

            ### Personal Information  
            - **Name:** [full name]
            - **Father's/Husband's Name:** [if applicable]
            - **Date of Birth:** [if found]
            - **Gender:** [if mentioned]

            ### Dates
            - **Issue Date:** [if found]
            - **Expiry Date:** [if found] 
            - **Other Dates:** [any other relevant dates]

            ### Contact & Address Information
            - **Address:** [full address if found]
            - **Phone:** [if mentioned]
            - **Email:** [if mentioned]

            ### Document-Specific Information
            - [Any information specific to this document type]

            ### Additional Details
            - [Any other relevant information not covered above]

            ### Notes
            > **Important:** Note any unclear, partial, or damaged text here

            Only include sections that contain actual extracted information. If no information is found for a category, omit that section entirely.
            Use proper markdown formatting with headers (##, ###), bold text (**text**), and bullet points (-).
            """

            # Make API call
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self.client.chat(
                model="mistral-large-latest",
                messages=messages,
                max_tokens=1000,
                temperature=0.1  # Low temperature for consistent extraction
            )

            extracted_content = response.choices[0].message.content.strip()

            # Parse the structured response
            parsed_data = self._parse_llm_response(extracted_content, document_type, text)
            
            return parsed_data

        except Exception as e:
            return {
                "document_type": document_type,
                "extraction_method": "LLM",
                "success": False,
                "error": str(e),
                "raw_text": text
            }
    
    def _parse_llm_response(self, llm_response: str, document_type: str, original_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format.
        
        Args:
            llm_response: Response from LLM
            document_type: Type of document
            original_text: Original OCR text
            
        Returns:
            Structured dictionary with extracted data
        """
        data = {
            "document_type": document_type,
            "extraction_method": "LLM",
            "success": True,
            "raw_text": original_text,
            "extracted_information": llm_response,
            "structured_data": {}
        }
        
        try:
            # Extract information from different sections
            sections = {
                "primary_identifiers": [],
                "personal_information": [],
                "dates": [],
                "addresses": [],
                "document_specific_information": [],
                "additional_details": []
            }
            
            # Parse the structured response
            current_section = None
            lines = llm_response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Identify section headers
                if "Primary Identifiers:" in line:
                    current_section = "primary_identifiers"
                elif "Personal Information:" in line:
                    current_section = "personal_information"
                elif "Dates:" in line:
                    current_section = "dates"
                elif "Addresses:" in line:
                    current_section = "addresses"
                elif "Document-Specific Information:" in line:
                    current_section = "document_specific_information"
                elif "Additional Details:" in line:
                    current_section = "additional_details"
                elif line.startswith('- ') and current_section:
                    # Extract bullet point content
                    content = line[2:].strip()  # Remove '- '
                    if content and content != "None" and content != "Not found":
                        sections[current_section].append(content)
            
            # Store parsed sections
            data["structured_data"] = sections
            
            # Create a summary of key information
            key_info = {}
            
            # Try to identify the most important pieces of information
            for identifier in sections["primary_identifiers"]:
                if any(keyword in identifier.lower() for keyword in ["number", "id", "pan", "aadhar", "license", "passport"]):
                    key_info["primary_id"] = identifier
                    break
            
            # Extract the first name found
            for personal in sections["personal_information"]:
                if "name" in personal.lower() and "father" not in personal.lower():
                    key_info["name"] = personal
                    break
            
            # Extract the first date found
            if sections["dates"]:
                key_info["primary_date"] = sections["dates"][0]
            
            # Extract the first address found
            if sections["addresses"]:
                key_info["address"] = sections["addresses"][0]
            
            data["key_information"] = key_info
            
        except Exception as e:
            data["parsing_error"] = str(e)
        
        return data
    
    
    def extract_data(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        Extract data using intelligent LLM-based analysis.
        
        Args:
            text: OCR extracted text
            document_type: Type of document
            
        Returns:
            Dictionary containing extracted data
        """
        if not text or len(text.strip()) < 10:
            return {
                "document_type": document_type,
                "extraction_method": "LLM",
                "success": False,
                "error": "Insufficient text for extraction",
                "raw_text": text
            }
        
        # Use LLM-based extraction for all document types
        return self.extract_data_with_llm(text, document_type)
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted data for completeness and quality.
        
        Args:
            data: Dictionary containing extracted data
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check if extraction was successful
        if not data.get("success", True):
            validation_result["errors"].append(f"Extraction failed: {data.get('error', 'Unknown error')}")
            validation_result["is_valid"] = False
            return validation_result
        
        # Check if sufficient text was provided
        raw_text = data.get("raw_text", "")
        if not raw_text or len(raw_text.strip()) < 10:
            validation_result["errors"].append("Insufficient text extracted from document")
            validation_result["is_valid"] = False
        
        # Check if any structured data was extracted
        structured_data = data.get("structured_data", {})
        key_info = data.get("key_information", {})
        
        if not structured_data and not key_info:
            validation_result["warnings"].append("No structured data could be extracted")
        
        # Check for key information based on document type
        doc_type = data.get("document_type", "").lower()
        
        if any(keyword in doc_type for keyword in ["pan", "aadhar", "driving", "passport", "voter"]):
            # Identity documents should have at least a name and an ID
            if not key_info.get("name") and not any("name" in str(item).lower() for section in structured_data.values() for item in section):
                validation_result["warnings"].append("No name found in identity document")
            
            if not key_info.get("primary_id") and not structured_data.get("primary_identifiers"):
                validation_result["warnings"].append("No identification number found")
        
        elif "certificate" in doc_type:
            # Certificates should have some form of registration or certificate number
            if not structured_data.get("primary_identifiers"):
                validation_result["warnings"].append("No certificate or registration number found")
        
        elif "bill" in doc_type or "statement" in doc_type:
            # Bills should have account numbers and amounts
            if not structured_data.get("primary_identifiers"):
                validation_result["warnings"].append("No account or reference number found")
        
        # Check for common issues
        if data.get("parsing_error"):
            validation_result["warnings"].append(f"Some parsing issues occurred: {data['parsing_error']}")
        
        # Check extraction quality
        extracted_info = data.get("extracted_information", "")
        if "unclear" in extracted_info.lower() or "partial" in extracted_info.lower():
            validation_result["warnings"].append("Some information in the document is unclear or partially visible")
        
        return validation_result
