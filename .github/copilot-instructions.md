<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# ExtractTool Project Instructions

This is a Python project for OCR document classification and data extraction using Mistral APIs.

## Key Technologies
- **Mistral AI APIs**: For OCR and document analysis
- **OpenCV & Pillow**: For image processing
- **Streamlit**: For web interface
- **Python**: Core programming language

## Project Guidelines
- Use async/await patterns for API calls when possible
- Implement proper error handling for API requests
- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Include docstrings for all classes and methods

## API Integration
- Use the Mistral client library for API interactions
- Implement retry logic for failed API requests
- Cache API responses when appropriate
- Handle rate limiting gracefully

## Document Processing
- Support common image formats (JPG, PNG, PDF)
- Implement image preprocessing for better OCR accuracy
- Validate extracted data against expected patterns
- Provide confidence scores for extracted information
