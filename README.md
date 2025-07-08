# ExtractTool - Universal Document Classification and Data Extraction

An advanced OCR application that uses Mistral APIs to classify and extract data from **any official document**. While optimized for Indian identity documents, it can handle various types of official documents worldwide.

## Features

- **Universal Document Classification**: Automatically identifies any type of official document
- **Primary Document Support**: Full data extraction for Driving License, PAN Card, AADHAR Card
- **Extended Document Support**: Basic data extraction for Passport, Voter ID, Certificates, Bills, and more
- **OCR Processing**: Extracts text from images using Mistral OCR with preprocessing
- **Smart Data Extraction**: Context-aware field extraction based on document type
- **Web Interface**: Modern Streamlit interface for easy interaction
- **Image Preprocessing**: Enhances image quality for better OCR accuracy

## Setup

### Prerequisites

- Python 3.8 or higher
- Mistral API key (sign up at [Mistral AI](https://mistral.ai/))

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd ExtractTool
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

4. **Verify setup**:
   ```bash
   python test_setup.py
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

   The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Document**: Click "Choose a document image..." and select an image file
2. **Configure Settings**: Use the sidebar to adjust preprocessing and confidence settings
3. **Process**: Click "Start Processing" to begin analysis
4. **View Results**: Check the results in different tabs:
   - **Extracted Data**: Structured data fields
   - **Raw OCR Text**: Complete text extracted from the image
   - **Classification Details**: Document type and confidence scores
   - **Validation**: Data quality and completeness checks

## Supported Documents

### 🎯 Primary Documents (Full Data Extraction)

### 🚗 Driving License
- License Number
- Name
- Father's/Husband's Name
- Date of Birth
- Address
- Issue Date
- Valid Till
- Vehicle Classes

### 💳 PAN Card
- PAN Number (format: ABCDE1234F)
- Name
- Father's Name
- Date of Birth

### 🆔 AADHAR Card
- AADHAR Number (12 digits)
- Name
- Date of Birth/Year of Birth
- Gender
- Address

### 🌐 Extended Documents (Basic Data Extraction)

### 📘 Passport
- Passport Number
- Name, Nationality
- Date of Birth, Place of Birth
- Issue/Expiry Dates

### 🗳️ Voter ID
- Voter ID/EPIC Number
- Name, Father's Name
- Address, Constituency

### 📜 Certificates
- Certificate Number
- Holder Name
- Issuing Authority
- Issue Date

### 💰 Bills & Statements
- Account/Consumer Number
- Customer Name
- Amounts and Dates
- Service Provider

### 📋 Other Official Documents
- Document Numbers
- Names and Dates
- Addresses
- Issuing Authorities

**Note**: The system can classify and extract basic information from any official document, even if not explicitly listed above.

## Project Structure

```
ExtractTool/
├── app.py                      # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_classifier.py  # Document type classification
│   ├── ocr_processor.py        # OCR text extraction
│   ├── data_extractor.py       # Extract specific data fields
│   └── utils.py               # Utility functions
├── test_setup.py              # Setup verification script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .github/
│   └── copilot-instructions.md # Copilot customization
└── README.md
```

## API Configuration

The application uses Mistral AI's APIs:

- **Pixtral-12B-2409**: For vision tasks (classification and OCR)
- **Image Processing**: Supports base64 encoded images
- **Rate Limiting**: Built-in retry logic and error handling

## Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure your `.env` file contains a valid Mistral API key
   - Check that the key has appropriate permissions

2. **Import Errors**:
   - Run `pip install -r requirements.txt` to install all dependencies
   - Ensure you're using Python 3.8 or higher

3. **Image Upload Issues**:
   - Supported formats: JPG, JPEG, PNG, TIFF, BMP
   - Maximum recommended size: 1024x1024 pixels

4. **Poor OCR Results**:
   - Ensure the document image is clear and well-lit
   - Enable image preprocessing in the sidebar
   - Try different image orientations

### Development

To contribute or modify the project:

1. **Run tests**:
   ```bash
   python test_setup.py
   ```

2. **Debug mode**:
   ```bash
   streamlit run app.py --logger.level=debug
   ```

3. **VS Code Tasks**:
   - Use Ctrl+Shift+P → "Tasks: Run Task" → "Run ExtractTool App"

## License

This project is for educational and research purposes. Please ensure compliance with applicable laws and regulations when processing identity documents.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `python test_setup.py` to verify your setup
3. Review the application logs in the terminal
