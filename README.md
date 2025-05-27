# DocuTranslate Pro 📄✨

An intelligent PDF processing and translation application that extracts content from PDFs, filters out logos and unwanted elements, and converts documents to translated Word format with preserved layout and formatting.

## 🚀 Features

- **Smart PDF Processing**: Extract text, images, and layout information from PDF documents
- **Logo Detection & Filtering**: Automatically detect and remove logos or watermarks using computer vision
- **AI-Powered Translation**: Translate documents while preserving formatting and structure
- **Layout Preservation**: Maintain original document layout in the output Word document
- **Interactive Web Interface**: Easy-to-use Streamlit web application
- **Batch Processing**: Handle multiple documents efficiently
- **Custom Logo Upload**: Upload your own logo images for filtering
- **Document Intelligence**: Extract summaries and key information using Azure Document Intelligence

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: 
  - TensorFlow/Keras (VGG16 for image feature extraction)
  - Azure OpenAI (GPT-4 for translation and content processing)
  - Azure Document Intelligence
- **Document Processing**: 
  - PyMuPDF for PDF manipulation
  - python-docx for Word document generation
  - PIL/Pillow for image processing
- **Language Processing**: LangChain framework

## 📋 Prerequisites

- Python 3.12 (recommended)
- Azure OpenAI API access
- Azure Document Intelligence service

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/docutranslate-pro.git
   cd docutranslate-pro
   ```

2. **Create and activate virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd app
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   
   Create a `.env` file in the `app` directory with your Azure credentials:
   ```env
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_KEY=your_azure_openai_key
   
   # Azure Document Intelligence Configuration
   AZURE_DOCUMENTINT_ENDPOINT=your_document_intelligence_endpoint
   AZURE_DOCUMENTINT_KEY=your_document_intelligence_key
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   cd app
   streamlit run app.py
   ```

2. **Access the web interface**
   
   Open your browser and navigate to `http://localhost:8501`

3. **Process documents**
   - Upload a PDF file using the file uploader
   - Optionally upload a logo image to filter out from the document
   - Click "Process and Translate" to start the conversion
   - Download the translated Word document when processing is complete

## 📁 Project Structure

```
docutranslate-pro/
├── app/
│   ├── app.py                    # Main Streamlit application
│   ├── backend.py                # Core processing logic
│   ├── translator.py             # Translation functionality
│   ├── table_generate_agent.py   # Document analysis and summary generation
│   └── requirements.txt          # Python dependencies
├── Data/
│   ├── input_file/              # Sample input PDF files
│   ├── output_file/             # Generated output files
│   ├── logo/                    # Default logo files
│   └── example_output/          # Example outputs
├── venv/                        # Virtual environment (created after setup)
└── README.md                    # This file
```

## 🔄 Processing Pipeline

1. **Document Upload**: PDF files are uploaded via the web interface
2. **Layout Analysis**: Azure Document Intelligence extracts text and layout information
3. **Image Processing**: Images are extracted and analyzed using VGG16 neural network
4. **Logo Detection**: Custom logo images are compared against extracted images
5. **Content Filtering**: Logos and unwanted elements are filtered out
6. **Translation**: Text content is translated using Azure OpenAI GPT-4
7. **Document Generation**: Translated content is formatted into a Word document
8. **Output**: Clean, translated Word document is generated for download

## 🎯 Use Cases

- **Document Localization**: Translate business documents while maintaining professional formatting
- **Content Migration**: Convert PDF documents to editable Word format
- **Brand Compliance**: Remove competitor logos or watermarks from documents
- **Academic Research**: Process and translate research papers and documentation
- **Legal Documents**: Translate contracts and legal documents with layout preservation

## 📦 Dependencies

Key dependencies include:
- `streamlit>=1.45.0` - Web application framework
- `tensorflow>=2.17.0` - Machine learning framework
- `azure-ai-documentintelligence>=1.0.0b4` - Azure document processing
- `langchain>=0.3.25` - LLM application framework
- `PyMuPDF>=1.22.5` - PDF processing
- `python-docx>=0.8.11` - Word document generation
- `pillow>=11.2.1` - Image processing

For a complete list, see [`requirements.txt`](app/requirements.txt).

## 🔧 Configuration

### Azure Services Setup

1. **Azure OpenAI**:
   - Create an Azure OpenAI resource
   - Deploy a GPT-4 model
   - Get your endpoint and API key

2. **Azure Document Intelligence**:
   - Create a Document Intelligence resource
   - Get your endpoint and API key

### Custom Logo Configuration

Place your default logo images in the `Data/logo/` directory. The application will use `logo.png` as the default logo for filtering.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Azure OpenAI Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Azure Document Intelligence Documentation](https://docs.microsoft.com/en-us/azure/applied-ai-services/form-recognizer/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)

## 📞 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/your-username/docutranslate-pro/issues) on GitHub.

---

**Made with ❤️ for intelligent document processing**
