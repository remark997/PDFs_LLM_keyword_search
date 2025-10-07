# PDF Keyword Search with AI-Powered Term Expansion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/release/remark997/PDFs_LLM_keyword_search.svg)](https://github.com/remark997/PDFs_LLM_keyword_search/releases/)

A powerful Python tool for searching keywords in PDF documents with automatic term expansion using Google Gemini API and advanced OCR capabilities.

---

## 🌟 Features

- **🔍 Intelligent Keyword Expansion**: Uses Google Gemini API to expand search terms with synonyms, related words, and semantic variations
- **📄 Advanced PDF Processing**: Supports both direct text extraction and OCR for scanned documents
- **🎯 Precise Word Matching**: Uses word boundary detection to avoid partial matches (e.g., "tree" won't match "street")
- **🤖 Multiple OCR Options**:
  - Local OCR using Hugging Face models (dots.ocr)
  - EasyOCR for reliable text recognition
  - Google Gemini API for cloud-based OCR
  - Automatic fallback between methods
- **📊 Comprehensive Reports**: Generates detailed Markdown reports with search statistics and highlighted matches
- **💻 Windows Support**: Full compatibility with Windows systems with detailed setup guides
- **⚙️ Configurable**: Extensive configuration options for different use cases

---

## 🆕 What's New in Latest Version

### 🐛 Bug Fixes
- **Fixed word boundary matching**: Keywords now match only complete words (e.g., "tree" no longer matches "street")
- **Fixed "document closed" error**: Improved PDF document handling with proper resource cleanup
- **Fixed Windows console encoding**: Replaced emoji characters with text markers for Windows compatibility

### 📚 New Documentation
- Added comprehensive Windows installation guide (Chinese)
- Added comprehensive Windows installation guide (English)
- Improved error messages and logging

### 🔧 Improvements
- Enhanced PDF page-level error handling
- Better resource management with try-finally blocks
- More robust OCR fallback mechanisms

See [CHANGELOG.md](CHANGELOG.md) for full version history.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/remark997/PDFs_LLM_keyword_search.git
cd PDFs_LLM_keyword_search

# Install dependencies
pip install requests PyPDF2 PyMuPDF

# (Optional) Install OCR dependencies
pip install easyocr torch torchvision Pillow numpy
```

### Configuration

1. Copy `config.example.py` to `config.py`:
```bash
cp config.example.py config.py
```

2. Edit `config.py` and add your Google Gemini API key:
```python
GEMINI_API_KEY = "your_actual_api_key_here"
```

3. Get your API key from: https://aistudio.google.com/app/apikey

### Basic Usage

```bash
# Basic search
python pdf_keyword_search.py -k "deforestation" -p "./pdf" -o "./results"

# Search multiple keywords
python pdf_keyword_search.py -k "deforestation,forest,conservation" -p "./pdf" -o "./results"

# Force OCR for scanned PDFs
python pdf_keyword_search.py -k "biodiversity" -p "./pdf" -o "./results" --use-ocr

# Specify OCR mode
python pdf_keyword_search.py -k "sustainability" -p "./pdf" --ocr-mode gemini
```

---

## 📖 Detailed Usage

### Command Line Options

```bash
python pdf_keyword_search.py [OPTIONS]

Required Arguments:
  -k, --keywords KEYWORDS    Keywords to search (comma-separated)
  -p, --pdf-dir PDF_DIR     Directory containing PDF files

Optional Arguments:
  -o, --output-dir DIR      Output directory (default: current directory)
  -a, --api-key KEY        Google Gemini API key (or use config.py)
  --use-ocr                Force OCR for all pages
  --no-ocr                 Disable OCR completely
  --ocr-mode MODE          OCR mode: local, gemini, auto (default: auto)
```

### Examples

#### Environmental Research
```bash
python pdf_keyword_search.py \
  -k "deforestation,reforestation,forest management,biodiversity" \
  -p "./environmental_reports" \
  -o "./results/environmental"
```

#### Business Intelligence
```bash
python pdf_keyword_search.py \
  -k "sustainability,corporate responsibility,ESG" \
  -p "./annual_reports" \
  --ocr-mode auto
```

#### Technical Documentation
```bash
python pdf_keyword_search.py \
  -k "machine learning,artificial intelligence,automation" \
  -p "./technical_docs" \
  --use-ocr
```

### Python API Usage

```python
from pdf_keyword_search import PDFKeywordSearcher

# Initialize searcher
searcher = PDFKeywordSearcher(
    api_key="your_api_key",
    pdf_directory="./pdf",
    output_directory="./results"
)

# Perform search
report_path = searcher.search_pdfs("forest,conservation,biodiversity")
print(f"Report saved to: {report_path}")
```

---

## 🔧 Configuration

### config.py Settings

```python
# API Configuration
GEMINI_API_KEY = "your_api_key_here"

# Default Paths
DEFAULT_PDF_DIRECTORY = "./pdf"
DEFAULT_OUTPUT_DIRECTORY = "./results"

# OCR Configuration
OCR_MODE = "auto"  # "local", "gemini", "auto"
OCR_TEXT_THRESHOLD = 50  # Minimum chars to trigger OCR
OCR_IMAGE_SCALE = 2  # Image scaling for better OCR

# Local OCR Model
LOCAL_OCR_MODEL = "rednote-hilab/dots.ocr"
LOCAL_OCR_DEVICE = "auto"
LOCAL_OCR_TORCH_DTYPE = "bfloat16"
```

### OCR Modes

- **`local`**: Uses local Hugging Face models (dots.ocr) with EasyOCR fallback
- **`gemini`**: Uses Google Gemini API for OCR
- **`auto`**: Tries local OCR first, falls back to Gemini if needed (recommended)

---

## 📊 Output Format

The tool generates comprehensive Markdown reports with:

### Search Statistics
- Total matches found
- Number of unique keywords
- Files processed

### Keyword Expansion Results
- Original keywords and their expanded terms
- Semantic variations and synonyms

### Detailed Results Table
- Document name
- Page number
- Paragraph content with **highlighted** matches
- Matched keywords (sorted alphabetically)

### Example Output

```markdown
# Keyword Search Results Report

**Search Keywords:** forest,conservation
**Generated Time:** 2024-08-14 10:30:45

## Search Statistics
- Total Matches: 25
- Number of Keywords: 8
- Files Involved: 3

## Detailed Search Results (Table Format)

| Document Name | Page | Paragraph Content | Matched Keywords |
|---------------|------|-------------------|------------------|
| Annual_Report_2023.pdf | 48 | The **forest** **conservation** program protects local ecosystems... | **conservation**, **forest** |
```

---

## 🛠️ Development

### Project Structure

```
PDFs_LLM_keyword_search/
├── pdf_keyword_search.py         # Main application
├── config.py                     # Configuration settings
├── config.example.py             # Example configuration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── CHANGELOG.md                  # Version history
├── LICENSE                       # MIT License
├── WINDOWS_SETUP_GUIDE_CN.md    # Windows setup (Chinese)
├── WINDOWS_SETUP_GUIDE_EN.md    # Windows setup (English)
├── examples/                     # Usage examples
│   └── example_usage.py
├── pdf/                          # PDF files to search (create this)
└── results/                      # Search results output
```

### Dependencies

- **Core**: Python 3.8+, requests, PyPDF2, PyMuPDF
- **OCR**: torch, transformers, easyocr, Pillow
- **Optional**: qwen-vl-utils (for advanced local OCR)

### Running Tests

```bash
# Run example usage
python examples/example_usage.py

# Test with sample data
python pdf_keyword_search.py -k "test,example" -p "./examples" --no-ocr
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. API Rate Limits
**Error**: `429 Client Error: Too Many Requests`

**Solution**: The tool includes automatic retry logic. For heavy usage, consider using local OCR mode:
```bash
python pdf_keyword_search.py -k "keywords" -p "./pdf" --ocr-mode local
```

#### 2. OCR Model Loading Issues
**Error**: `Flash Attention not available` or CUDA errors

**Solution**: The tool automatically falls back to EasyOCR. Alternatively:
```python
# In config.py
OCR_MODE = "gemini"  # or "auto" for automatic fallback
```

#### 3. No PDF Files Found
**Error**: `No PDF files found`

**Solution**: Ensure PDF files are in the specified directory and have `.pdf` extension.

#### 4. Keywords Matching Partial Words
**Solution**: This has been fixed in the latest version. Keywords now use word boundary matching to ensure exact word matches only.

#### 5. "Document Closed" Error
**Solution**: This bug has been fixed. If you still encounter this, please update to the latest version.

### Debug Mode

Enable detailed logging:
```python
# In config.py
LOG_LEVEL = "DEBUG"
```

Check the log file: `pdf_search.log`

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/PDFs_LLM_keyword_search.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python examples/example_usage.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Gemini API for keyword expansion and OCR
- Hugging Face for local OCR models
- EasyOCR team for reliable OCR functionality
- PyMuPDF team for excellent PDF processing

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/remark997/PDFs_LLM_keyword_search/issues)
- **Documentation**: Check the guides in the repository
- **Examples**: See the `examples/` directory

---

## 📈 Roadmap

- [ ] Support for more document formats (Word, Excel)
- [ ] Web interface for easier usage
- [ ] Batch processing optimization
- [ ] Export results to multiple formats (JSON, CSV, Excel)
- [ ] Integration with cloud storage (Google Drive, Dropbox)

---

**Made with ❤️ for researchers, analysts, and anyone who needs to search through lots of PDFs efficiently.**

**Star ⭐ this repository if you find it helpful!**
