# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-10-07

### 🐛 Fixed

- **Word Boundary Matching**: Fixed critical bug where keywords would match partial words
  - Example: Searching for "tree" would incorrectly match "street", "trees", etc.
  - Solution: Added regex word boundary (`\b`) to ensure exact word matching
  - Affected functions: `search_keywords_in_text()`, `generate_markdown_report()`

- **PDF Document Closed Error**: Fixed "document closed" error during PDF processing
  - Issue: Document was being closed before accessing its properties
  - Solution: Store page count before processing, added proper try-finally blocks
  - Improved: Page-level error handling to prevent complete failure

- **Windows Console Encoding**: Fixed UnicodeEncodeError on Windows systems
  - Issue: Emoji characters (🔧, ✅, ❌) couldn't be displayed in Windows console
  - Solution: Replaced emoji with text markers ([INFO], [SUCCESS], [ERROR])
  - Impact: Full Windows compatibility restored

### ✨ Added

- **Improved Documentation**:
  - Added detailed usage examples
  - Added troubleshooting section
  - Added Python API usage examples

- **Git Configuration**:
  - Added `.gitignore` file with comprehensive exclusions
  - Excludes sensitive files (config.py, API keys)
  - Excludes temporary files (logs, cache, pycache)

### 🔧 Changed

- **Error Handling**: Enhanced PDF processing error handling
  - Each page now processed in isolated try-catch block
  - Failures on individual pages no longer crash entire process
  - Better error logging with page numbers and context

- **Resource Management**: Improved cleanup and resource handling
  - Document objects now properly closed in finally blocks
  - Prevents resource leaks on exceptions
  - Better memory management for large PDF batches

- **Logging**: Improved log messages for better debugging
  - More informative success messages include file names
  - Better context in error messages
  - Clearer progress indicators

### 📝 Documentation

- Updated README.md with latest features and bug fixes
- Added "What's New" section highlighting recent improvements
- Improved installation instructions
- Enhanced troubleshooting guide with solutions to common issues

---

## [1.0.0] - 2024-XX-XX

### ✨ Initial Release

- **Keyword Expansion**: Uses Google Gemini API to expand search terms
  - Generates synonyms, related terms, and semantic variations
  - Configurable expansion parameters

- **PDF Processing**:
  - Direct text extraction using PyMuPDF
  - OCR support for scanned documents
  - Multiple OCR backends (local, Gemini, EasyOCR)

- **Search Functionality**:
  - Case-insensitive keyword search
  - Multi-file batch processing
  - Context-aware paragraph extraction

- **OCR Support**:
  - Local OCR using Hugging Face models (dots.ocr)
  - Cloud OCR using Google Gemini API
  - EasyOCR fallback
  - Automatic mode selection

- **Report Generation**:
  - Markdown-formatted search results
  - Statistics and summaries
  - Highlighted keyword matches
  - Organized by document and page

- **Configuration**:
  - Extensive config.py options
  - OCR mode selection
  - Image scaling parameters
  - Text extraction thresholds

- **Command Line Interface**:
  - Flexible argument parsing
  - Multiple search modes
  - OCR control options

---

## Release Notes

### Version Numbering

We follow Semantic Versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Upgrade Instructions

#### From 1.0.0 to 1.1.0

1. **Update your code**:
   ```bash
   git pull origin main
   ```

2. **No breaking changes**: All existing code should work without modification

3. **Note**: Keywords now match exact words only (word boundary matching)
   - This is a behavioral change but improves accuracy
   - Previous partial matches will no longer appear in results

### Known Issues

- Large PDF files (>500 pages) may take significant time with OCR enabled
- Gemini API rate limits may cause delays during heavy usage
- Local OCR models require significant disk space (2-3GB)

### Future Plans

See the [Roadmap](README.md#-roadmap) section in README.md for upcoming features.

---

## Support

For questions or issues:
- Open an issue: https://github.com/remark997/PDFs_LLM_keyword_search/issues
- Check documentation: [README.md](README.md)

---

**Thank you for using PDF Keyword Search!** ⭐
