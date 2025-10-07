#!/usr/bin/env python3
"""
Example Configuration File for PDF Keyword Search

Copy this file to config.py and modify the settings according to your needs.
"""

# =====================================================
# API Configuration
# =====================================================

# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
# IMPORTANT: Replace this with your actual API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# =====================================================
# Default Paths
# =====================================================

# Default directory containing PDF files to search
# Modify this path to point to your PDF collection
DEFAULT_PDF_DIRECTORY = "./pdf"

# Default output directory for search results
DEFAULT_OUTPUT_DIRECTORY = "./results"

# =====================================================
# Search Configuration
# =====================================================

# Number of context paragraphs to include around matches
MAX_CONTEXT_PARAGRAPHS = 2

# Whether search is case sensitive (False = case insensitive)
SEARCH_CASE_SENSITIVE = False

# =====================================================
# OCR Configuration
# =====================================================

# OCR mode selection:
# - "gemini": Use Gemini API for OCR (requires API calls)
# - "local": Use local OCR model (dots.ocr or EasyOCR)
# - "auto": Try local OCR first, fallback to Gemini
OCR_MODE = "auto"

# Deprecated: Use OCR_MODE instead
USE_GEMINI_OCR = False

# Minimum text length threshold to trigger OCR (characters)
# If a page has fewer characters than this, OCR will be used
# Lower values = more OCR usage, higher values = less OCR usage
OCR_TEXT_THRESHOLD = 50

# OCR image scaling factor (higher = better quality, slower processing)
# Values: 1.0 (normal), 2.0 (2x), 3.0 (3x)
OCR_IMAGE_SCALE = 2

# =====================================================
# Local OCR Model Configuration
# =====================================================

# Hugging Face model name for local OCR
# Default: "rednote-hilab/dots.ocr" (good for mixed language content)
# Alternative: "microsoft/trocr-base-printed" (for English text)
LOCAL_OCR_MODEL = "rednote-hilab/dots.ocr"

# Device selection for local OCR:
# - "auto": Automatically choose best available device
# - "cuda": Use GPU (if available)
# - "cpu": Use CPU only
LOCAL_OCR_DEVICE = "auto"

# Data type for model inference:
# - "bfloat16": Best for modern hardware (recommended)
# - "float16": Good balance of speed and memory
# - "float32": Highest precision, more memory usage
LOCAL_OCR_TORCH_DTYPE = "bfloat16"

# =====================================================
# Logging Configuration
# =====================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
# DEBUG: Very detailed logging
# INFO: General information (recommended)
# WARNING: Only warnings and errors
# ERROR: Only errors
LOG_LEVEL = "INFO"

# Log file name
LOG_FILE = "pdf_search.log"

# =====================================================
# Report Configuration
# =====================================================

# Maximum number of matches to display in detailed results
# Set to -1 for unlimited
MAX_MATCHES_DISPLAY = 1000

# Whether to include paragraph context in results
INCLUDE_PARAGRAPH_CONTEXT = True

# Maximum paragraph length in report (characters)
# Longer paragraphs will be truncated
MAX_PARAGRAPH_LENGTH = 2000

# =====================================================
# Performance Configuration
# =====================================================

# Maximum number of PDF files to process in one batch
# Lower values use less memory but may be slower
MAX_BATCH_SIZE = 10

# Timeout for API requests (seconds)
API_TIMEOUT = 60

# Maximum retries for failed API requests
MAX_API_RETRIES = 3

# =====================================================
# Advanced Settings
# =====================================================

# Enable experimental features (use with caution)
ENABLE_EXPERIMENTAL_FEATURES = False

# Cache OCR results to avoid re-processing
ENABLE_OCR_CACHE = True

# Cache directory for OCR results
OCR_CACHE_DIR = ".cache/ocr"
