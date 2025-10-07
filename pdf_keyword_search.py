#!/usr/bin/env python3
"""
PDF Keyword Search with Term Expansion

Features:
1. Keyword term expansion using Google Gemini API
2. Search keywords and their expanded terms in PDF documents
3. Extract matching paragraphs, page numbers, and file names
4. Generate detailed Markdown search reports

Author: AI Assistant
Date: 2024
"""

import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import time
import random

# Third-party library imports
try:
    import requests
    import PyPDF2
    import fitz  # PyMuPDF for better PDF processing
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please run: pip install requests PyPDF2 PyMuPDF")
    exit(1)


class PDFKeywordSearcher:
    """PDF keyword searcher class"""
    
    def __init__(self, api_key: str, pdf_directory: str, output_directory: str = None):
        """
        Initialize the searcher
        
        Args:
            api_key: Google Gemini API key
            pdf_directory: PDF files directory
            output_directory: Output directory, defaults to current directory
        """
        self.api_key = api_key
        self.pdf_directory = Path(pdf_directory)
        self.output_directory = Path(output_directory) if output_directory else Path(".")
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # OCR related attributes
        self.local_ocr_model = None
        self.force_ocr = False
        self.disable_ocr = False
        
        # Initialize local OCR model
        self._init_local_ocr()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('pdf_search.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_local_ocr(self):
        """Initialize local OCR model"""
        try:
            from config import LOCAL_OCR_MODEL, LOCAL_OCR_DEVICE, LOCAL_OCR_TORCH_DTYPE
            
            # Try to initialize local OCR model
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                import torch
                
                self.logger.info(f"Loading local OCR model: {LOCAL_OCR_MODEL}")
                
                # Load model and processor
                self.local_ocr_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    LOCAL_OCR_MODEL,
                    torch_dtype=getattr(torch, LOCAL_OCR_TORCH_DTYPE, torch.bfloat16),
                    device_map=LOCAL_OCR_DEVICE
                )
                self.local_ocr_processor = AutoProcessor.from_pretrained(LOCAL_OCR_MODEL)
                
                self.logger.info("Local OCR model loaded successfully")
                
            except Exception as e:
                self.logger.warning(f"Failed to load local OCR model: {e}")
                self.local_ocr_model = None
                
        except ImportError:
            self.logger.info("Config file not found, using default settings")
            self.local_ocr_model = None
    
    def expand_keywords(self, keywords: str) -> List[str]:
        """
        Expand keywords using Gemini API
        
        Args:
            keywords: Original keywords (comma-separated)
            
        Returns:
            List of expanded keywords
        """
        try:
            self.logger.info(f"Starting keyword expansion: {keywords}")
            
            # Create expansion prompt
            prompt = f"""
Please expand the following keywords with semantic and root-related terms in English.

Include the following types of expansions:
1. Synonyms and near-synonyms
2. Related terms (semantically related)
3. Root-related words
4. Compound words and phrases
5. Technical terms
6. Different forms (verbs, nouns, adjectives)

Please output in a simple list format, one term per line, without explanations.
Format:
- word1
- word2
- phrase example
...

Keywords: {keywords}
"""
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.gemini_url}?key={self.api_key}",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            expanded_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Parse expansion results
            expanded_keywords = []
            for line in expanded_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    # Remove list markers
                    keyword = line.lstrip('- •').strip()
                    if keyword:
                        expanded_keywords.append(keyword)
            
            # Add original keywords
            original_keywords = [k.strip() for k in keywords.split(',')]
            all_keywords = original_keywords + expanded_keywords
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for kw in all_keywords:
                if kw.lower() not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw.lower())
            
            self.logger.info(f"Successfully expanded to {len(unique_keywords)} keywords")
            return unique_keywords
            
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            # Return original keywords if expansion fails
            return [k.strip() for k in keywords.split(',')]
    
    def extract_pdf_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF with OCR fallback

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries containing page text and metadata
        """
        pages_text = []
        doc = None

        try:
            self.logger.info(f"Extracting PDF text: {pdf_path.name}")

            # Use PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            for page_num in range(total_pages):
                try:
                    page = doc[page_num]

                    # First try direct text extraction
                    direct_text = page.get_text()

                    # Import OCR configuration
                    from config import OCR_MODE, OCR_TEXT_THRESHOLD, OCR_IMAGE_SCALE

                    # Determine if OCR is needed
                    should_use_ocr = False
                    ocr_method = OCR_MODE  # Default from config

                    if self.disable_ocr:
                        should_use_ocr = False
                    elif self.force_ocr:
                        should_use_ocr = True
                        self.logger.info(f"Page {page_num + 1} forced to use OCR")
                    elif len(direct_text.strip()) < OCR_TEXT_THRESHOLD:
                        should_use_ocr = True
                        self.logger.info(f"Page {page_num + 1} has little text ({len(direct_text.strip())} chars), using OCR")

                    if should_use_ocr:
                        # Convert page to image
                        pix = page.get_pixmap(matrix=fitz.Matrix(OCR_IMAGE_SCALE, OCR_IMAGE_SCALE))
                        img_data = pix.tobytes("png")

                        # Choose OCR method based on configuration
                        ocr_text = ""
                        if ocr_method == "local":
                            # Try local OCR first, fallback to EasyOCR
                            if self.local_ocr_model is not None:
                                ocr_text = self._local_ocr(img_data, page_num + 1)
                            if not ocr_text:
                                ocr_text = self._easyocr_extract(img_data, page_num + 1)
                        elif ocr_method == "gemini":
                            ocr_text = self._gemini_ocr(img_data, page_num + 1)
                        elif ocr_method == "auto":
                            # Auto mode: try local OCR (dots.ocr or EasyOCR), fallback to Gemini
                            if self.local_ocr_model is not None:
                                ocr_text = self._local_ocr(img_data, page_num + 1)
                            if not ocr_text:
                                ocr_text = self._easyocr_extract(img_data, page_num + 1)
                            if not ocr_text:
                                ocr_text = self._gemini_ocr(img_data, page_num + 1)

                        text = ocr_text if ocr_text else direct_text
                    else:
                        text = direct_text

                    # Split text into paragraphs
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

                    for para_idx, paragraph in enumerate(paragraphs):
                        if paragraph:
                            pages_text.append({
                                'page_number': page_num + 1,
                                'paragraph_index': para_idx,
                                'text': paragraph,
                                'file_name': pdf_path.name
                            })

                except Exception as page_error:
                    self.logger.warning(f"Error processing page {page_num + 1} of {pdf_path.name}: {page_error}")
                    continue

            self.logger.info(f"Successfully extracted {total_pages} pages of text from {pdf_path.name}")

        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            # Fallback to traditional method
            pages_text = self._fallback_text_extraction(pdf_path)

        finally:
            # Ensure document is always closed
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass

        return pages_text
    
    def _gemini_ocr(self, image_data: bytes, page_num: int) -> str:
        """
        OCR text recognition using Gemini API
        
        Args:
            image_data: Image data (PNG format)
            page_num: Page number
            
        Returns:
            Recognized text
        """
        max_retries = 3
        base_delay = 1  # Base delay 1 second
        
        for attempt in range(max_retries):
            try:
                import base64
                
                # Add delay to avoid API rate limiting
                if attempt > 0:
                    delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                    self.logger.info(f"Page {page_num} attempt {attempt+1}, waiting {delay:.1f}s...")
                    time.sleep(delay)
                
                # Convert image data to base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": "Please extract all text content from this image, maintaining original formatting and paragraph structure. If it's a table, please maintain table structure. Please only return text content without any explanations."
                                },
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": image_b64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 8192
                    }
                }
                
                headers = {
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    f"{self.gemini_url}?key={self.api_key}",
                    json=payload,
                    headers=headers,
                    timeout=60  # OCR may take longer
                )
                response.raise_for_status()
                
                result = response.json()
                ocr_text = result['candidates'][0]['content']['parts'][0]['text']
                
                self.logger.info(f"Page {page_num} OCR successful, extracted text length: {len(ocr_text)}")
                return ocr_text
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    self.logger.warning(f"Page {page_num} Gemini API rate limit, attempt {attempt+1}")
                    if attempt == max_retries - 1:
                        self.logger.error(f"Page {page_num} Gemini OCR reached max retries, skipping")
                        return ""
                    continue
                else:
                    self.logger.error(f"Page {page_num} Gemini OCR HTTP error: {e}")
                    return ""
            except Exception as e:
                self.logger.error(f"Page {page_num} Gemini OCR failed: {e}")
                if attempt == max_retries - 1:
                    return ""
                continue
        
        return ""
    
    def _local_ocr(self, image_data: bytes, page_num: int) -> str:
        """
        Text recognition using local OCR model
        
        Args:
            image_data: Image data (PNG format)
            page_num: Page number
            
        Returns:
            Recognized text
        """
        try:
            if self.local_ocr_model is None:
                return ""
            
            from PIL import Image
            import io
            import torch
            from qwen_vl_utils import process_vision_info
            
            # Convert bytes to PIL image
            image = Image.open(io.BytesIO(image_data))
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "Extract all text from this image, maintaining formatting."},
                    ],
                }
            ]
            
            # Apply chat template
            text = self.local_ocr_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.local_ocr_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.local_ocr_model.device)
            
            # Generate
            generated_ids = self.local_ocr_model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.local_ocr_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            self.logger.info(f"Page {page_num} local OCR successful, extracted text length: {len(output_text)}")
            return output_text
            
        except Exception as e:
            self.logger.error(f"Page {page_num} local OCR failed: {e}")
            return ""
    
    def _easyocr_extract(self, image_data: bytes, page_num: int) -> str:
        """
        Text recognition using EasyOCR
        
        Args:
            image_data: Image data (PNG format)
            page_num: Page number
            
        Returns:
            Recognized text
        """
        try:
            import easyocr
            import numpy as np
            from PIL import Image
            import io
            
            # Initialize EasyOCR (if not already initialized)
            if not hasattr(self, '_easyocr_reader'):
                self.logger.info("Initializing EasyOCR...")
                self._easyocr_reader = easyocr.Reader(['ch_tra', 'en'])  # Fixed language configuration
            
            # Convert bytes data to PIL image, then to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            # Use EasyOCR to recognize text
            results = self._easyocr_reader.readtext(image_array)
            
            # Extract text content
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Only keep text with high confidence
                    extracted_text.append(text)
            
            ocr_text = '\n'.join(extracted_text)
            self.logger.info(f"Page {page_num} EasyOCR successful, extracted text length: {len(ocr_text)}")
            return ocr_text
            
        except Exception as e:
            self.logger.error(f"Page {page_num} EasyOCR failed: {e}")
            return ""
    
    def _fallback_text_extraction(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Traditional text extraction method as fallback
        
        Args:
            pdf_path: PDF file path
            
        Returns:
            List of text data
        """
        pages_text = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            # Split into paragraphs
                            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                            
                            for para_idx, paragraph in enumerate(paragraphs):
                                if paragraph:
                                    pages_text.append({
                                        'page_number': page_num + 1,
                                        'paragraph_index': para_idx,
                                        'text': paragraph,
                                        'file_name': pdf_path.name
                                    })
                    except Exception as e:
                        self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Fallback extraction failed for {pdf_path.name}: {e}")
        
        return pages_text
    
    def search_keywords_in_text(self, text_data: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Search keywords in extracted text

        Args:
            text_data: Extracted text data
            keywords: List of keywords to search

        Returns:
            List of matches
        """
        matches = []

        for text_item in text_data:
            text = text_item['text']
            found_keywords = []

            # Search each keyword
            for keyword in keywords:
                # Use word boundary \b to match whole words only
                # Case-insensitive search
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                if pattern.search(text):
                    found_keywords.append(keyword)

            if found_keywords:
                matches.append({
                    'file_name': text_item['file_name'],
                    'page_number': text_item['page_number'],
                    'paragraph': text,
                    'keywords': found_keywords
                })

        return matches
    
    def generate_markdown_report(self, keywords: str, expanded_keywords: List[str], 
                                 all_matches: List[Dict[str, Any]]) -> str:
        """
        Generate Markdown search report
        
        Args:
            keywords: Original keywords
            expanded_keywords: Expanded keywords list
            all_matches: All search matches
            
        Returns:
            Markdown report content
        """
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count statistics
        total_matches = len(all_matches)
        unique_keywords = set()
        unique_files = set()
        
        for match in all_matches:
            unique_keywords.update(match['keywords'])
            unique_files.add(match['file_name'])
        
        # Start building report
        report = f"# Keyword Search Results Report\n\n"
        report += f"**Search Keywords:** {keywords}\n"
        report += f"**Generated Time:** {timestamp}\n\n"
        
        # Search statistics
        report += "## Search Statistics\n\n"
        report += f"- Total Matches: {total_matches}\n"
        report += f"- Number of Keywords: {len(unique_keywords)}\n"
        report += f"- Files Involved: {len(unique_files)}\n\n"
        
        # Keyword expansion results
        report += "## Keyword Expansion Results\n\n"
        original_keywords = [k.strip() for k in keywords.split(',')]
        
        for kw in original_keywords:
            # Find expanded terms for this keyword
            expanded_for_kw = [ek for ek in expanded_keywords if ek.lower() != kw.lower()]
            if expanded_for_kw:
                report += f"**{kw}**: {', '.join(expanded_for_kw[:10])}\n\n"  # Show first 10
            else:
                report += f"**{kw}**: No expanded terms\n\n"
        
        # Detailed search results (table format)
        if all_matches:
            report += "## Detailed Search Results (Table Format)\n\n"
            report += "| Document Name | Page | Paragraph Content | Matched Keywords |\n"
            report += "|---------------|------|-------------------|------------------|\n"
            
            # Sort matches by file name and page number
            sorted_matches = sorted(all_matches, key=lambda x: (x['file_name'], x['page_number']))
            
            # Group by paragraph to avoid duplicates
            seen_paragraphs = set()
            unique_matches = []
            
            for match in sorted_matches:
                para_key = (match['file_name'], match['page_number'], match['paragraph'][:100])
                if para_key not in seen_paragraphs:
                    seen_paragraphs.add(para_key)
                    unique_matches.append(match)
            
            # Sort paragraphs by file and page
            sorted_paragraphs = sorted(unique_matches, key=lambda x: (x['file_name'], x['page_number']))
            
            current_file = ""
            for para_data in sorted_paragraphs:
                # Don't repeat document name for consecutive rows from same file
                if para_data['file_name'] != current_file:
                    file_display = para_data['file_name'].replace('|', '\\|')
                    current_file = para_data['file_name']
                else:
                    file_display = ""
                
                # Clean paragraph content for table display
                paragraph = para_data['paragraph'].replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')

                # Bold matched keywords within paragraph content (use word boundary)
                for keyword in para_data['keywords']:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    paragraph = pattern.sub(lambda m: f"**{m.group()}**", paragraph)
                
                # Sort and deduplicate matched keywords
                unique_keywords = sorted(list(set(para_data['keywords'])), key=str.lower)
                keywords_display = ", ".join([f"**{kw}**" for kw in unique_keywords])
                
                report += f"| {file_display} | {para_data['page_number']} | {paragraph} | {keywords_display} |\n"
        
        # Generation notes
        report += "\n## Generation Notes\n\n"
        report += "This report is automatically generated by a Python script, based on Google Gemini API for keyword term expansion and then searching for matches in PDF documents.\n\n"
        report += "The search scope includes original keywords and their expanded terms (roots, semantically related words, synonyms, etc.).\n\n"
        report += "The report contains specific locations of matched terms (file name, page number, paragraph) and contextual information.\n"
        
        return report
    
    def search_pdfs(self, keywords: str) -> str:
        """
        Main search function
        
        Args:
            keywords: Keywords to search (comma-separated)
            
        Returns:
            Path to generated report
        """
        self.logger.info("Starting PDF keyword search task")
        
        # Step 1: Expand keywords
        expanded_keywords = self.expand_keywords(keywords)
        
        # Step 2: Find PDF files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_directory}")
        
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Step 3: Process each PDF file
        all_matches = []
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing file: {pdf_file.name}")
            
            # Extract text
            pages_text = self.extract_pdf_text(pdf_file)
            
            # Search keywords
            self.logger.info(f"Searching keywords in file {pdf_file.name}")
            matches = self.search_keywords_in_text(pages_text, expanded_keywords)
            
            all_matches.extend(matches)
            self.logger.info(f"Found {len(matches)} matches in {pdf_file.name}")
        
        # Step 4: Generate report
        self.logger.info("Generating Markdown report")
        report_content = self.generate_markdown_report(keywords, expanded_keywords, all_matches)
        
        # Step 5: Save report
        timestamp = datetime.now().strftime("%Y-%m-%d")
        report_filename = f"Keyword_Search_Results_{timestamp}.md"
        report_path = self.output_directory / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Search completed! Report saved to: {report_filename}")
        self.logger.info(f"Total matches found: {len(all_matches)}")
        
        return str(report_path)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PDF Keyword Search with Term Expansion")
    parser.add_argument('-k', '--keywords', required=True, 
                        help='Keywords to search (comma-separated)')
    parser.add_argument('-p', '--pdf-dir', required=True, 
                        help='Directory containing PDF files')
    parser.add_argument('-o', '--output-dir', default='.', 
                        help='Output directory for results (default: current directory)')
    parser.add_argument('-a', '--api-key', 
                        help='Google Gemini API key (can also use config.py)')
    parser.add_argument('--use-ocr', action='store_true', 
                        help='Force use OCR for all pages')
    parser.add_argument('--no-ocr', action='store_true', 
                        help='Disable OCR completely')
    parser.add_argument('--ocr-mode', choices=['local', 'gemini', 'auto'], default='auto',
                        help='OCR mode selection (default: auto)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        try:
            from config import GEMINI_API_KEY
            api_key = GEMINI_API_KEY
        except ImportError:
            print("[ERROR] No API key provided. Please use --api-key parameter or create config.py file")
            return

    if not api_key:
        print("[ERROR] API key is empty")
        return

    # Display OCR mode
    print(f"[INFO] OCR mode set to: {args.ocr_mode}")
    
    try:
        # Create searcher
        searcher = PDFKeywordSearcher(api_key, args.pdf_dir, args.output_dir)
        
        # Set OCR options
        if args.use_ocr:
            searcher.force_ocr = True
        if args.no_ocr:
            searcher.disable_ocr = True
        
        # Perform search
        report_path = searcher.search_pdfs(args.keywords)

        print(f"\n[SUCCESS] Search completed!")
        print(f"[INFO] Report file: {report_path}")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
