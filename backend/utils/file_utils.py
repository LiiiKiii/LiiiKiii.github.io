#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File processing utilities.
"""

import os
import re
import zipfile
import shutil
import warnings
import sys
import io
import contextlib
from werkzeug.utils import secure_filename
from typing import List

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*FontBBox.*')
warnings.filterwarnings('ignore', message='.*gray non-stroke color.*')
warnings.filterwarnings('ignore', message='.*invalid float value.*')


def count_txt_files(folder_path: str) -> int:
    """
    Count TXT and PDF files in a folder, excluding TXT files generated from PDF conversion.
    This is used to validate uploads and should count only original files.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    count = 0
    if not os.path.isdir(folder_path):
        return 0
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".pdf"):
                count += 1
            elif fname.lower().endswith(".txt") and not fname.lower().endswith("_pdf.txt"):
                count += 1
    return count


def count_all_txt_files_after_conversion(folder_path: str) -> int:
    """
    Count all TXT files after conversion, including original TXT files and TXT files generated from PDFs.
    This is used by keyword extraction and later processing stages.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    count = 0
    if not os.path.isdir(folder_path):
        return 0
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".txt"):
                count += 1
    return count


def count_pdf_files(folder_path: str) -> int:
    """
    Count PDF files in a folder.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    count = 0
    if not os.path.isdir(folder_path):
        return 0
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".pdf"):
                count += 1
    return count


def get_txt_file_paths(folder_path: str) -> List[str]:
    """
    Get all TXT file paths in a folder, excluding TXT files generated from PDFs.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    paths = []
    if not os.path.isdir(folder_path):
        return paths
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".txt"):
                if not fname.lower().endswith("_pdf.txt"):
                    paths.append(os.path.join(root, fname))
    return sorted(paths)


def get_pdf_file_paths(folder_path: str) -> List[str]:
    """
    Get all PDF file paths in a folder.
    Exclude macOS resource-fork files (starting with ._) and other hidden files.
    """
    paths = []
    if not os.path.isdir(folder_path):
        return paths
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".pdf"):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a ZIP file."""
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting zip: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing illegal characters."""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename[:100]  # Limit length
    return filename


def create_output_zip(folder_path: str, zip_path: str) -> bool:
    """Package a folder as a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
        return True
    except Exception as e:
        print(f"Error creating zip: {e}")
        return False


def cleanup_user_data(folder_name: str, base_dir: str) -> dict:
    """Handle cleanup user data."""
    result = {
        "success": True,
        "deleted": {
            "uploads": False,
            "results": False,
            "outputs": False
        },
        "message": ""
    }
    
    uploads_dir = os.path.join(base_dir, "data", "uploads", folder_name)
    results_dir = os.path.join(base_dir, "data", "results", folder_name)
    outputs_dir = os.path.join(base_dir, "data", "outputs", folder_name)
    outputs_zip = os.path.join(base_dir, "data", "outputs", f"{folder_name}_recommended.zip")
    
    deleted_items = []
    
    if os.path.exists(uploads_dir):
        try:
            shutil.rmtree(uploads_dir, ignore_errors=True)
            result["deleted"]["uploads"] = True
            deleted_items.append("uploaded files")
        except Exception as e:
            print(f"Failed to delete uploaded files: {e}")
            result["success"] = False
    
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir, ignore_errors=True)
            result["deleted"]["results"] = True
            deleted_items.append("processed results")
        except Exception as e:
            print(f"Failed to delete processed results: {e}")
            result["success"] = False
    
    if os.path.exists(outputs_dir):
        try:
            shutil.rmtree(outputs_dir, ignore_errors=True)
            result["deleted"]["outputs"] = True
            deleted_items.append("output files")
        except Exception as e:
            print(f"Failed to delete output files: {e}")
            result["success"] = False
    
    if os.path.exists(outputs_zip):
        try:
            os.remove(outputs_zip)
            deleted_items.append("output ZIP file")
        except Exception as e:
            print(f"Failed to delete output ZIP file: {e}")
    
    if deleted_items:
        result["message"] = f"Cleaned up: {', '.join(deleted_items)}"
    else:
        result["message"] = "No files needed cleanup"
    
    return result


def convert_pdf_to_txt(pdf_path: str, output_txt_path: str = None) -> str:
    """Convert pdf to txt."""
    if not os.path.isfile(pdf_path):
        print(f"PDF file does not exist: {pdf_path}")
        return None
    
    if output_txt_path is None:
        base_name = os.path.splitext(pdf_path)[0]
        output_txt_path = f"{base_name}_pdf.txt"
    
    try:
        text_content = []
        
        if PDFPLUMBER_AVAILABLE:
            try:
                import logging
                pdfplumber_logger = logging.getLogger('pdfplumber')
                original_level = pdfplumber_logger.level
                pdfplumber_logger.setLevel(logging.ERROR)
                
                with contextlib.redirect_stderr(io.StringIO()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with pdfplumber.open(pdf_path) as pdf:
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text_content.append(page_text)
                
                pdfplumber_logger.setLevel(original_level)
            except Exception as e:
                text_content = []
        
        if not text_content and PYPDF2_AVAILABLE:
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(pdf_path, 'rb') as file:
                            pdf_reader = PyPDF2.PdfReader(file)
                            for page in pdf_reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text_content.append(page_text)
            except Exception as e:
                return None
        
        if not text_content:
            return None
        
        full_text = "\n\n".join(text_content)
        if len(full_text.strip()) < 50:
            return None
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        filename = os.path.basename(pdf_path)
        print(f"PDF conversion succeeded: {filename} -> {os.path.basename(output_txt_path)}")
        return output_txt_path
    
    except Exception as e:
        return None


def convert_all_pdfs_to_txt(folder_path: str) -> dict:
    """Convert all pdfs to txt."""
    pdf_files = get_pdf_file_paths(folder_path)
    success_count = 0
    failed_count = 0
    converted_files = []
    failed_files = []
    
    for pdf_path in pdf_files:
        txt_path = convert_pdf_to_txt(pdf_path)
        if txt_path:
            success_count += 1
            converted_files.append((pdf_path, txt_path))
        else:
            failed_count += 1
            failed_files.append(os.path.basename(pdf_path))
    
    if failed_files:
        print(f"PDF files that failed to convert ({len(failed_files)} total; possibly corrupted or unsupported):")
        for failed_file in failed_files[:5]:  # Show only the first 5 entries
            print(f"  - {failed_file}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more files failed to convert")
    
    return {
        "success_count": success_count,
        "failed_count": failed_count,
        "converted_files": converted_files
    }
