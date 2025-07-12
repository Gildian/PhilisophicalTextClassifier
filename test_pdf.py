#!/usr/bin/env python3
"""
Test script for PDF functionality
"""

import PyPDF2
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num}: {e}")
                    continue
            
            return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def clean_extracted_text(text):
    """Clean and normalize extracted text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (basic patterns)
    text = re.sub(r'\n\d+\n', ' ', text)
    text = re.sub(r'\nPage \d+\n', ' ', text)
    
    # Remove very short lines that are likely formatting artifacts
    lines = text.split('\n')
    filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
    
    return ' '.join(filtered_lines)

def split_text_into_chunks(text, max_chunk_size=500):
    """Split text into manageable chunks for training."""
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
    return chunks

if __name__ == "__main__":
    print("PDF Processing Test")
    print("==================")
    print("PyPDF2 version:", PyPDF2.__version__)
    print("\nThis script can extract text from philosophical PDFs and prepare them for training.")
    print("\nTo use with a PDF file:")
    print("  python test_pdf.py path/to/your/philosophical_text.pdf")
    print("\nThe text will be extracted, cleaned, and split into training chunks.")
    
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if Path(pdf_path).exists():
            print(f"\nProcessing: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            if text:
                cleaned_text = clean_extracted_text(text)
                chunks = split_text_into_chunks(cleaned_text)
                
                print(f"Extracted {len(text)} characters")
                print(f"Cleaned to {len(cleaned_text)} characters")
                print(f"Created {len(chunks)} training chunks")
                
                if chunks:
                    print(f"\nFirst chunk preview:")
                    print(f"'{chunks[0][:200]}...'")
                    
                    print(f"\nAll chunks are ready for training!")
                    print("Each chunk should be labeled as either 'Continental' or 'Analytic' philosophy.")
            else:
                print("No text could be extracted from the PDF.")
        else:
            print(f"Error: File {pdf_path} does not exist.")
