# from pdfminer.high_level import extract_text

# def extract_text_from_file(uploaded_file):
#     if uploaded_file.name.endswith('.pdf'):
#         # Save uploaded file to disk temporarily
#         with open("temp_uploaded.pdf", "wb") as f:
#             f.write(uploaded_file.read())
#         return extract_text("temp_uploaded.pdf")

#     elif uploaded_file.name.endswith('.txt'):
#         return uploaded_file.read().decode("utf-8")

#     else:
#         return "Unsupported file type."


import PyPDF2
from typing import Union
from transformers import pipeline
import re

def extract_text_from_file(file) -> str:
    """Extract text from PDF or TXT file"""
    if file.name.endswith('.pdf'):
        try:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages])
            return clean_text(text)
        except Exception as e:
            raise ValueError(f"PDF extraction error: {str(e)}")
    elif file.name.endswith('.txt'):
        return clean_text(file.read().decode('utf-8'))
    else:
        raise ValueError("Unsupported file format")

def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
    return text.strip()

def generate_summary(text: str, max_length: int = 150) -> str:
    """Generate a concise summary of the text"""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Split text into chunks if too long
    chunks = []
    if len(text.split()) > 1024:
        words = text.split()
        chunks = [' '.join(words[i:i+800]) for i in range(0, len(words), 800)]
    else:
        chunks = [text]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        summaries.append(summary)
    
    # Combine summaries if needed
    if len(summaries) > 1:
        combined = ' '.join(summaries)
        if len(combined.split()) > max_length:
            return summarizer(
                combined,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )[0]['summary_text']
        return combined
    return summaries[0]