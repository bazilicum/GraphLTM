"""
A helper class to manage PDF reading, text cleaning, and chunking.
"""

import re
from bidi.algorithm import get_display
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from flask_socketio import SocketIO  # For progress updates during PDF processing
from utils.logging_utils import setup_logging
from config import Config

#load config and setup logging
config = Config()
logger = setup_logging(config, __name__)

class PDFProcessor:
    """
    A helper class to manage PDF reading, text cleaning, and chunking.
    """

    @staticmethod
    def clean_text(page_text: str) -> str:
        """
        Cleans the extracted text by:
          - Replacing multiple spaces with a single space within paragraphs
          - Removing hyphenated line breaks within paragraphs
          - Preserving paragraph structure (double newlines)
          - Trimming whitespace
        """
        # Split into paragraphs first
        paragraphs = page_text.split('\n\n')
        cleaned_paragraphs = []
        
        for para in paragraphs:
            # Clean within paragraphs
            cleaned = re.sub(r"\s+", " ", para)
            cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
            cleaned_paragraphs.append(cleaned.strip())
        
        # Join paragraphs back with double newlines
        return '\n\n'.join(cleaned_paragraphs)

    @staticmethod
    def chunk_text(text: str, max_words: int = 1000, overlap_sentences: int = 1) -> list:
        """
        Splits a paragraph into chunks if it's too long, trying to maintain sentence boundaries.
        If the paragraph is short enough, returns it as a single chunk.
        Overlaps between chunks are handled by repeating the last `overlap_sentences` sentences.
        """
        # Split paragraph into sentences
        sentences = sent_tokenize(text)
        paragraph_word_count = sum(len(s.split()) for s in sentences)
        
        # If the paragraph fits within the word limit, return it as a single chunk
        if paragraph_word_count <= max_words:
            return [text]
        
        # If paragraph is too long, process it sentence by sentence
        chunks = []
        current_chunk = []
        current_word_count = 0
        overlap = []

        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # If adding the sentence exceeds max_words limit, start a new chunk
            if current_word_count + sentence_word_count > max_words:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                    current_chunk = overlap.copy()
                    current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

        # Add the final chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def is_sentence_complete(text: str) -> bool:
        """
        Checks if the text ends with . ! or ? to determine completeness.
        """
        text = text.strip()
        return bool(text) and text[-1] in {'.', '!', '?'}

    @staticmethod
    def is_uppercase(text: str, threshold: float = 0.8) -> bool:
        """
        Determines if a significant portion of the text is uppercase.
        
        Args:
            text: Text to check
            threshold: Minimum proportion of uppercase characters required
                       (e.g., 0.8 means at least 80% of characters must be uppercase)
        """
        if not text:
            return False
        uppercase_chars = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        if total_letters == 0:
            return False
        return (uppercase_chars / total_letters) >= threshold

    @staticmethod
    def pretty_print_filename(filename: str) -> str:
        """
        Converts a filename into a citation-friendly camel case format.
        """
        import os
        base, _ = os.path.splitext(filename)
        import re
        words = re.split(r'[_\- ]+', base.strip())
        return ''.join(word.capitalize() for word in words)

    @classmethod
    def chunk_pdf_text(
        cls,
        pdf_file_path: str,
        original_file_name: str,
        socketio_instance: SocketIO,
        socket_id: str,
        max_words: int = 1000,
        overlap_sentences: int = 1,
        min_sentences_per_page: int = 3,
        uppercase_threshold: float = 0.8
    ) -> list:
        """
        Reads a PDF, processes text page-by-page, and returns a list of chunk dictionaries.
        Each page can have multiple chunks, all sharing the same page number.
        """
        pdf_chunks = []
        residual_fragment = ""
        global_chunk_id = 0  # Track chunk_id across all pages

        with pdfplumber.open(pdf_file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_number, page in enumerate(pdf.pages):

                # Emit progress every 10 pages or the last page
                if (page_number + 1) % 10 == 0 or (page_number + 1) == total_pages:
                    progress = (page_number + 1) / total_pages * 100
                    socketio_instance.emit('processing_progress', {'progress': progress}, room=socket_id)
                    socketio_instance.emit(
                        'status',
                        {'message': f'Chunking page {page_number + 1} of {total_pages}...'},
                        room=socket_id
                    )

                text = page.extract_text()
                if text:
                    # Improve paragraph detection
                    # PDFs often use single newlines (\n) for line breaks within paragraphs
                    # Identify paragraphs by looking for patterns like:
                    # 1. Double newlines (\n\n)
                    # 2. Lines ending with punctuation (possible paragraph end)
                    # 3. Short lines followed by lines starting with capital letters
                    
                    lines = text.split("\n")
                    processed_lines = [get_display(line) for line in lines]
                    
                    # Identify potential paragraphs
                    paragraphs = []
                    current_paragraph = []
                    
                    for i, line in enumerate(processed_lines):
                        line = line.strip()
                        if not line:  # Empty line - definite paragraph break
                            if current_paragraph:
                                paragraphs.append(' '.join(current_paragraph))
                                current_paragraph = []
                            continue
                            
                        # Check if this line likely ends a paragraph
                        is_paragraph_end = False
                        
                        # If line ends with punctuation (.!?), it might be the end of a paragraph
                        if line and line[-1] in {'.', '!', '?'}:
                            # Check next line - if it starts with capital letter or is indented, likely a new paragraph
                            if i < len(processed_lines) - 1:
                                next_line = processed_lines[i+1].strip()
                                if next_line and (next_line[0].isupper() or next_line.startswith('  ')):
                                    is_paragraph_end = True
                        
                        current_paragraph.append(line)
                        
                        if is_paragraph_end:
                            paragraphs.append(' '.join(current_paragraph))
                            current_paragraph = []
                    
                    # Add the last paragraph if non-empty
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                    
                    # Join detected paragraphs with double newlines
                    text = '\n\n'.join(paragraphs)

                    # Append any leftover from the previous page
                    if residual_fragment:
                        text = residual_fragment + "\n\n" + text
                        residual_fragment = ""

                    # Clean text while preserving paragraph structure
                    cleaned_text = cls.clean_text(text)
                    
                    # Process each paragraph separately
                    paragraphs = cleaned_text.split('\n\n')
                    total_sentences = 0
                    
                    for para in paragraphs:
                        sentences = sent_tokenize(para)
                        total_sentences += len(sentences)
                        
                        # Check if this paragraph is predominantly uppercase
                        upper_check = cls.is_uppercase(para, threshold=uppercase_threshold)
                        
                        # Decide if it's a 'regular' paragraph or a header
                        is_regular_para = (len(sentences) >= min_sentences_per_page) and (not upper_check)
                        
                        if is_regular_para:
                            # If the last sentence is incomplete, keep it for the next round
                            if not cls.is_sentence_complete(para) and sentences:
                                residual_fragment = sentences.pop(-1)
                                para = ' '.join(sentences)
                            
                            # Let chunk_text handle the paragraph - it will return one chunk if short enough
                            chunks = cls.chunk_text(para, max_words, overlap_sentences)
                            
                            # Add all chunks from this paragraph
                            for chunk in chunks:
                                pdf_chunks.append({
                                    "pdf_name": cls.pretty_print_filename(original_file_name),
                                    "page": page_number + 1,
                                    "chunk_id": global_chunk_id,
                                    "text": chunk
                                })
                                global_chunk_id += 1
                        else:
                            # Possibly a header or title paragraph; clear any leftovers
                            residual_fragment = ""
                            # Add the header/title as a single chunk
                            pdf_chunks.append({
                                "pdf_name": cls.pretty_print_filename(original_file_name),
                                "page": page_number + 1,
                                "chunk_id": global_chunk_id,
                                "text": para
                            })
                            global_chunk_id += 1

            # Handle any leftover sentence after processing all pages
            if residual_fragment:
                # Let chunk_text handle the residual - it will return one chunk if short enough
                chunks = cls.chunk_text(residual_fragment, max_words, overlap_sentences)
                
                for chunk in chunks:
                    pdf_chunks.append({
                        "pdf_name": cls.pretty_print_filename(original_file_name),
                        "page": total_pages,
                        "chunk_id": global_chunk_id,
                        "text": chunk
                    })
                    global_chunk_id += 1

        return pdf_chunks
