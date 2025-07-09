import os
import PyPDF2
from docx import Document
from typing import List, Dict, Tuple
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, file_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers and metadata."""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        # Split page text into chunks
                        page_chunks = self._chunk_text(text, page_num, file_path)
                        chunks.extend(page_chunks)
                        
            logger.info(f"Processed PDF: {file_path} - {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def extract_text_from_docx(self, file_path: str) -> List[Dict]:
        """Extract text from DOCX file with paragraph numbers."""
        chunks = []
        try:
            doc = Document(file_path)
            full_text = ""
            
            for para_num, paragraph in enumerate(doc.paragraphs, 1):
                full_text += paragraph.text + "\n"
            
            if full_text.strip():
                # Split document text into chunks
                chunks = self._chunk_text(full_text, 1, file_path)
                
            logger.info(f"Processed DOCX: {file_path} - {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
            return []
    
    def _chunk_text(self, text: str, page_num: int, file_path: str) -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            # Single chunk for short text
            chunk_text = " ".join(words)
            chunks.append({
                'text': chunk_text,
                'page': page_num,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'folder': os.path.dirname(file_path),
                'chunk_id': f"{os.path.basename(file_path)}_p{page_num}_chunk1"
            })
        else:
            # Multiple chunks with overlap
            chunk_id = 1
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunks.append({
                    'text': chunk_text,
                    'page': page_num,
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'folder': os.path.dirname(file_path),
                    'chunk_id': f"{os.path.basename(file_path)}_p{page_num}_chunk{chunk_id}"
                })
                chunk_id += 1
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict]:
        """Process a document based on its file extension."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension in ['.doc', '.docx']:
            return self.extract_text_from_docx(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return []
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all supported documents in a directory."""
        all_chunks = []
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return all_chunks
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                
                if file_extension in self.supported_extensions:
                    logger.info(f"Processing: {file_path}")
                    chunks = self.process_document(file_path)
                    all_chunks.extend(chunks)
        
        logger.info(f"Total chunks processed: {len(all_chunks)}")
        return all_chunks 