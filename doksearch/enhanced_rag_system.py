#!/usr/bin/env python3
"""
Enhanced RAG System with Hybrid Ensemble Retriever
Supports incremental processing, file tracking, and vector database management.
Built for web UI integration with Streamlit.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import warnings
import shutil

import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PDFPlumberLoader 
from logging_config import setup_logging, get_logger

loggers = setup_logging(log_level=logging.INFO, log_file="advanced_chat.log")
logger = get_logger('advanced_chat')
# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Manages metadata for processed documents."""
    
    def __init__(self, metadata_path: str = "./vector_db/metadata.json"):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
                return {"processed_files": {}, "created_at": datetime.now().isoformat()}
        return {"processed_files": {}, "created_at": datetime.now().isoformat()}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def is_file_processed(self, file_path: str) -> bool:
        """Check if a file has been processed and hasn't changed."""
        file_path = str(Path(file_path).resolve())
        current_hash = self.get_file_hash(file_path)
        
        if file_path in self.metadata["processed_files"]:
            stored_hash = self.metadata["processed_files"][file_path].get("hash", "")
            return current_hash == stored_hash and current_hash != ""
        return False
    
    def mark_file_processed(self, file_path: str, chunk_count: int):
        """Mark a file as processed with its metadata."""
        file_path = str(Path(file_path).resolve())
        file_hash = self.get_file_hash(file_path)
        
        self.metadata["processed_files"][file_path] = {
            "hash": file_hash,
            "chunk_count": chunk_count,
            "processed_at": datetime.now().isoformat(),
            "filename": Path(file_path).name,
            "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        self._save_metadata()
    
    def get_processed_files(self) -> List[Dict]:
        """Get list of all processed files with metadata."""
        processed_files = []
        for file_path, metadata in self.metadata["processed_files"].items():
            file_info = {
                "filepath": file_path,
                "filename": metadata.get("filename", Path(file_path).name),
                "chunk_count": metadata.get("chunk_count", 0),
                "processed_at": metadata.get("processed_at", "Unknown"),
                "size": metadata.get("size", 0),
                "exists": os.path.exists(file_path)
            }
            processed_files.append(file_info)
        return processed_files
    
    def clear_metadata(self):
        """Clear all metadata."""
        self.metadata = {"processed_files": {}, "created_at": datetime.now().isoformat()}
        self._save_metadata()


class EnhancedPDFProcessor:
    """Enhanced PDF processor with incremental processing capabilities."""
    
    def __init__(self, metadata_manager: DocumentMetadata):
        self.metadata_manager = metadata_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    
    def process_single_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file into document chunks."""
        
        if self.metadata_manager.is_file_processed(pdf_path):
            logger.info(f"Skipping already processed file: {Path(pdf_path).name}")
            return []
        
        # Extract text from PDF efficiently
        try:
            loader = PDFPlumberLoader(pdf_path)
            docs = loader.load()
            if not docs:
                logger.warning(f"No content extracted from {pdf_path}")
                return []
            
            # Split documents directly (more efficient than text conversion)
            chunks = self.text_splitter.split_documents(docs)
            
            # Update metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "filename": Path(pdf_path).name,
                    "chunk_id": i,
                    "filepath": str(pdf_path)
                })
            
            # Mark file as processed
            self.metadata_manager.mark_file_processed(pdf_path, len(chunks))
            logger.info(f"✅ Processed {Path(pdf_path).name}: {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return []
    
    def process_pdf_folder(self, folder_path: str, force_reprocess: bool = False) -> List[Document]:
        """Process all PDFs in a folder with incremental processing."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return []
        
        all_documents = []
        processed_count = 0
        skipped_count = 0
        
        logger.info(f"Found {len(pdf_files)} PDF files...")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            if force_reprocess or not self.metadata_manager.is_file_processed(str(pdf_file)):
                documents = self.process_single_pdf(str(pdf_file))
                all_documents.extend(documents)
                if documents:
                    processed_count += 1
            else:
                skipped_count += 1
        
        logger.info(f"Processed: {processed_count} files, Skipped: {skipped_count} files, "
                   f"Total chunks: {len(all_documents)}")
        
        return all_documents


class EnhancedHybridRAGSystem:
    """Enhanced RAG system with incremental processing and database management."""
    
    def __init__(self, 
                 vector_db_path: str = "./vector_db",
                 embedding_model: str = "nomic-embed-text:latest",
                 llm_model: str = "gemma3:4b",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the enhanced RAG system.
        
        Args:
            vector_db_path: Path to store vector database and metadata
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama model name
            ollama_base_url: Ollama server URL
        """
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize metadata manager
        self.metadata_manager = DocumentMetadata(
            metadata_path=str(self.vector_db_path / "metadata.json")
        )
        
        # Initialize components
        self.pdf_processor = EnhancedPDFProcessor(self.metadata_manager)
        self.embeddings = None
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Documents storage
        self.documents = []
        self.is_initialized = False
        
        logger.info("Enhanced RAG System initialized")
    
    def setup_embeddings(self):
        """Setup the embedding model."""
        if self.embeddings is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model_name,
                base_url=self.ollama_base_url
            )
            logger.info(f"Ollama Embeddings loaded successfully: {self.embedding_model_name}")
            # self.embeddings = HuggingFaceEmbeddings(
            #     model_name=self.embedding_model_name,
            #     model_kwargs={'device': 'cpu'},
            #     encode_kwargs={'normalize_embeddings': True}
            # )
            # logger.info("Embeddings loaded successfully")
    
    def setup_llm(self):
        """Setup the Ollama LLM."""
        if self.llm is None:
            logger.info(f"Connecting to Ollama model: {self.llm_model_name}")
            try:
                self.llm = OllamaLLM(
                    model=self.llm_model_name,
                    base_url=self.ollama_base_url,
                    temperature=0.1
                )
                logger.info("LLM connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                raise
    
    def load_existing_vector_store(self) -> bool:
        """Load existing vector store if available."""
        vector_store_path = self.vector_db_path / "vector_store"
        
        if vector_store_path.exists():
            try:
                self.setup_embeddings()
                self.vector_store = FAISS.load_local(
                    str(vector_store_path), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Existing vector store loaded successfully")
                return True
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")
                return False
        return False
    
    def create_vector_store(self, documents: List[Document]):
        """Create or update FAISS vector store from documents."""
        if not documents:
            logger.warning("No new documents to add to vector store")
            return
        
        self.setup_embeddings()
        
        if self.vector_store is None:
            # Create new vector store
            logger.info("Creating new FAISS vector store...")
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        else:
            # Add documents to existing vector store
            logger.info(f"Adding {len(documents)} documents to existing vector store...")
            self.vector_store.add_documents(documents)
        
        logger.info("Vector store updated successfully")
    
    def create_bm25_retriever(self, documents: List[Document]):
        """Create BM25 retriever from all documents."""
        if not documents:
            logger.warning("No documents available for BM25 retriever")
            return
        
        logger.info("Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 4
        logger.info("BM25 retriever created successfully")
    
    def create_ensemble_retriever(self):
        """Create ensemble retriever combining BM25 and semantic search."""
        if not self.vector_store:
            raise ValueError("Vector store must be created first")
        
        logger.info("Creating ensemble retriever...")
        # Create semantic retriever from vector store
        semantic_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create ensemble retriever
        if self.bm25_retriever:
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, semantic_retriever],
                weights=[0.5, 0.5]
            )
        else:
            # Fallback to semantic only if no BM25 retriever
            self.ensemble_retriever = semantic_retriever
        
        logger.info("Ensemble retriever created successfully")
    
    def setup_qa_chain(self):
        """Setup the question-answering chain."""
        if not self.ensemble_retriever or not self.llm:
            raise ValueError("Retriever and LLM must be setup first")
        
        prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.ensemble_retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        logger.info("QA chain setup successfully")
    
    def get_all_processed_documents(self) -> List[Document]:
        """Get all documents from processed files for BM25 retriever."""
        all_documents = []
        processed_files = self.metadata_manager.get_processed_files()
        
        for file_info in processed_files:
            if file_info["exists"]:
                try:
                    # Re-process the file to get documents for BM25
                    documents = self.pdf_processor.process_single_pdf(file_info["filepath"])
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Error re-processing {file_info['filename']}: {e}")
        
        return all_documents
    
    def load_pdf_folder(self, folder_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Load and process PDFs from a folder with incremental processing."""
        logger.info(f"Loading PDFs from: {folder_path}")
        
        # Load existing vector store if available
        vector_store_loaded = self.load_existing_vector_store()
        
        # Process new/changed PDFs
        new_documents = self.pdf_processor.process_pdf_folder(folder_path, force_reprocess)
        
        # Create or update vector store with new documents
        if new_documents:
            self.create_vector_store(new_documents)
        
        # Get all documents for BM25 retriever (including previously processed)
        all_documents = self.get_all_processed_documents()
        
        if all_documents:
            self.create_bm25_retriever(all_documents)
        
        # Create ensemble retriever if vector store exists
        if self.vector_store:
            self.create_ensemble_retriever()
            
            # Setup LLM and QA chain
            if not self.llm:
                self.setup_llm()
            self.setup_qa_chain()
            
            self.is_initialized = True
            logger.info("RAG system is ready for questions!")
        
        # Save vector store
        self.save_vector_store()
        
        # Return processing summary
        processed_files = self.metadata_manager.get_processed_files()
        return {
            "total_files": len(processed_files),
            "new_files_processed": len(new_documents) > 0,
            "total_chunks": sum(f["chunk_count"] for f in processed_files),
            "processed_files": processed_files
        }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if not self.is_initialized or not self.qa_chain:
            raise ValueError("RAG system not initialized. Load PDF folder first.")
        
        logger.info(f"Processing question: {question}")
        
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": []
            }
            
            # Add source information
            for doc in result.get("source_documents", []):
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                response["sources"].append(source_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {e}",
                "sources": []
            }
    
    def save_vector_store(self):
        """Save the vector store to disk."""
        if self.vector_store:
            vector_store_path = self.vector_db_path / "vector_store"
            self.vector_store.save_local(str(vector_store_path))
            logger.info(f"Vector store saved to {vector_store_path}")
    
    def clear_vector_database(self):
        """Clear the entire vector database and metadata."""
        try:
            # Clear vector store
            vector_store_path = self.vector_db_path / "vector_store"
            if vector_store_path.exists():
                shutil.rmtree(vector_store_path)
            
            # Clear metadata
            self.metadata_manager.clear_metadata()
            
            # Reset system state
            self.vector_store = None
            self.bm25_retriever = None
            self.ensemble_retriever = None
            self.qa_chain = None
            self.documents = []
            self.is_initialized = False
            
            logger.info("Vector database and metadata cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        processed_files = self.metadata_manager.get_processed_files()
        
        stats = {
            "total_files": len(processed_files),
            "total_chunks": sum(f["chunk_count"] for f in processed_files),
            "database_size": 0,
            "last_updated": None,
            "processed_files": processed_files
        }
        
        # Calculate database size
        if self.vector_db_path.exists():
            for file_path in self.vector_db_path.rglob("*"):
                if file_path.is_file():
                    stats["database_size"] += file_path.stat().st_size
        
        # Get last updated time
        if processed_files:
            latest_time = max(f["processed_at"] for f in processed_files)
            stats["last_updated"] = latest_time
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """Check the health of the RAG system components."""
        health = {
            "ollama_connection": False,
            "embeddings_loaded": False,
            "vector_store_exists": False,
            "system_initialized": self.is_initialized
        }
        
        # Check Ollama connection
        try:
            if self.llm is None:
                self.setup_llm()
            # Try a simple test
            test_response = self.llm.invoke("test")
            health["ollama_connection"] = True
        except Exception:
            health["ollama_connection"] = False
        
        # Check embeddings
        try:
            if self.embeddings is None:
                self.setup_embeddings()
            health["embeddings_loaded"] = True
        except Exception:
            health["embeddings_loaded"] = False
        
        # Check vector store
        health["vector_store_exists"] = self.vector_store is not None
        
        return health 