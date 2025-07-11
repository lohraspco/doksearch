#!/usr/bin/env python3
"""
RAG System with Hybrid Ensemble Retriever
Combines BM25 and semantic search for enhanced document retrieval from PDFs.
Uses local Gemma3:4b via Ollama and nomic-embed-text embeddings.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
import fitz  # PyMuPDF
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and preprocessing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_pdf_folder(self, folder_path: str) -> List[Document]:
        """Process all PDFs in a folder and return chunked documents."""
        folder = Path(folder_path)
        pdf_files = list(folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return []
        
        documents = []
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            text = self.extract_text_from_pdf(str(pdf_file))
            if text.strip():
                # Create chunks
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": str(pdf_file),
                            "filename": pdf_file.name,
                            "chunk_id": i
                        }
                    )
                    documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks from {len(pdf_files)} PDFs")
        return documents


class HybridRAGSystem:
    """Main RAG system with hybrid ensemble retriever."""
    
    def __init__(self, 
                 embedding_model: str = "nomic-embed-text:latest",
                 llm_model: str = "gemma3:4b",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_model: Ollama model name
            ollama_base_url: Ollama server URL
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.embeddings = None
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.llm = None
        self.qa_chain = None
        
        # Documents storage
        self.documents = []
        
        logger.info("RAG System initialized")
    
    def setup_embeddings(self):
        """Setup the embedding model."""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = SentenceTransformer(self.embedding_model_name, device="cpu")
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=self.embedding_model_name,
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        logger.info("Embeddings loaded successfully")
    
    def setup_llm(self):
        """Setup the Ollama LLM."""
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
    
    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents."""
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        logger.info("Creating FAISS vector store...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        logger.info("Vector store created successfully")
    
    def create_bm25_retriever(self, documents: List[Document]):
        """Create BM25 retriever from documents."""
        logger.info("Creating BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 4  # Number of documents to retrieve
        logger.info("BM25 retriever created successfully")
    
    def create_ensemble_retriever(self):
        """Create ensemble retriever combining BM25 and semantic search."""
        if not self.vector_store or not self.bm25_retriever:
            raise ValueError("Vector store and BM25 retriever must be created first")
        
        logger.info("Creating ensemble retriever...")
        # Create semantic retriever from vector store
        semantic_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, semantic_retriever],
            weights=[0.5, 0.5]  # Equal weight for BM25 and semantic search
        )
        logger.info("Ensemble retriever created successfully")
    
    def setup_qa_chain(self):
        """Setup the question-answering chain."""
        if not self.ensemble_retriever or not self.llm:
            raise ValueError("Retriever and LLM must be setup first")
        
        # Custom prompt template
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
    
    def load_pdf_folder(self, folder_path: str):
        """Load and process all PDFs from a folder."""
        logger.info(f"Loading PDFs from: {folder_path}")
        
        # Process PDFs
        self.documents = self.pdf_processor.process_pdf_folder(folder_path)
        
        if not self.documents:
            raise ValueError(f"No documents could be processed from {folder_path}")
        
        # Setup embeddings if not already done
        if not self.embeddings:
            self.setup_embeddings()
        
        # Create retrievers
        self.create_vector_store(self.documents)
        self.create_bm25_retriever(self.documents)
        self.create_ensemble_retriever()
        
        # Setup LLM and QA chain
        if not self.llm:
            self.setup_llm()
        self.setup_qa_chain()
        
        logger.info("RAG system is ready for questions!")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources."""
        if not self.qa_chain:
            raise ValueError("RAG system not initialized. Call load_pdf_folder() first.")
        
        logger.info(f"Processing question: {question}")
        
        try:
            result = self.qa_chain({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result["result"],
                "sources": []
            }
            
            # Add source information
            for doc in result.get("source_documents", []):
                source_info = {
                    "filename": doc.metadata.get("filename", "Unknown"),
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
    
    def save_vector_store(self, path: str):
        """Save the vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str):
        """Load the vector store from disk."""
        if not self.embeddings:
            self.setup_embeddings()
        
        self.vector_store = FAISS.load_local(path, self.embeddings)
        logger.info(f"Vector store loaded from {path}")


def main():
    """Example usage of the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System with Hybrid Retriever")
    parser.add_argument("--pdf_folder", type=str, required=True, 
                      help="Path to folder containing PDF files")
    parser.add_argument("--question", type=str, 
                      help="Question to ask (interactive mode if not provided)")
    parser.add_argument("--embedding_model", type=str, 
                      default="nomic-ai/nomic-embed-text-v1",
                      help="HuggingFace embedding model")
    parser.add_argument("--llm_model", type=str, default="gemma3:4b",
                      help="Ollama model name")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = HybridRAGSystem(
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )
    
    try:
        # Load PDFs
        rag.load_pdf_folder(args.pdf_folder)
        
        if args.question:
            # Single question mode
            response = rag.ask_question(args.question)
            print(f"\nQuestion: {response['question']}")
            print(f"Answer: {response['answer']}")
            print(f"\nSources ({len(response['sources'])}):")
            for i, source in enumerate(response['sources'], 1):
                print(f"{i}. {source['filename']}")
                print(f"   Preview: {source['content_preview']}")
        else:
            # Interactive mode
            print("\n🤖 RAG System ready! Type 'quit' to exit.")
            while True:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if question:
                    response = rag.ask_question(question)
                    print(f"\n💡 Answer: {response['answer']}")
                    
                    if response['sources']:
                        print(f"\n📚 Sources ({len(response['sources'])}):")
                        for i, source in enumerate(response['sources'], 1):
                            print(f"  {i}. {source['filename']}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 