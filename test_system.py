#!/usr/bin/env python3
"""
Test script for the RAG Document System
=======================================

This script demonstrates the basic functionality of the RAG system
using the existing PDF files in the docsJuly directory.
"""

import os
import sys
from rag_system import RAGSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_processing():
    """Test document processing functionality."""
    print("🧪 Testing Document Processing...")
    
    rag_system = RAGSystem()
    
    # Check if docsJuly directory exists
    docs_dir = "./docsJuly"
    if not os.path.exists(docs_dir):
        print(f"❌ Directory {docs_dir} not found. Please ensure you have PDF files in the docsJuly directory.")
        return False
    
    # Process documents
    success = rag_system.process_local_documents(docs_dir)
    
    if success:
        stats = rag_system.get_system_stats()
        print(f"✅ Document processing successful!")
        print(f"📊 Vector store contains {stats['vector_store'].get('total_documents', 0)} chunks")
        return True
    else:
        print("❌ Document processing failed")
        return False

def test_question_answering():
    """Test question answering functionality."""
    print("\n🧪 Testing Question Answering...")
    
    rag_system = RAGSystem()
    
    # Test questions
    test_questions = [
        "What is the main topic of these documents?",
        "What are the key dates mentioned?",
        "Who are the parties involved?",
        "What are the financial terms discussed?"
    ]
    
    for question in test_questions:
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching for answer...")
        
        result = rag_system.ask_question(question, top_k=3)
        
        print(f"💡 Answer: {result['answer']}")
        print(f"🎯 Confidence: {result['confidence']:.2%}")
        print(f"📚 References: {result['search_results_count']}")
        
        if result['references']:
            print("📖 Top reference:")
            ref = result['references'][0]
            print(f"   File: {ref['file_name']} (Page {ref['page']})")
            print(f"   Similarity: {ref['similarity_score']:.2%}")
            print(f"   Text: {ref['text'][:100]}...")

def test_system_stats():
    """Test system statistics."""
    print("\n🧪 Testing System Statistics...")
    
    rag_system = RAGSystem()
    stats = rag_system.get_system_stats()
    
    print("📊 System Statistics:")
    print(f"  Vector Store: {stats['vector_store']}")
    print(f"  OpenAI Available: {stats['openai_available']}")
    print(f"  Supported Extensions: {', '.join(stats['supported_extensions'])}")

def main():
    """Run all tests."""
    print("🚀 RAG Document System - Test Suite")
    print("=" * 50)
    
    # Test 1: Document Processing
    if not test_document_processing():
        print("❌ Document processing test failed. Exiting.")
        return
    
    # Test 2: System Statistics
    test_system_stats()
    
    # Test 3: Question Answering
    test_question_answering()
    
    print("\n✅ All tests completed!")
    print("\n💡 Next steps:")
    print("  1. Try the interactive mode: python main.py interactive")
    print("  2. Launch the web interface: streamlit run streamlit_app.py")
    print("  3. Scrape documents from web: python main.py scrape-web --url https://emma.msrb.org")

if __name__ == "__main__":
    main() 