#!/usr/bin/env python3
"""
Test script to check vector store and search functionality
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, get_logger
import logging

# Set up logging first
loggers = setup_logging(log_level=logging.DEBUG, log_file="test_search.log")
logger = get_logger('test_search')

def test_vector_store():
    """Test vector store functionality."""
    try:
        logger.info("="*60)
        logger.info("TESTING VECTOR STORE")
        logger.info("="*60)
        
        # Import after logging setup
        from vector_store import VectorStore
        from rag_system import RAGSystem
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vs = VectorStore()
        
        # Get collection stats
        logger.info("Getting collection statistics...")
        stats = vs.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Check if collection has documents
        collection_count = vs.collection.count()
        logger.info(f"Collection document count: {collection_count}")
        
        if collection_count == 0:
            logger.warning("Collection is empty! No documents to search.")
            return False
        
        # Get some sample documents
        logger.info("Getting sample documents...")
        sample_docs = vs.get_all_documents(limit=3)
        logger.info(f"Sample documents: {len(sample_docs)}")
        
        for i, doc in enumerate(sample_docs[:2]):
            logger.info(f"Sample doc {i+1}: {doc['metadata']['file_name']} (page {doc['metadata']['page']})")
            logger.info(f"  Text preview: {doc['text'][:100]}...")
        
        # Test search functionality
        logger.info("="*60)
        logger.info("TESTING SEARCH FUNCTIONALITY")
        logger.info("="*60)
        
        test_queries = [
            "use of proceed",
            "proceeds",
            "funding",
            "money",
            "document"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            start_time = time.time()
            
            try:
                results = vs.search(query, top_k=5)
                search_time = time.time() - start_time
                
                logger.info(f"Search completed in {search_time:.2f}s")
                logger.info(f"Found {len(results)} results")
                
                if results:
                    for i, result in enumerate(results[:2]):
                        logger.info(f"  Result {i+1}: {result['metadata']['file_name']} (page {result['metadata']['page']}) - similarity: {result['similarity_score']:.3f}")
                        logger.info(f"    Text: {result['text'][:100]}...")
                else:
                    logger.warning(f"No results found for query: '{query}'")
                    
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
        
        # Test RAG system
        logger.info("="*60)
        logger.info("TESTING RAG SYSTEM")
        logger.info("="*60)
        
        rag = RAGSystem()
        system_stats = rag.get_system_stats()
        logger.info(f"RAG system stats: {system_stats}")
        
        # Test a simple question
        test_question = "use of proceed"
        logger.info(f"Testing RAG question: '{test_question}'")
        
        start_time = time.time()
        try:
            answer = rag.ask_question(test_question)
            total_time = time.time() - start_time
            
            logger.info(f"RAG answer completed in {total_time:.2f}s")
            logger.info(f"Answer: {answer['answer']}")
            logger.info(f"Confidence: {answer['confidence']}")
            logger.info(f"References: {len(answer['references'])}")
            
        except Exception as e:
            logger.error(f"Error getting RAG answer: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_vector_store: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting vector store and search tests...")
    
    success = test_vector_store()
    
    if success:
        logger.info("✅ All tests completed successfully")
    else:
        logger.error("❌ Tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 