#!/usr/bin/env python3
"""
Test script to verify search works with fixed settings
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, get_logger
import logging

# Set up logging
loggers = setup_logging(log_level=logging.INFO, log_file="test_search_fixed.log")
logger = get_logger('test_search_fixed')

def test_search():
    """Test search functionality with fixed settings."""
    try:
        logger.info("="*60)
        logger.info("TESTING SEARCH WITH FIXED SETTINGS")
        logger.info("="*60)
        
        # Import after logging setup
        from vector_store import VectorStore
        from rag_system import RAGSystem
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vs = VectorStore()
        
        # Get collection stats
        stats = vs.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Check if collection has documents
        collection_count = vs.collection.count()
        logger.info(f"Collection document count: {collection_count}")
        
        if collection_count == 0:
            logger.warning("Collection is empty! No documents to search.")
            return False
        
        # Test search with different queries
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
                    for i, result in enumerate(results[:3]):
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
            
            if answer['references']:
                for i, ref in enumerate(answer['references'][:2]):
                    logger.info(f"  Reference {i+1}: {ref['file_name']} (page {ref['page']}) - similarity: {ref['similarity_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error getting RAG answer: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in test_search: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting search tests with fixed settings...")
    
    success = test_search()
    
    if success:
        logger.info("✅ All tests completed successfully")
    else:
        logger.error("❌ Tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 