#!/usr/bin/env python3
"""
Test script to verify that the UI loop fix works correctly.
This simulates the Streamlit session state behavior to ensure no infinite loops.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from logging_config import get_logger

logger = get_logger('test_ui_loop')

def test_question_processing():
    """Test that questions are processed correctly without loops."""
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Test questions
    test_questions = [
        "use of proceed",
        "What is the main topic?",
        "What are the key findings?",
        "Who are the parties involved?"
    ]
    
    logger.info("Testing question processing...")
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n--- Test {i}: '{question}' ---")
        
        try:
            # Process the question
            result = rag_system.ask_question(question, top_k=5)
            
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Results found: {result['search_results_count']}")
            
            if result['references']:
                logger.info("References:")
                for j, ref in enumerate(result['references'][:2], 1):  # Show first 2
                    logger.info(f"  {j}. {ref['file_name']} (Page {ref['page']}) - {ref['similarity_score']:.2%}")
            else:
                logger.info("No references found")
                
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
    
    logger.info("\n✅ Question processing test completed")

def test_similarity_threshold():
    """Test that similarity threshold is working correctly."""
    
    rag_system = RAGSystem()
    
    # Test with a specific query that should have low similarity
    query = "use of proceed"
    
    logger.info(f"\n--- Testing similarity threshold for '{query}' ---")
    
    # Get raw search results
    search_results = rag_system.vector_store.search(query, top_k=10)
    
    logger.info(f"Raw search results: {len(search_results)}")
    
    if search_results:
        logger.info("Top results:")
        for i, result in enumerate(search_results[:5], 1):
            logger.info(f"  {i}. {result['metadata']['file_name']} (Page {result['metadata']['page']}) - {result['similarity_score']:.3f}")
    else:
        logger.info("No results found above threshold")
    
    # Test with a broader query
    broad_query = "document"
    logger.info(f"\n--- Testing broader query '{broad_query}' ---")
    
    broad_results = rag_system.vector_store.search(broad_query, top_k=5)
    logger.info(f"Broad query results: {len(broad_results)}")
    
    if broad_results:
        logger.info("Top results:")
        for i, result in enumerate(broad_results[:3], 1):
            logger.info(f"  {i}. {result['metadata']['file_name']} (Page {result['metadata']['page']}) - {result['similarity_score']:.3f}")

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("UI LOOP FIX TEST")
    logger.info("=" * 60)
    
    # Test question processing
    test_question_processing()
    
    # Test similarity threshold
    test_similarity_threshold()
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 