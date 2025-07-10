#!/usr/bin/env python3
"""
Test script to verify logging works without Unicode issues
"""

import sys
import os
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, get_logger
import logging

def test_logging():
    """Test logging functionality."""
    print("Testing logging system...")
    
    # Set up logging
    loggers = setup_logging(log_level=logging.DEBUG, log_file="test_logging.log")
    logger = get_logger('test_logging')
    
    # Test various log messages with emojis
    logger.info("✅ This is a test success message")
    logger.warning("⚠️ This is a test warning message")
    logger.error("❌ This is a test error message")
    logger.debug("🔍 This is a test debug message")
    
    # Test with special characters and emojis
    logger.info("Testing special characters: áéíóú ñ ç 🎉")
    logger.info("Testing numbers: 123 456 789 📊")
    logger.info("Testing symbols: @#$%^&*() 🚀")
    
    print("Logging test completed. Check logs/test_logging.log")

def test_vector_store_logging():
    """Test vector store logging specifically."""
    print("Testing vector store logging...")
    
    try:
        from vector_store import VectorStore
        
        # This should trigger the logging that was causing issues
        vs = VectorStore()
        stats = vs.get_collection_stats()
        print(f"Vector store stats: {stats}")
        
    except Exception as e:
        print(f"Error testing vector store: {e}")

if __name__ == "__main__":
    test_logging()
    test_vector_store_logging()
    print("All tests completed!") 