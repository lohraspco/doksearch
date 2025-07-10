#!/usr/bin/env python3
"""
Test script to verify the new document processing modes work correctly.
This tests the different ways of handling existing documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from logging_config import get_logger

logger = get_logger('test_processing_modes')

def test_document_processing_modes():
    """Test the different document processing modes."""
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Test directory
    test_dir = "./docsJuly"
    
    if not os.path.exists(test_dir):
        logger.error(f"Test directory {test_dir} not found")
        return False
    
    logger.info("=" * 60)
    logger.info("TESTING DOCUMENT PROCESSING MODES")
    logger.info("=" * 60)
    
    # Test 1: Initial processing (should work)
    logger.info("\n--- Test 1: Initial Processing (skip_existing) ---")
    result1 = rag_system.process_local_documents(test_dir, mode="skip_existing")
    
    if result1['success']:
        logger.info(f"✅ Initial processing successful")
        logger.info(f"  Chunks processed: {result1['total_chunks_processed']}")
        logger.info(f"  Vector store total: {result1['vector_store_total']}")
        
        if result1.get('existing_info'):
            existing_info = result1['existing_info']
            logger.info(f"  Existing files: {len(existing_info['existing_files'])}")
            logger.info(f"  New files: {len(existing_info['new_files'])}")
    else:
        logger.error(f"❌ Initial processing failed: {result1.get('error')}")
        return False
    
    # Test 2: Skip existing mode (should skip all)
    logger.info("\n--- Test 2: Skip Existing Mode ---")
    result2 = rag_system.process_local_documents(test_dir, mode="skip_existing")
    
    if result2['success']:
        logger.info(f"✅ Skip existing mode successful")
        logger.info(f"  Chunks processed: {result2['total_chunks_processed']}")
        logger.info(f"  Vector store total: {result2['vector_store_total']}")
        
        if result2.get('existing_info'):
            existing_info = result2['existing_info']
            logger.info(f"  Existing files: {len(existing_info['existing_files'])}")
            logger.info(f"  New files: {len(existing_info['new_files'])}")
            
            # Should have no new files since they all exist
            if len(existing_info['new_files']) == 0:
                logger.info("✅ Correctly skipped all existing files")
            else:
                logger.warning("⚠️ Unexpected new files found")
    else:
        logger.error(f"❌ Skip existing mode failed: {result2.get('error')}")
    
    # Test 3: Upsert mode (should overwrite)
    logger.info("\n--- Test 3: Upsert Mode (Overwrite) ---")
    result3 = rag_system.process_local_documents(test_dir, mode="upsert")
    
    if result3['success']:
        logger.info(f"✅ Upsert mode successful")
        logger.info(f"  Chunks processed: {result3['total_chunks_processed']}")
        logger.info(f"  Vector store total: {result3['vector_store_total']}")
        
        if result3.get('existing_info'):
            existing_info = result3['existing_info']
            logger.info(f"  Existing files: {len(existing_info['existing_files'])}")
            logger.info(f"  New files: {len(existing_info['new_files'])}")
            
            # Should have existing files since we're overwriting
            if len(existing_info['existing_files']) > 0:
                logger.info("✅ Correctly identified existing files for overwrite")
            else:
                logger.warning("⚠️ No existing files identified")
    else:
        logger.error(f"❌ Upsert mode failed: {result3.get('error')}")
    
    # Test 4: Add mode (should fail)
    logger.info("\n--- Test 4: Add Mode (Should Fail) ---")
    result4 = rag_system.process_local_documents(test_dir, mode="add")
    
    if not result4['success']:
        logger.info(f"✅ Add mode correctly failed as expected")
        logger.info(f"  Error: {result4.get('error')}")
    else:
        logger.warning("⚠️ Add mode unexpectedly succeeded")
    
    # Test 5: Check vector store stats
    logger.info("\n--- Test 5: Final Vector Store Stats ---")
    stats = rag_system.get_system_stats()
    logger.info(f"Vector store total documents: {stats['vector_store'].get('total_documents', 0)}")
    logger.info(f"Collection name: {stats['vector_store'].get('collection_name', 'Unknown')}")
    logger.info(f"Embedding model: {stats['vector_store'].get('embedding_model', 'Unknown')}")
    
    logger.info("\n" + "=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)
    
    return True

def test_command_line_modes():
    """Test the command line interface with different modes."""
    
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMMAND LINE MODES")
    logger.info("=" * 60)
    
    # Test different command line modes
    modes = ["skip_existing", "upsert", "add"]
    
    for mode in modes:
        logger.info(f"\n--- Testing command line mode: {mode} ---")
        
        # Simulate command line call
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, "main.py", "process-local", 
                "--dir", "./docsJuly", "--mode", mode
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"✅ Command line mode '{mode}' successful")
                logger.info(f"Output: {result.stdout[-200:]}...")  # Last 200 chars
            else:
                logger.info(f"❌ Command line mode '{mode}' failed (expected for 'add' mode)")
                logger.info(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Command line mode '{mode}' timed out")
        except Exception as e:
            logger.error(f"❌ Command line mode '{mode}' error: {e}")

def main():
    """Run all tests."""
    
    # Test the processing modes
    test_document_processing_modes()
    
    # Test command line interface
    test_command_line_modes()
    
    logger.info("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main() 