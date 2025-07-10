#!/usr/bin/env python3
"""
Script to clear the vector database and start fresh
"""

import os
import shutil
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_config import setup_logging, get_logger
import logging

# Set up logging
loggers = setup_logging(log_level=logging.INFO, log_file="clear_database.log")
logger = get_logger('clear_database')

def clear_vector_database():
    """Clear the vector database completely."""
    try:
        logger.info("🗑️ Clearing vector database...")
        
        # Import vector store
        from vector_store import VectorStore
        
        # Initialize vector store
        vs = VectorStore()
        
        # Get current stats
        stats = vs.get_collection_stats()
        logger.info(f"Current database stats: {stats}")
        
        # Delete the collection
        success = vs.delete_collection()
        if success:
            logger.info("✅ Collection deleted successfully")
        else:
            logger.warning("⚠️ Collection deletion failed, trying reset...")
            success = vs.reset_collection()
            if success:
                logger.info("✅ Collection reset successfully")
            else:
                logger.error("❌ Collection reset also failed")
                return False
        
        # Also delete the chroma_db directory to ensure complete cleanup
        chroma_dir = "./chroma_db"
        if os.path.exists(chroma_dir):
            try:
                shutil.rmtree(chroma_dir)
                logger.info("✅ ChromaDB directory deleted")
            except Exception as e:
                logger.warning(f"⚠️ Could not delete ChromaDB directory: {e}")
        
        # Verify the database is empty
        try:
            new_vs = VectorStore()
            new_stats = new_vs.get_collection_stats()
            logger.info(f"New database stats: {new_stats}")
            
            if new_stats.get('total_documents', 0) == 0:
                logger.info("✅ Database cleared successfully - ready for fresh start")
                return True
            else:
                logger.warning("⚠️ Database may not be completely cleared")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error verifying database clear: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")
        return False

def clear_logs():
    """Clear old log files."""
    try:
        logs_dir = "./logs"
        if os.path.exists(logs_dir):
            # Keep only the most recent log files
            log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
            for log_file in log_files:
                if log_file not in ['clear_database.log']:  # Keep this one
                    try:
                        os.remove(os.path.join(logs_dir, log_file))
                        logger.info(f"🗑️ Deleted old log: {log_file}")
                    except:
                        pass
    except Exception as e:
        logger.warning(f"⚠️ Could not clear old logs: {e}")

def main():
    """Main function to clear database."""
    logger.info("="*60)
    logger.info("🗑️ VECTOR DATABASE CLEARING SCRIPT")
    logger.info("="*60)
    
    # Clear old logs
    clear_logs()
    
    # Clear vector database
    success = clear_vector_database()
    
    if success:
        logger.info("✅ Database clearing completed successfully!")
        logger.info("📝 You can now reprocess your documents")
        logger.info("📝 Run: streamlit run advanced_chat.py")
    else:
        logger.error("❌ Database clearing failed")
        logger.info("📝 You may need to manually delete the chroma_db directory")
    
    logger.info("="*60)

if __name__ == "__main__":
    main() 