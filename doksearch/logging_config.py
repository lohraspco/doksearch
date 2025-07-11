import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_file="rag_system.log"):
    """Set up comprehensive logging to both console and file."""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter('%(message)s')
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Configure console for Unicode support on Windows
    try:
        # On Windows, set console to UTF-8 mode
        if os.name == 'nt':  # Windows
            # Enable UTF-8 mode for the console
            os.system('chcp 65001 >nul 2>&1')
            
            # Reconfigure stdout if possible
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
                
    except Exception as e:
        # If Unicode setup fails, continue without it
        pass
    
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (detailed formatting)
    file_handler = RotatingFileHandler(
        f"logs/{log_file}",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'  # Ensure UTF-8 encoding for Unicode support
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Create specific loggers for different components
    loggers = {
        'rag_system': logging.getLogger('rag_system'),
        'vector_store': logging.getLogger('vector_store'),
        'document_processor': logging.getLogger('document_processor'),
        'local_embeddings': logging.getLogger('local_embeddings'),
        'local_llm': logging.getLogger('local_llm'),
        'web_scraper': logging.getLogger('web_scraper'),
        'search': logging.getLogger('search')
    }
    
    # Set levels for specific loggers
    for logger_name, logger in loggers.items():
        logger.setLevel(log_level)
    
    # Log startup message
    logging.info("="*80)
    logging.info("RAG SYSTEM LOGGING STARTED")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")
    logging.info(f"Log file: logs/{log_file}")
    logging.info("="*80)
    
    return loggers

def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)

def log_search_attempt(query, results_count, search_time, logger=None):
    """Log search attempt details."""
    if logger is None:
        logger = logging.getLogger('search')
    
    logger.info(f"Search Query: '{query}'")
    logger.info(f"Results Found: {results_count}")
    logger.info(f"Search Time: {search_time:.2f}s")
    
    if results_count == 0:
        logger.warning(f"No results found for query: '{query}'")

def log_embedding_info(model_name, dimension, device, logger=None):
    """Log embedding model information."""
    if logger is None:
        logger = logging.getLogger('local_embeddings')
    
    logger.info(f"Embedding Model: {model_name}")
    logger.info(f"Embedding Dimension: {dimension}")
    logger.info(f"Device: {device}")

def log_vector_store_stats(stats, logger=None):
    """Log vector store statistics."""
    if logger is None:
        logger = logging.getLogger('vector_store')
    
    logger.info("Vector Store Statistics:")
    logger.info(f"  Total Documents: {stats.get('total_documents', 0)}")
    logger.info(f"  Collection Name: {stats.get('collection_name', 'unknown')}")
    logger.info(f"  Embedding Model: {stats.get('embedding_model', 'unknown')}") 