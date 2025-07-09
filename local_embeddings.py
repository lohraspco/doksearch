import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalEmbeddingsManager:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.model_name = None
        
        # Initialize based on configuration
        if self.config.USE_LOCAL_EMBEDDINGS:
            self._initialize_local_embeddings()
    
    def _initialize_local_embeddings(self):
        """Initialize local embedding model."""
        try:
            model_name = self.config.LOCAL_EMBEDDING_MODEL
            
            logger.info(f"🔄 Loading embedding model: {model_name}")
            
            # Load the model
            self.model = SentenceTransformer(model_name, device=self.config.DEVICE_MAP)
            self.model_name = model_name
            
            logger.info(f"✅ Embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.model:
            raise ValueError("Local embedding model not available")
        
        try:
            # Encode texts
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if not self.model:
            return 0
        
        # Get dimension from model
        return self.model.get_sentence_embedding_dimension()
    
    def is_available(self) -> bool:
        """Check if local embeddings are available."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model:
            return {
                "model": self.model_name,
                "available": True,
                "dimension": self.get_embedding_dimension(),
                "device": str(next(self.model.parameters()).device) if hasattr(self.model, 'parameters') else "cpu"
            }
        else:
            return {
                "model": "none",
                "available": False,
                "dimension": 0,
                "device": "none"
            }
    
    def test_embedding(self) -> bool:
        """Test the embedding model with a simple example."""
        if not self.model:
            return False
        
        try:
            test_text = "This is a test sentence."
            embedding = self.encode_single(test_text)
            
            # Check if embedding has reasonable shape
            if embedding.shape[0] > 0:
                logger.info(f"✅ Embedding test successful: {embedding.shape}")
                return True
            else:
                logger.error("❌ Embedding test failed: empty embedding")
                return False
                
        except Exception as e:
            logger.error(f"❌ Embedding test failed: {e}")
            return False

class NomicEmbeddingsManager(LocalEmbeddingsManager):
    """Specialized manager for Nomic embeddings."""
    
    def __init__(self):
        super().__init__()
    
    def _initialize_local_embeddings(self):
        """Initialize Nomic embedding model."""
        try:
            model_name = self.config.LOCAL_EMBEDDING_MODEL
            
            # Handle Nomic model names
            if model_name == "nomic-embed-text":
                # Use the latest Nomic embed text model
                model_name = "nomic-ai/nomic-embed-text-v1.5"
            elif model_name == "nomic-embed-text-v2":
                model_name = "nomic-ai/nomic-embed-text-v2"
            
            logger.info(f"🔄 Loading Nomic embedding model: {model_name}")
            
            # Load the model
            self.model = SentenceTransformer(model_name, device=self.config.DEVICE_MAP)
            self.model_name = model_name
            
            logger.info(f"✅ Nomic embedding model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Nomic embedding model: {e}")
            self.model = None
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Nomic embeddings with proper preprocessing."""
        if not self.model:
            raise ValueError("Nomic embedding model not available")
        
        try:
            # Nomic models work better with specific preprocessing
            processed_texts = []
            for text in texts:
                # Clean and normalize text
                processed_text = text.strip()
                if not processed_text:
                    processed_text = "empty"
                processed_texts.append(processed_text)
            
            # Encode texts
            embeddings = self.model.encode(
                processed_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Nomic models benefit from normalization
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts with Nomic: {e}")
            raise 