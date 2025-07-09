import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
from config import Config
from local_embeddings import LocalEmbeddingsManager, NomicEmbeddingsManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.persist_directory = Config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_model = Config.EMBEDDING_MODEL
        self.top_k = Config.TOP_K_RESULTS
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model_instance = self._initialize_embedding_model()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _initialize_embedding_model(self):
        """Initialize the appropriate embedding model."""
        config = Config()
        
        if config.USE_LOCAL_EMBEDDINGS:
            # Use local embeddings
            if config.LOCAL_EMBEDDING_MODEL.startswith("nomic"):
                logger.info("🔄 Initializing Nomic embeddings...")
                embeddings_manager = NomicEmbeddingsManager()
            else:
                logger.info("🔄 Initializing local embeddings...")
                embeddings_manager = LocalEmbeddingsManager()
            
            if embeddings_manager.is_available():
                logger.info("✅ Local embeddings initialized successfully")
                return embeddings_manager
            else:
                logger.warning("⚠️ Local embeddings failed, falling back to default")
        
        # Fallback to default sentence transformers
        logger.info(f"🔄 Loading default embedding model: {self.embedding_model}")
        try:
            model = SentenceTransformer(self.embedding_model)
            logger.info("✅ Default embedding model loaded")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load default embedding model: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings using the appropriate model."""
        try:
            if hasattr(self.embedding_model_instance, 'encode'):
                # Local embeddings manager
                embeddings = self.embedding_model_instance.encode(texts)
                return embeddings.tolist()
            else:
                # Sentence transformers model
                embeddings = self.embedding_model_instance.encode(texts)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict]) -> bool:
        """Add document chunks to the vector store."""
        if not chunks:
            logger.warning("No chunks to add")
            return False
        
        try:
            # Prepare data for ChromaDB
            texts = [chunk['text'] for chunk in chunks]
            ids = [chunk['chunk_id'] for chunk in chunks]
            
            # Create metadata
            metadatas = []
            for chunk in chunks:
                metadata = {
                    'file_name': chunk['file_name'],
                    'file_path': chunk['file_path'],
                    'folder': chunk['folder'],
                    'page': str(chunk['page']),
                    'chunk_id': chunk['chunk_id']
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self._encode_texts(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for similar documents."""
        if not top_k:
            top_k = self.top_k
        
        try:
            # Generate query embedding
            query_embedding = self._encode_texts([query])
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity_score = 1 - distance
                    
                    if similarity_score >= self.similarity_threshold:
                        formatted_results.append({
                            'text': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
            
            logger.info(f"Found {len(formatted_results)} relevant results for query: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get embedding model info
            if hasattr(self.embedding_model_instance, 'get_model_info'):
                model_info = self.embedding_model_instance.get_model_info()
                embedding_info = f"{model_info['model']} ({model_info['dimension']}d)"
            else:
                embedding_info = self.embedding_model
            
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': embedding_info
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection by deleting and recreating it."""
        if self.delete_collection():
            self.collection = self._get_or_create_collection()
            return True
        return False
    
    def get_embedding_model_info(self) -> Dict:
        """Get information about the embedding model."""
        if hasattr(self.embedding_model_instance, 'get_model_info'):
            return self.embedding_model_instance.get_model_info()
        else:
            return {
                "model": self.embedding_model,
                "available": True,
                "dimension": self.embedding_model_instance.get_sentence_embedding_dimension(),
                "device": "default"
            }
    
    def get_all_documents(self, limit: int = 1000) -> List[Dict]:
        """Get all documents from the collection (with optional limit)."""
        try:
            # Get all documents from collection
            results = self.collection.get(
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            # Format results
            formatted_results = []
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                    formatted_results.append({
                        'text': doc,
                        'file_name': metadata.get('file_name', 'Unknown'),
                        'file_path': metadata.get('file_path', 'Unknown'),
                        'folder': metadata.get('folder', 'Unknown'),
                        'page': metadata.get('page', 'Unknown'),
                        'chunk_id': metadata.get('chunk_id', f'chunk_{i}'),
                        'rank': i + 1
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents from collection")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving all documents: {e}")
            return [] 