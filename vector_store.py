import os
import time
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
from config import Config
from local_embeddings import LocalEmbeddingsManager, NomicEmbeddingsManager
from logging_config import get_logger

logger = get_logger('vector_store')

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
        
        # Always use the EMBEDDING_MODEL from config as the primary model
        primary_model = config.EMBEDDING_MODEL
        
        # Check if we should use local embeddings
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
                logger.warning("⚠️ Local embeddings failed, falling back to primary model")
        
        # Use the primary embedding model from config
        logger.info(f"🔄 Loading primary embedding model: {primary_model}")
        try:
            model = SentenceTransformer(primary_model)
            logger.info("✅ Primary embedding model loaded")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load primary embedding model: {e}")
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
    
    def check_existing_documents(self, chunks: List[Dict]) -> Dict:
        """Check which documents already exist in the vector store."""
        existing_files = set()
        new_files = set()
        existing_chunks = 0
        new_chunks = 0
        
        try:
            # Get all existing documents to check against
            existing_docs = self.collection.get(
                limit=10000,  # Get a large number to check against
                include=['metadatas']
            )
            
            existing_file_paths = set()
            existing_chunk_ids = set()
            
            if existing_docs['metadatas']:
                for metadata in existing_docs['metadatas']:
                    if metadata:
                        file_path = metadata.get('file_path', '')
                        chunk_id = metadata.get('chunk_id', '')
                        if file_path:
                            existing_file_paths.add(file_path)
                        if chunk_id:
                            existing_chunk_ids.add(chunk_id)
            
            # Check each chunk
            for chunk in chunks:
                file_path = chunk['file_path']
                chunk_id = chunk['chunk_id']
                
                if file_path in existing_file_paths:
                    existing_files.add(file_path)
                    existing_chunks += 1
                else:
                    new_files.add(file_path)
                    new_chunks += 1
            
            return {
                'existing_files': list(existing_files),
                'new_files': list(new_files),
                'existing_chunks': existing_chunks,
                'new_chunks': new_chunks,
                'total_files': len(existing_files) + len(new_files),
                'total_chunks': existing_chunks + new_chunks
            }
            
        except Exception as e:
            logger.error(f"Error checking existing documents: {e}")
            return {
                'existing_files': [],
                'new_files': [],
                'existing_chunks': 0,
                'new_chunks': len(chunks),
                'total_files': 0,
                'total_chunks': len(chunks)
            }
    
    def add_documents(self, chunks: List[Dict], mode: str = "add") -> bool:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            mode: "add" (fails on duplicates), "upsert" (overwrites duplicates), "skip_existing" (skips duplicates)
        """
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
            
            if mode == "upsert":
                # Use upsert to overwrite existing documents
                self.collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully upserted {len(chunks)} chunks to vector store")
                
            elif mode == "skip_existing":
                # Check for existing IDs and filter them out
                existing_docs = self.collection.get(
                    limit=10000,
                    include=['metadatas']
                )
                
                existing_chunk_ids = set()
                if existing_docs['metadatas']:
                    for metadata in existing_docs['metadatas']:
                        if metadata and 'chunk_id' in metadata:
                            existing_chunk_ids.add(metadata['chunk_id'])
                
                # Filter out existing chunks
                filtered_chunks = []
                filtered_texts = []
                filtered_ids = []
                filtered_metadatas = []
                filtered_embeddings = []
                
                for i, chunk_id in enumerate(ids):
                    if chunk_id not in existing_chunk_ids:
                        filtered_chunks.append(chunks[i])
                        filtered_texts.append(texts[i])
                        filtered_ids.append(chunk_id)
                        filtered_metadatas.append(metadatas[i])
                        filtered_embeddings.append(embeddings[i])
                
                if filtered_chunks:
                    self.collection.add(
                        embeddings=filtered_embeddings,
                        documents=filtered_texts,
                        metadatas=filtered_metadatas,
                        ids=filtered_ids
                    )
                    logger.info(f"Successfully added {len(filtered_chunks)} new chunks (skipped {len(chunks) - len(filtered_chunks)} existing)")
                else:
                    logger.info("All chunks already exist, nothing to add")
                    
            else:  # mode == "add"
                # Use regular add (will fail on duplicates)
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
        
        start_time = time.time()
        logger.info(f"Starting search for query: '{query}' with top_k={top_k}")
        
        # Add timeout protection
        if time.time() - start_time > 30:  # 30 second timeout
            logger.error("Search timeout - taking too long")
            return []
        
        try:
            # Check if collection has documents
            collection_count = self.collection.count()
            logger.info(f"Collection has {collection_count} documents")
            
            if collection_count == 0:
                logger.warning("Collection is empty, no documents to search")
                return []
            
            # Generate query embedding
            logger.info("Generating query embedding...")
            query_embedding = self._encode_texts([query])
            logger.info(f"Query embedding generated, shape: {len(query_embedding[0])}")
            
            # Search in collection with timeout protection
            logger.info("Executing search in collection...")
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, collection_count),  # Don't ask for more than available
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                logger.info(f"Raw results returned: {len(results['documents'][0])}")
                
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
                    similarity_score = 1 - distance
                    
                    logger.debug(f"Result {i+1}: similarity={similarity_score:.3f}, threshold={self.similarity_threshold}")
                    
                    if similarity_score >= self.similarity_threshold:
                        formatted_results.append({
                            'text': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
                        logger.debug(f"Added result {i+1}: {metadata.get('file_name', 'unknown')} (page {metadata.get('page', 'unknown')})")
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.2f}s")
            logger.info(f"Found {len(formatted_results)} relevant results for query: '{query}'")
            
            if len(formatted_results) == 0:
                logger.warning(f"No results above threshold {self.similarity_threshold} for query: '{query}'")
                # Log some raw results for debugging
                if results['documents'] and results['documents'][0]:
                    logger.info("Top raw results (below threshold):")
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0][:3],  # Show top 3
                        results['metadatas'][0][:3],
                        results['distances'][0][:3]
                    )):
                        similarity = 1 - distance
                        logger.info(f"  {i+1}. {metadata.get('file_name', 'unknown')} (page {metadata.get('page', 'unknown')}) - similarity: {similarity:.3f}")
            
            return formatted_results
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"Error searching vector store after {search_time:.2f}s: {e}")
            logger.error(f"Query: '{query}', top_k: {top_k}")
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