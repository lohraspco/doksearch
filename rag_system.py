import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from document_processor import DocumentProcessor
from vector_store import VectorStore
from web_scraper import WebScraper
from local_llm import LocalLLMManager
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.config = Config()
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.web_scraper = WebScraper()
        
        # Initialize OpenAI client
        if self.config.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not found.")
        
        # Initialize local LLM
        self.local_llm = LocalLLMManager()
        
        # Log system configuration
        self._log_system_config()
    
    def _log_system_config(self):
        """Log the current system configuration."""
        logger.info("🔧 RAG System Configuration:")
        logger.info(f"  OpenAI Available: {self.openai_client is not None}")
        logger.info(f"  Local LLM Available: {self.local_llm.is_available()}")
        logger.info(f"  Local Embeddings: {self.config.USE_LOCAL_EMBEDDINGS}")
        
        if self.local_llm.is_available():
            llm_info = self.local_llm.get_model_info()
            logger.info(f"  Local LLM: {llm_info['model']} ({llm_info['provider']})")
        
        embedding_info = self.vector_store.get_embedding_model_info()
        logger.info(f"  Embeddings: {embedding_info['model']}")
    
    def process_local_documents(self, directory_path: str) -> bool:
        """Process documents from a local directory."""
        try:
            logger.info(f"Processing documents from: {directory_path}")
            
            # Process documents
            chunks = self.document_processor.process_directory(directory_path)
            
            if chunks:
                # Add to vector store
                success = self.vector_store.add_documents(chunks)
                if success:
                    logger.info(f"Successfully processed {len(chunks)} chunks from {directory_path}")
                    return True
                else:
                    logger.error("Failed to add documents to vector store")
                    return False
            else:
                logger.warning(f"No document chunks found in {directory_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing local documents: {e}")
            return False
    
    def scrape_and_process_web_documents(self, url: str, max_documents: int = 10) -> bool:
        """Scrape documents from web and process them."""
        try:
            logger.info(f"Scraping documents from: {url}")
            
            # Scrape documents
            scraped_docs = self.web_scraper.scrape_site(url, max_documents=max_documents)
            
            if scraped_docs:
                # Process downloaded documents
                all_chunks = []
                for doc in scraped_docs:
                    chunks = self.document_processor.process_document(doc['file_path'])
                    all_chunks.extend(chunks)
                
                if all_chunks:
                    # Add to vector store
                    success = self.vector_store.add_documents(all_chunks)
                    if success:
                        logger.info(f"Successfully processed {len(all_chunks)} chunks from {len(scraped_docs)} web documents")
                        return True
                    else:
                        logger.error("Failed to add web documents to vector store")
                        return False
                else:
                    logger.warning("No chunks extracted from scraped documents")
                    return False
            else:
                logger.warning(f"No documents found at {url}")
                return False
                
        except Exception as e:
            logger.error(f"Error scraping and processing web documents: {e}")
            return False
    
    def ask_question(self, question: str, top_k: int = None) -> Dict:
        """Ask a question and get an answer with references."""
        try:
            # Search for relevant documents
            search_results = self.vector_store.search(question, top_k=top_k)
            
            if not search_results:
                return {
                    'answer': "I couldn't find any relevant information to answer your question.",
                    'references': [],
                    'confidence': 0.0
                }
            
            # Prepare context for LLM
            context = self._prepare_context(search_results)
            
            # Generate answer using available LLM
            answer = self._generate_answer(question, context, search_results)
            
            # Prepare references
            references = self._prepare_references(search_results)
            
            return {
                'answer': answer,
                'references': references,
                'confidence': self._calculate_confidence(search_results),
                'search_results_count': len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'references': [],
                'confidence': 0.0
            }
    
    def _prepare_context(self, search_results: List[Dict]) -> str:
        """Prepare context from search results for LLM."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Document {i} (Similarity: {result['similarity_score']:.3f}):\n{result['text']}\n")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str, search_results: List[Dict]) -> str:
        """Generate answer using available LLM (prioritizing local LLM if configured)."""
        
        # Determine LLM priority based on configuration
        use_local_first = self.config.USE_LOCAL_LLM
        
        # Try local LLM first if configured as default
        if use_local_first and self.local_llm.is_available():
            try:
                prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

                answer = self.local_llm.generate_response(
                    prompt, 
                    max_tokens=self.config.LOCAL_LLM_MAX_TOKENS
                )
                
                if answer:
                    return answer.strip()
                    
            except Exception as e:
                logger.error(f"Error generating answer with local LLM: {e}")
                # Fall back to OpenAI or keyword matching
        
        # Try OpenAI if available and not using local LLM as default
        if self.openai_client and not use_local_first:
            try:
                prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

                response = self.openai_client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Always be accurate and cite specific information from the documents when possible."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.LOCAL_LLM_MAX_TOKENS,
                    temperature=self.config.LOCAL_LLM_TEMPERATURE
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"Error generating answer with OpenAI: {e}")
                # Fall back to local LLM or keyword matching
        
        # Fallback: Try local LLM if not tried yet
        if not use_local_first and self.local_llm.is_available():
            try:
                prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {question}

Answer:"""

                answer = self.local_llm.generate_response(
                    prompt, 
                    max_tokens=self.config.LOCAL_LLM_MAX_TOKENS
                )
                
                if answer and not answer.startswith("Error"):
                    return answer.strip()
                else:
                    logger.warning("Local LLM returned error, falling back to keyword matching")
                    
            except Exception as e:
                logger.error(f"Error generating answer with local LLM: {e}")
        
        # Fallback to keyword matching
        return self._fallback_answer(question, context)
    
    def _fallback_answer(self, question: str, context: str) -> str:
        """Fallback answer method when no LLM is available."""
        # Simple keyword-based answer
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Find the most relevant sentence
        sentences = context.split('.')
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                score = sum(1 for word in question_lower.split() if word in sentence.lower())
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
        
        if best_sentence:
            return f"Based on the available documents, here's the most relevant information: {best_sentence}."
        else:
            return "I found some documents but couldn't extract a specific answer to your question. Please review the references for more details."
    
    def _prepare_references(self, search_results: List[Dict]) -> List[Dict]:
        """Prepare formatted references from search results."""
        references = []
        
        for result in search_results:
            metadata = result['metadata']
            references.append({
                'text': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                'file_name': metadata['file_name'],
                'folder': metadata['folder'],
                'page': metadata['page'],
                'similarity_score': result['similarity_score'],
                'rank': result['rank']
            })
        
        return references
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """Calculate confidence score based on search results."""
        if not search_results:
            return 0.0
        
        # Average similarity score of top results
        avg_similarity = sum(r['similarity_score'] for r in search_results) / len(search_results)
        
        # Boost confidence if we have multiple high-quality results
        if len(search_results) >= 3:
            avg_similarity *= 1.1
        
        return min(avg_similarity, 1.0)
    
    def get_system_stats(self) -> Dict:
        """Get system statistics."""
        vector_stats = self.vector_store.get_collection_stats()
        embedding_info = self.vector_store.get_embedding_model_info()
        
        stats = {
            'vector_store': vector_stats,
            'openai_available': self.openai_client is not None,
            'local_llm_available': self.local_llm.is_available(),
            'local_embeddings': self.config.USE_LOCAL_EMBEDDINGS,
            'supported_extensions': self.config.SUPPORTED_EXTENSIONS,
            'embedding_model': embedding_info
        }
        
        if self.local_llm.is_available():
            stats['local_llm_info'] = self.local_llm.get_model_info()
        
        return stats
    
    def reset_system(self) -> bool:
        """Reset the entire system (clear vector store)."""
        try:
            success = self.vector_store.reset_collection()
            if success:
                logger.info("System reset successfully")
            return success
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            return False
    
    def test_local_models(self) -> Dict:
        """Test local models and return status."""
        results = {
            'local_llm': False,
            'local_embeddings': False,
            'messages': []
        }
        
        # Test local LLM
        if self.local_llm.is_available():
            try:
                test_response = self.local_llm.generate_response("Hello, this is a test.", max_tokens=10)
                if test_response and not test_response.startswith("Error"):
                    results['local_llm'] = True
                    results['messages'].append("✅ Local LLM working")
                else:
                    results['messages'].append("❌ Local LLM test failed")
            except Exception as e:
                results['messages'].append(f"❌ Local LLM error: {e}")
        else:
            results['messages'].append("⚠️ Local LLM not available")
        
        # Test local embeddings
        if self.config.USE_LOCAL_EMBEDDINGS:
            try:
                embedding_info = self.vector_store.get_embedding_model_info()
                if embedding_info['available']:
                    results['local_embeddings'] = True
                    results['messages'].append("✅ Local embeddings working")
                else:
                    results['messages'].append("❌ Local embeddings not available")
            except Exception as e:
                results['messages'].append(f"❌ Local embeddings error: {e}")
        else:
            results['messages'].append("⚠️ Local embeddings not enabled")
        
        return results 