import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-3.5-turbo"
    
    # Local LLM Configuration (Default to local LLM)
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma3:4b")
    LOCAL_LLM_PROVIDER = os.getenv("LOCAL_LLM_PROVIDER", "ollama")  # ollama, llama.cpp, transformers
    
    # Available Local Models
    AVAILABLE_LOCAL_MODELS = {
        "gemma3:4b": "Gemma 3 4B (Ollama)",
        "gemma2:3b": "Gemma 2 3B (Ollama)",
        "gemma2:7b": "Gemma 2 7B (Ollama)", 
        "llama3.2:3b": "Llama 3.2 3B (Ollama)",
        "llama3.2:8b": "Llama 3.2 8B (Ollama)",
        "mistral:7b": "Mistral 7B (Ollama)",
        "qwen2.5:3b": "Qwen 2.5 3B (Ollama)",
        "phi3:3.8b": "Phi-3 3.8B (Ollama)",
        "nous-hermes2:7b": "Nous Hermes 2 7B (Ollama)"
    }
    
    # Embedding Configuration
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Available Local Embedding Models
    AVAILABLE_EMBEDDING_MODELS = {
        "nomic-embed-text": "Nomic Embed Text (Latest)",
        "nomic-embed-text-v2": "Nomic Embed Text V2",
        "all-MiniLM-L6-v2": "Sentence Transformers MiniLM",
        "all-mpnet-base-v2": "Sentence Transformers MPNet",
        "text-embedding-3-small": "OpenAI Text Embedding 3 Small",
        "text-embedding-3-large": "OpenAI Text Embedding 3 Large"
    }
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "documents"
    
    # Document Processing
    SUPPORTED_EXTENSIONS = ['.pdf', '.doc', '.docx']
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Web Scraping
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    
    # File Storage
    DOWNLOADS_DIR = "./downloads"
    PROCESSED_DIR = "./processed"
    
    # Embedding Model Configuration
    # This is the primary embedding model used throughout the app
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Search Configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.2  # Lowered from 0.3 to catch more edge cases and exact matches
    
    # Local LLM Settings
    LOCAL_LLM_TEMPERATURE = 0.3
    LOCAL_LLM_MAX_TOKENS = 500
    LOCAL_LLM_TOP_P = 0.9
    LOCAL_LLM_TOP_K = 40
    
    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Model Loading Settings
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    DEVICE_MAP = os.getenv("DEVICE_MAP", "cpu")  # cpu, cuda, mps (auto causes issues) 