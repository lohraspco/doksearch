import os
import logging
import requests
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLMManager:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.ollama_client = None
        
        # Initialize based on configuration
        if self.config.USE_LOCAL_LLM:
            self._initialize_local_llm()
    
    def _initialize_local_llm(self):
        """Initialize the local LLM based on provider."""
        try:
            if self.config.LOCAL_LLM_PROVIDER == "ollama":
                self._initialize_ollama()
            elif self.config.LOCAL_LLM_PROVIDER == "transformers":
                self._initialize_transformers()
            else:
                logger.error(f"Unsupported LLM provider: {self.config.LOCAL_LLM_PROVIDER}")
                
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            self.model = None
    
    def _initialize_ollama(self):
        """Initialize Ollama client."""
        try:
            # Test Ollama connection
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                logger.info("✅ Ollama connection successful")
                self.ollama_client = OllamaClient(self.config.OLLAMA_BASE_URL)
            else:
                logger.error(f"❌ Ollama connection failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            logger.info("💡 Make sure Ollama is running: ollama serve")
    
    def _initialize_transformers(self):
        """Initialize transformers model."""
        try:
            logger.info(f"🔄 Loading model: {self.config.LOCAL_LLM_MODEL}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.LOCAL_LLM_MODEL,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": self.config.DEVICE_MAP
            }
            
            if self.config.LOAD_IN_8BIT:
                model_kwargs["load_in_8bit"] = True
            elif self.config.LOAD_IN_4BIT:
                model_kwargs["load_in_4bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.LOCAL_LLM_MODEL,
                **model_kwargs
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.config.DEVICE_MAP
            )
            
            logger.info("✅ Transformers model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load transformers model: {e}")
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using local LLM."""
        if not self.model and not self.ollama_client:
            return "Local LLM not available. Please check configuration."
        
        try:
            if self.ollama_client:
                return self._generate_ollama_response(prompt, max_tokens)
            elif self.pipeline:
                return self._generate_transformers_response(prompt, max_tokens)
            else:
                return "No local LLM available."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_ollama_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using Ollama."""
        if not max_tokens:
            max_tokens = self.config.LOCAL_LLM_MAX_TOKENS
        
        try:
            response = self.ollama_client.generate(
                model=self.config.LOCAL_LLM_MODEL,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.LOCAL_LLM_TEMPERATURE,
                top_p=self.config.LOCAL_LLM_TOP_P,
                top_k=self.config.LOCAL_LLM_TOP_K
            )
            return response.strip()
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)}"
    
    def _generate_transformers_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using transformers."""
        if not max_tokens:
            max_tokens = self.config.LOCAL_LLM_MAX_TOKENS
        
        try:
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=self.config.LOCAL_LLM_TEMPERATURE,
                top_p=self.config.LOCAL_LLM_TOP_P,
                top_k=self.config.LOCAL_LLM_TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the input prompt from the response
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Transformers generation error: {e}")
            return f"Error: {str(e)}"
    
    def is_available(self) -> bool:
        """Check if local LLM is available."""
        return self.model is not None or self.ollama_client is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.ollama_client:
            return {
                "provider": "ollama",
                "model": self.config.LOCAL_LLM_MODEL,
                "available": True
            }
        elif self.model:
            return {
                "provider": "transformers",
                "model": self.config.LOCAL_LLM_MODEL,
                "available": True,
                "device": str(next(self.model.parameters()).device)
            }
        else:
            return {
                "provider": "none",
                "model": "none",
                "available": False
            }

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def generate(self, model: str, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(url, json=payload, stream=False)
        response.raise_for_status()
        
        return response.json()["response"]
    
    def list_models(self) -> list:
        """List available models."""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        
        return response.json()["models"]
    
    def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama."""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model}
        
        response = requests.post(url, json=payload)
        return response.status_code == 200 