#!/usr/bin/env python3
"""
Model Manager for RAG System
============================

This module provides a user-friendly interface to manage and configure
local LLM models and embedding models.
"""

import os
import subprocess
import sys
import requests
from typing import Dict, List, Optional
from config import Config

class ModelManager:
    def __init__(self):
        self.config = Config()
    
    def list_available_models(self) -> Dict:
        """List all available models."""
        return {
            'llm_models': self.config.AVAILABLE_LOCAL_MODELS,
            'embedding_models': self.config.AVAILABLE_EMBEDDING_MODELS
        }
    
    def check_ollama_status(self) -> Dict:
        """Check if Ollama is running and list available models."""
        try:
            response = requests.get(f"{self.config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            print(response.json())
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'status': 'running',
                    'models': [model['name'] for model in models],
                    'url': self.config.OLLAMA_BASE_URL
                }
            else:
                return {
                    'status': 'error',
                    'error': f"HTTP {response.status_code}",
                    'url': self.config.OLLAMA_BASE_URL
                }
        except requests.exceptions.ConnectionError:
            return {
                'status': 'not_running',
                'error': 'Cannot connect to Ollama',
                'url': self.config.OLLAMA_BASE_URL
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'url': self.config.OLLAMA_BASE_URL
            }
    
    def install_ollama_model(self, model_name: str) -> Dict:
        """Install a model using Ollama."""
        try:
            url = f"{self.config.OLLAMA_BASE_URL}/api/pull"
            payload = {"name": model_name}
            
            print(f"🔄 Installing {model_name}...")
            response = requests.post(url, json=payload, stream=True)
            
            if response.status_code == 200:
                # Stream the response to show progress
                for line in response.iter_lines():
                    if line:
                        try:
                            data = line.decode('utf-8')
                            if data.strip():
                                print(f"  {data}")
                        except:
                            pass
                
                return {
                    'success': True,
                    'message': f"Successfully installed {model_name}"
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to install {model_name}: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Error installing {model_name}: {str(e)}"
            }
    
    def test_model(self, model_name: str, model_type: str = 'llm') -> Dict:
        """Test if a model is working."""
        try:
            if model_type == 'llm':
                # Test LLM model
                if self.config.LOCAL_LLM_PROVIDER == 'ollama':
                    url = f"{self.config.OLLAMA_BASE_URL}/api/generate"
                    payload = {
                        "model": model_name,
                        "prompt": "Hello, this is a test.",
                        "max_tokens": 10
                    }
                    
                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 200:
                        return {
                            'success': True,
                            'message': f"✅ {model_name} is working"
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"❌ {model_name} test failed: HTTP {response.status_code}"
                        }
                else:
                    return {
                        'success': False,
                        'error': f"❌ Transformers models not yet supported for testing"
                    }
            
            elif model_type == 'embedding':
                # Test embedding model
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(model_name)
                    test_embedding = model.encode("Test sentence")
                    return {
                        'success': True,
                        'message': f"✅ {model_name} is working (dimension: {len(test_embedding)})"
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"❌ {model_name} test failed: {str(e)}"
                    }
            
            else:
                return {
                    'success': False,
                    'error': f"❌ Unknown model type: {model_type}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"❌ Test failed: {str(e)}"
            }
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration settings."""
        try:
            # Read current .env file
            env_file = '.env'
            if not os.path.exists(env_file):
                # Create from example
                if os.path.exists('env_example.txt'):
                    with open('env_example.txt', 'r') as f:
                        content = f.read()
                    with open(env_file, 'w') as f:
                        f.write(content)
                else:
                    return False
            
            # Read current content
            with open(env_file, 'r') as f:
                lines = f.readlines()
            
            # Update values
            updated = False
            for key, value in kwargs.items():
                key_upper = key.upper()
                found = False
                
                for i, line in enumerate(lines):
                    if line.strip().startswith(f"{key_upper}="):
                        lines[i] = f"{key_upper}={value}\n"
                        found = True
                        updated = True
                        break
                
                if not found:
                    # Add new setting
                    lines.append(f"{key_upper}={value}\n")
                    updated = True
            
            # Write back to file
            if updated:
                with open(env_file, 'w') as f:
                    f.writelines(lines)
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False
    
    def get_current_config(self) -> Dict:
        """Get current configuration."""
        return {
            'use_local_llm': self.config.USE_LOCAL_LLM,
            'local_llm_provider': self.config.LOCAL_LLM_PROVIDER,
            'local_llm_model': self.config.LOCAL_LLM_MODEL,
            'use_local_embeddings': self.config.USE_LOCAL_EMBEDDINGS,
            'local_embedding_model': self.config.LOCAL_EMBEDDING_MODEL,
            'openai_available': bool(self.config.OPENAI_API_KEY),
            'ollama_url': self.config.OLLAMA_BASE_URL
        }
    
    def setup_ollama(self) -> Dict:
        """Help user set up Ollama."""
        print("🔧 Ollama Setup Guide")
        print("=" * 40)
        
        # Check if Ollama is installed
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Ollama is installed: {result.stdout.strip()}")
            else:
                print("❌ Ollama is not installed")
                print("\n📥 Install Ollama:")
                print("  Visit: https://ollama.ai/download")
                print("  Or run: curl -fsSL https://ollama.ai/install.sh | sh")
                return {'success': False, 'message': 'Ollama not installed'}
        except FileNotFoundError:
            print("❌ Ollama is not installed")
            print("\n📥 Install Ollama:")
            print("  Visit: https://ollama.ai/download")
            print("  Or run: curl -fsSL https://ollama.ai/install.sh | sh")
            return {'success': False, 'message': 'Ollama not installed'}
        
        # Check if Ollama is running
        ollama_status = self.check_ollama_status()
        if ollama_status['status'] == 'running':
            print("✅ Ollama is running")
            print(f"📋 Available models: {', '.join(ollama_status['models'])}")
            return {'success': True, 'message': 'Ollama is ready'}
        else:
            print("❌ Ollama is not running")
            print("\n🚀 Start Ollama:")
            print("  ollama serve")
            return {'success': False, 'message': 'Ollama not running'}
    
    def recommend_models(self) -> Dict:
        """Recommend models based on system capabilities."""
        recommendations = {
            'llm_models': [],
            'embedding_models': []
        }
        
        # Check system resources
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 8:
                    recommendations['llm_models'].extend([
                        'gemma2:7b',
                        'llama3.2:8b',
                        'mistral:7b'
                    ])
                else:
                    recommendations['llm_models'].extend([
                        'gemma2:3b',
                        'llama3.2:3b',
                        'phi3:3.8b'
                    ])
            else:
                recommendations['llm_models'].extend([
                    'gemma2:3b',
                    'llama3.2:3b',
                    'phi3:3.8b'
                ])
        except:
            recommendations['llm_models'].extend([
                'gemma2:3b',
                'llama3.2:3b',
                'phi3:3.8b'
            ])
        
        # Embedding recommendations
        recommendations['embedding_models'].extend([
            'nomic-embed-text',
            'all-MiniLM-L6-v2'
        ])
        
        return recommendations

def main():
    """Interactive model manager."""
    manager = ModelManager()
    
    print("🤖 RAG System Model Manager")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. 📋 List available models")
        print("2. 🔍 Check Ollama status")
        print("3. 📥 Install Ollama model")
        print("4. 🧪 Test model")
        print("5. ⚙️  Update configuration")
        print("6. 📊 Show current config")
        print("7. 🔧 Setup Ollama")
        print("8. 💡 Get recommendations")
        print("9. ❌ Exit")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == '1':
            models = manager.list_available_models()
            print("\n📋 Available Models:")
            print("\nLLM Models:")
            for key, desc in models['llm_models'].items():
                print(f"  • {key}: {desc}")
            print("\nEmbedding Models:")
            for key, desc in models['embedding_models'].items():
                print(f"  • {key}: {desc}")
        
        elif choice == '2':
            status = manager.check_ollama_status()
            print(f"\n🔍 Ollama Status: {status['status']}")
            if status['status'] == 'running':
                print(f"📋 Models: {', '.join(status['models'])}")
            else:
                print(f"❌ Error: {status.get('error', 'Unknown')}")
        
        elif choice == '3':
            model_name = input("Enter model name (e.g., gemma2:3b): ").strip()
            if model_name:
                result = manager.install_ollama_model(model_name)
                print(f"\n{result['message']}")
        
        elif choice == '4':
            model_name = input("Enter model name: ").strip()
            model_type = input("Enter model type (llm/embedding): ").strip()
            if model_name and model_type:
                result = manager.test_model(model_name, model_type)
                print(f"\n{result['message']}")
        
        elif choice == '5':
            print("\n⚙️ Update Configuration:")
            use_local_llm = input("Use local LLM? (true/false): ").strip()
            if use_local_llm:
                manager.update_config(use_local_llm=use_local_llm)
            
            if use_local_llm.lower() == 'true':
                llm_model = input("LLM model name: ").strip()
                if llm_model:
                    manager.update_config(local_llm_model=llm_model)
            
            use_local_embeddings = input("Use local embeddings? (true/false): ").strip()
            if use_local_embeddings:
                manager.update_config(use_local_embeddings=use_local_embeddings)
            
            if use_local_embeddings.lower() == 'true':
                embedding_model = input("Embedding model name: ").strip()
                if embedding_model:
                    manager.update_config(local_embedding_model=embedding_model)
            
            print("✅ Configuration updated")
        
        elif choice == '6':
            config = manager.get_current_config()
            print("\n📊 Current Configuration:")
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        elif choice == '7':
            result = manager.setup_ollama()
            print(f"\n{result['message']}")
        
        elif choice == '8':
            recommendations = manager.recommend_models()
            print("\n💡 Recommendations:")
            print("\nLLM Models:")
            for model in recommendations['llm_models']:
                print(f"  • {model}")
            print("\nEmbedding Models:")
            for model in recommendations['embedding_models']:
                print(f"  • {model}")
        
        elif choice == '9':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 