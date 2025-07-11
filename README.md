# 📚 RAG Document System

A comprehensive Retrieval-Augmented Generation (RAG) system for processing PDF and DOC files with web scraping capabilities. This system can extract documents from websites like EMMA MSRB, process them, and provide intelligent question-answering with proper references.

## 🚀 Features

### Core Functionality
- **Document Processing**: Extract text from PDF, DOC, and DOCX files with page-level granularity
- **Web Scraping**: Download documents from websites including EMMA MSRB
- **Vector Storage**: Store document embeddings using ChromaDB for efficient retrieval
- **Question Answering**: Get answers with references to source documents
- **Reference Tracking**: Full traceability with folder, file, and page information

### Advanced Features
- **Local LLM Support**: Use Gemma 3B, Llama 3.2, and other local models via Ollama
- **Local Embeddings**: Support for Nomic embeddings and other local embedding models
- **Chunking**: Intelligent text chunking with overlap for better context
- **Similarity Search**: Semantic search using sentence transformers
- **Confidence Scoring**: Measure answer reliability
- **Multiple Interfaces**: Command-line and web-based interfaces
- **Fallback Support**: Works without OpenAI API using keyword matching

## 📋 Requirements

- Python 3.8+
- Chrome browser (for web scraping)
- OpenAI API key (optional, for enhanced question answering)
- Ollama (optional, for local LLM models)

## 🛠️ Installation

### Quick Setup (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd docsearch
   ```

2. **Run the automated setup**:
   ```bash
   # Use default environment name 'venv'
   python setup.py
   
   # Use custom environment name
   python setup.py my_rag_env
   
   # Force recreation of existing environment
   python setup.py --force
   
   # Skip tests after installation
   python setup.py --skip-tests
   
   # Show all options
   python setup.py --help
   ```

   This will:
   - ✅ Create a new virtual environment (with custom name if specified)
   - ✅ Install all dependencies
   - ✅ Create necessary directories
   - ✅ Set up environment variables
   - ✅ Test the system (unless --skip-tests is used)
   - ✅ Create convenient run scripts

3. **Activate the virtual environment**:
   
   **Windows:**
   ```bash
   # If using default 'venv' name
   venv\Scripts\activate
   
   # If using custom name (e.g., my_rag_env)
   my_rag_env\Scripts\activate
   
   # Or use the activation script
   activate_rag.bat
   ```
   
   **macOS/Linux:**
   ```bash
   # If using default 'venv' name
   source venv/bin/activate
   
   # If using custom name (e.g., my_rag_env)
   source my_rag_env/bin/activate
   
   # Or use the activation script
   ./activate_rag.sh
   ```

### Manual Setup (Alternative)

If you prefer manual setup:

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**:
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key if needed
   ```

## 🚀 Quick Start

### 1. Setup

Run the automated setup script:

```bash
python setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up configuration files
- Create necessary directories

### 2. Configure Environment

Copy the example environment file and configure your settings:

```bash
# Copy the example file
cp env_example.txt .env

# Edit with your preferred settings
# Key settings:
# - OPENAI_API_KEY (optional, for OpenAI models)
# - USE_LOCAL_LLM=true (to use local models via Ollama)
# - EMBEDDING_MODEL=all-MiniLM-L6-v2 (embedding model to use)
```

### 3. Process Documents

```bash
# Process documents from your docsJuly folder
python main.py process-local --dir ./docsJuly --mode skip_existing

# Or use the convenient script
./scripts/run_rag.bat process-local --dir ./docsJuly   # Windows
```

### 4. Scrape Documents from Web

```bash
# Scrape documents from EMMA MSRB
python main.py scrape-web --url https://emma.msrb.org --max-docs 10
```

### 5. Ask Questions

```bash
# Ask a question about your documents
python main.py ask --question "What is the main topic of these documents?"
```

### 6. Launch Chat Interface

```bash
# Launch the advanced chat interface
streamlit run advanced_chat.py

# Or use the convenient scripts
./scripts/run_chat.bat   # Windows (from scripts folder)
start_chat.bat          # Windows (simple launcher in root)
```

### 7. Interactive Mode

```bash
# Run in interactive mode for multiple questions
python main.py interactive
```

## 📖 Usage Examples

### Command Line Interface

```bash
# Process local documents
python main.py process-local --dir ./docsJuly

# Scrape from EMMA MSRB
python main.py scrape-web --url https://emma.msrb.org --max-docs 15

# Ask specific questions
python main.py ask --question "What are the key financial terms mentioned?"
python main.py ask --question "Who are the parties involved?" --top-k 10

# Check system status
python main.py stats

# Reset the system
python main.py reset
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env_example.txt`:

```bash
# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Local LLM Configuration
USE_LOCAL_LLM=true
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_MODEL=gemma2:3b

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Local Embeddings Configuration
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=nomic-embed-text

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Optional overrides
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Available Local Models

#### LLM Models (via Ollama)
- `gemma3:4b` - Gemma 3 4B (recommended for most systems)
- `gemma2:3b` - Gemma 2 3B (recommended for most systems)
- `gemma2:7b` - Gemma 2 7B (requires more memory)
- `llama3.2:3b` - Llama 3.2 3B
- `llama3.2:8b` - Llama 3.2 8B
- `mistral:7b` - Mistral 7B
- `qwen2.5:3b` - Qwen 2.5 3B
- `phi3:3.8b` - Phi-3 3.8B

#### Embedding Models
- `nomic-embed-text` - Nomic Embed Text (Latest)
- `nomic-embed-text-v2` - Nomic Embed Text V2
- `all-MiniLM-L6-v2` - Sentence Transformers MiniLM
- `all-mpnet-base-v2` - Sentence Transformers MPNet

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Scraper   │    │ Document Proc.  │    │  Vector Store   │
│                 │    │                 │    │                 │
│ • EMMA MSRB     │───▶│ • PDF/DOC/DOCX  │───▶│ • ChromaDB      │
│ • General sites │    │ • Text chunking │    │ • Embeddings    │
│ • Downloads     │    │ • Metadata      │    │ • Search        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   RAG System    │    │  Question/Answer│
                       │                 │    │                 │
                       │ • Orchestration │◀───│ • OpenAI API    │
                       │ • Context prep  │    │ • Local LLM     │
                       │ • References    │    │ • Fallback      │
                       └─────────────────┘    └─────────────────┘
```

## 📊 Output Format

### Question Answer Response

```json
{
  "answer": "Based on the documents, the main topic is...",
  "references": [
    {
      "text": "Extracted text snippet...",
      "file_name": "document.pdf",
      "folder": "./docsJuly",
      "page": "5",
      "similarity_score": 0.85,
      "rank": 1
    }
  ],
  "confidence": 0.82,
  "search_results_count": 3
}
```

## 🌐 Web Scraping Support

### Supported Websites

- **EMMA MSRB**: Specialized scraper for municipal securities documents
- **General Websites**: Automatic detection and download of PDF/DOC files
- **Custom Sites**: Extensible framework for site-specific scrapers

### Scraping Features

- **Automatic Detection**: Finds document links on web pages
- **Download Management**: Handles large files with progress tracking
- **Error Recovery**: Retry logic with exponential backoff
- **Browser Automation**: Uses Selenium for JavaScript-heavy sites

## 🔍 Search Capabilities

### Semantic Search

- **Embedding Model**: Uses local embeddings (Nomic) or sentence transformers
- **Similarity Scoring**: Cosine similarity with configurable thresholds
- **Context Preservation**: Maintains document structure and metadata

### Question Answering

- **OpenAI Integration**: Uses GPT models for natural language answers
- **Local LLM Support**: Uses Ollama models (Gemma, Llama, etc.)
- **Fallback Mode**: Keyword-based matching when no LLM available
- **Context Assembly**: Combines multiple relevant chunks
- **Reference Tracking**: Links answers to source documents

## 📁 File Structure

```
docsearch/
├── ragvenv/                   # Virtual environment (created by setup)
├── docsJuly/                  # Your document directory
│   ├── document1.pdf
│   └── document2.pdf
├── downloads/                 # Web-scraped documents
├── chroma_db/                 # Vector database
├── logs/                      # System logs
├── scripts/                   # Utility scripts
│   ├── run_chat.bat          # Launch chat interface
│   ├── run_rag.bat           # Run main commands
│   ├── run_manager.bat       # Launch model manager
│   └── activate_rag.bat      # Activate environment
├── tests/                     # Test files directory
│   ├── __init__.py           # Test package initialization
│   ├── README.md             # Test documentation
│   ├── run_tests.py          # Test runner
│   ├── test_logging.py       # Logging tests
│   ├── test_search.py        # Search functionality tests
│   ├── test_system.py        # System integration tests
│   ├── test_ui_loop_fix.py   # UI loop fix tests
│   └── test_document_processing_modes.py # Document processing tests
├── config.py                  # Configuration settings
├── document_processor.py      # PDF/DOC processing
├── web_scraper.py            # Web scraping functionality
├── vector_store.py           # Vector database operations
├── local_llm.py              # Local LLM management
├── local_embeddings.py       # Local embeddings management
├── logging_config.py         # Logging configuration
├── rag_system.py             # Main RAG orchestration
├── model_manager.py          # Model configuration interface
├── main.py                   # Command-line interface
├── advanced_chat.py          # Advanced chat interface (main UI)
├── setup.py                  # Automated setup script
├── requirements.txt          # Python dependencies
├── env_example.txt           # Environment configuration example
└── README.md                 # This file
```

## 🧪 Testing

### Running Tests

The project includes a comprehensive test suite in the `tests/` directory.

```bash
# Run all tests
python tests/run_tests.py

# Run specific test
python tests/run_tests.py test_search

# Run individual tests
python tests/test_logging.py
python tests/test_search.py
python tests/test_system.py

# Clear database for fresh start
python tests/clear_database.py
```

### Test Structure

- **`tests/run_tests.py`** - Test runner for all tests
- **`tests/test_logging.py`** - Tests logging and Unicode support
- **`tests/test_search.py`** - Tests vector store and search
- **`tests/test_system.py`** - Tests complete RAG system
- **`tests/clear_database.py`** - Database clearing utility

See `tests/README.md` for detailed testing documentation.

## 🛠️ Scripts

All Windows batch (.bat) and shell (.sh) scripts are now located in the `scripts/` directory.

### Usage Examples

```bash
# Activate the RAG environment (Windows)
scripts\activate_rag.bat

# Deactivate the RAG environment (Windows)
scripts\deactivate_rag.bat

# Deactivate the RAG environment (Linux/Mac)
scripts/deactivate_rag.sh

# Restart the app (Windows)
scripts\restart_app.bat

# Run the chat interface (Windows)
scripts\run_chat.bat

# Run the RAG system (Windows)
scripts\run_rag.bat
```

Update any custom automation or documentation to use the new script paths.

## 🚨 Troubleshooting

### Common Issues

1. **Virtual Environment Issues**:
   ```bash
   # Recreate virtual environment with default name
   python setup.py --force
   
   # Create new environment with custom name
   python setup.py my_new_env --force
   ```

2. **Ollama Issues**:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Install model
   ollama pull gemma2:3b
   ```

3. **Dependency Issues**:
   ```bash
   # Activate virtual environment first
   source venv/bin/activate  # Unix
   venv\Scripts\activate     # Windows
   
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

4. **Model Loading Issues**:
   ```bash
   # Test models
   python model_manager.py
   # Choose option 4 to test specific models
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- **OpenAI** for language models
- **Ollama** for local LLM management
- **Streamlit** for web interface
- **Selenium** for web scraping

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**Happy Document Searching! 📚🔍** 

## 🖥️ User Interfaces

### Advanced Chat Interface (Recommended)

The main interface for interacting with your documents:

```bash
streamlit run advanced_chat.py
# Or: ./scripts/run_chat.bat
```

**Features:**
- 💬 **Interactive Chat**: Natural conversation with your documents
- 📁 **Document Management**: Upload files or process directories with options:
  - **Skip Existing**: Only process new documents (recommended)
  - **Overwrite**: Replace existing documents
  - **Add Only**: Fail if documents already exist
- 🔍 **Document Search**: Browse and search processed documents
- 📊 **Analytics**: View document statistics and processing info
- ⚙️ **Settings**: Configure search parameters and display options
- 📥 **Export**: Save conversations and references

### Command Line Interface

For automation and scripting:

```bash
# Process documents with different modes
python main.py process-local --dir ./docsJuly --mode skip_existing
python main.py process-local --dir ./docsJuly --mode upsert
python main.py process-local --dir ./docsJuly --mode add

# Web scraping
python main.py scrape-web --url https://emma.msrb.org --max-docs 15

# Ask questions
python main.py ask --question "What are the key findings?" --top-k 10

# Interactive mode
python main.py interactive

# System management
python main.py stats
python main.py reset
``` 