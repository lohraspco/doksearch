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

### 1. Configure Local Models (Optional)

```bash
# Run the model manager
python model_manager.py

# Or use the convenient script
./run_manager.sh  # macOS/Linux
run_manager.bat   # Windows
```

### 2. Process Local Documents

```bash
# Process documents from your docsJuly folder
python main.py process-local --dir ./docsJuly

# Or use the convenient script
./run_rag.sh process-local --dir ./docsJuly  # macOS/Linux
run_rag.bat process-local --dir ./docsJuly   # Windows
```

### 3. Scrape Documents from Web

```bash
# Scrape documents from EMMA MSRB
python main.py scrape-web --url https://emma.msrb.org --max-docs 10
```

### 4. Ask Questions

```bash
# Ask a question about your documents
python main.py ask --question "What is the main topic of these documents?"
```

### 5. Launch Chat Interface

```bash
# Launch the advanced chat interface
streamlit run advanced_chat.py

# Or use the convenient script
./run_chat.sh  # macOS/Linux
run_chat.bat   # Windows
```

### 6. Interactive Mode

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

### Web Interface

1. **Upload Documents**: Use the file uploader or specify a directory path
2. **Web Scraping**: Enter a URL and set the maximum number of documents
3. **Ask Questions**: Type questions and get answers with references
4. **View Documents**: Browse and search through processed documents

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
├── venv/                      # Virtual environment (created by setup)
├── docsJuly/                  # Your document directory
│   ├── document1.pdf
│   └── document2.pdf
├── downloads/                 # Web-scraped documents
├── chroma_db/                 # Vector database
├── config.py                  # Configuration settings
├── document_processor.py      # PDF/DOC processing
├── web_scraper.py            # Web scraping functionality
├── vector_store.py           # Vector database operations
├── local_llm.py              # Local LLM management
├── local_embeddings.py       # Local embeddings management
├── rag_system.py             # Main RAG orchestration
├── model_manager.py          # Model configuration interface
├── main.py                   # Command-line interface
├── chat_ui.py                # Simple chat interface
├── advanced_chat.py          # Advanced chat interface
├── streamlit_app.py          # Full dashboard interface
├── test_system.py            # System testing
├── setup.py                  # Automated setup script
├── requirements.txt          # Python dependencies
├── env_example.txt           # Environment variables template
├── activate_rag.sh           # Environment activation (Unix)
├── activate_rag.bat          # Environment activation (Windows)
├── run_rag.sh               # RAG system runner (Unix)
├── run_rag.bat              # RAG system runner (Windows)
├── run_chat.sh              # Chat interface runner (Unix)
├── run_chat.bat             # Chat interface runner (Windows)
├── run_manager.sh           # Model manager runner (Unix)
├── run_manager.bat          # Model manager runner (Windows)
└── README.md                # This file
```

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