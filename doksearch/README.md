# RAG System with Hybrid Ensemble Retriever

A sophisticated Retrieval-Augmented Generation (RAG) system that combines BM25 and semantic search for enhanced document retrieval from PDF files. Built with LangChain, using local Gemma3:4b via Ollama and nomic-embed-text embeddings.

## 🚀 Features

- **Hybrid Retrieval**: Combines BM25 (keyword-based) and semantic search for optimal results
- **Local LLM**: Uses Gemma3:4b running locally via Ollama
- **Local Embeddings**: Uses nomic-embed-text for embeddings (no API calls needed)
- **PDF Processing**: Extracts and processes text from multiple PDF files
- **Interactive Mode**: Chat-like interface for asking questions
- **Batch Processing**: Process multiple questions at once
- **Vector Store Persistence**: Save and load embeddings for faster subsequent runs

## 📋 Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Pull Required Model

```bash
ollama pull gemma3:4b
# Alternative models you can try:
# ollama pull llama3.1:8b
# ollama pull mistral:7b
```

### 3. Start Ollama Server

```bash
ollama serve
```

## 🛠️ Installation

1. **Clone or download this repository**

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

3. **Download additional language models** (optional but recommended)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📁 Setup

1. **Create a folder with your PDF files**

```bash
mkdir pdfs
# Copy your PDF files to this folder
```

2. **Verify Ollama is running**

```bash
curl http://localhost:11434/api/tags
```

## 🚀 Usage

### Web UI (Recommended)

Launch the modern web interface:

```bash
streamlit run web_ui.py
```

Then open your browser to `http://localhost:8501`

**Web UI Features:**
- 📁 **Folder Selection**: Browse and select PDF folders with file preview
- ⚡ **Incremental Processing**: Automatically skip already processed files
- 💾 **Vector Database Management**: View processed files, clear database
- 📊 **Analytics Dashboard**: File statistics, processing timeline, charts
- 💬 **Interactive Chat**: Natural language Q&A with source references
- 🏥 **System Health**: Real-time status monitoring
- ⚙️ **Settings**: Configure models and system parameters

### Command Line Interface

**Basic usage:**
```bash
python rag_system.py --pdf_folder ./pdfs --question "What is the main topic discussed?"
```

**Interactive mode:**
```bash
python rag_system.py --pdf_folder ./pdfs
```

**Custom model:**
```bash
python rag_system.py --pdf_folder ./pdfs --llm_model llama3.1:8b --embedding_model nomic-ai/nomic-embed-text-v1
```

### Programmatic Usage

**Enhanced RAG System (with incremental processing):**
```python
from enhanced_rag_system import EnhancedHybridRAGSystem

# Initialize the enhanced RAG system
rag = EnhancedHybridRAGSystem(
    vector_db_path="./vector_db",  # Persistent storage
    embedding_model="nomic-ai/nomic-embed-text-v1",
    llm_model="gemma3:4b"
)

# Load PDFs from a folder (incremental processing)
result = rag.load_pdf_folder("./pdfs")
print(f"Processed {result['total_files']} files, {result['total_chunks']} chunks")

# Ask a question
response = rag.ask_question("What are the key findings?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])} documents")

# Get database statistics
stats = rag.get_database_stats()
print(f"Database contains {stats['total_files']} files")
```

**Original RAG System:**
```python
from rag_system import HybridRAGSystem

# Initialize the RAG system
rag = HybridRAGSystem(
    embedding_model="nomic-ai/nomic-embed-text-v1",
    llm_model="gemma3:4b"
)

# Load PDFs from a folder
rag.load_pdf_folder("./pdfs")

# Ask a question
response = rag.ask_question("What are the key findings?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])} documents")
```

### Advanced Usage Examples

Run the example script for different usage patterns:

```bash
python example_usage.py
```

This includes:
- Basic usage example
- Interactive session
- Custom configuration
- Batch question processing

## 🌐 Web UI Guide

### Getting Started with Web UI

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama:**
   ```bash
   ollama serve
   ollama pull gemma3:4b
   ```

3. **Launch Web UI:**
   ```bash
   streamlit run web_ui.py
   ```

4. **Open your browser to:** `http://localhost:8501`

### Web UI Features

#### 📁 Document Management
- **Folder Selection**: Enter or browse to your PDF folder
- **File Preview**: See list of PDF files before processing
- **Incremental Processing**: Only processes new or changed files
- **Force Reprocess**: Option to reprocess all files
- **Processing Log**: Real-time status updates and error tracking

#### 💬 Chat Interface
- **Natural Language Q&A**: Ask questions about your documents
- **Source References**: See which documents were used for answers
- **Chat History**: Persistent conversation history
- **Chunk Information**: Detailed source tracking with chunk IDs

#### 📊 Analytics Dashboard
- **File Statistics**: Total files, chunks, database size
- **Processing Timeline**: When files were processed
- **File Size Distribution**: Visual charts of document sizes
- **Chunk Distribution**: How documents are split into chunks
- **Detailed Tables**: Complete file information with metadata

#### 💾 Vector Database Management
- **Persistent Storage**: Automatically saves processed documents
- **Database Status**: View processed files and their status
- **Clear Database**: Remove all processed data with confirmation
- **Incremental Updates**: Only process new or changed files
- **Missing File Detection**: Identifies moved or deleted source files

#### ⚙️ Settings & Configuration
- **Model Configuration**: Change embedding and LLM models
- **Chunk Settings**: Adjust chunk size and overlap
- **System Actions**: Restart, clear logs, export logs
- **Health Monitoring**: Check system component status
- **Database Path**: Configure storage location

### Web UI Benefits

✅ **User-Friendly**: No command line knowledge required  
✅ **Visual Feedback**: Progress bars, status indicators, charts  
✅ **Persistent State**: Remembers processed files between sessions  
✅ **Error Handling**: Clear error messages and troubleshooting tips  
✅ **Responsive Design**: Works on desktop and tablet devices  
✅ **Real-time Updates**: Live status monitoring and health checks

## 🔧 Configuration Options

### RAG System Parameters

```python
rag = HybridRAGSystem(
    embedding_model="nomic-ai/nomic-embed-text-v1",  # Embedding model
    llm_model="gemma3:4b",                           # Ollama model
    ollama_base_url="http://localhost:11434"         # Ollama server URL
)
```

### Text Chunking Parameters

Modify in `PDFProcessor.__init__()`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Size of each text chunk
    chunk_overlap=200,      # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

### Retriever Weights

Adjust in `create_ensemble_retriever()`:

```python
self.ensemble_retriever = EnsembleRetriever(
    retrievers=[self.bm25_retriever, semantic_retriever],
    weights=[0.5, 0.5]  # [BM25_weight, semantic_weight]
)
```

## 📊 Performance Optimization

### 1. Vector Store Persistence

Save processed embeddings to avoid recomputation:

```python
# Save vector store
rag.save_vector_store("./vector_store")

# Load vector store in subsequent runs
rag.load_vector_store("./vector_store")
```

### 2. Chunking Strategy

For better results with different document types:

- **Technical papers**: `chunk_size=1500, chunk_overlap=300`
- **Legal documents**: `chunk_size=800, chunk_overlap=150`
- **General text**: `chunk_size=1000, chunk_overlap=200` (default)

### 3. Retrieval Parameters

Adjust the number of retrieved documents:

```python
# In create_ensemble_retriever()
semantic_retriever = self.vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}  # Retrieve more documents
)

# In create_bm25_retriever()
self.bm25_retriever.k = 6  # Match semantic retriever
```

## 🔍 Troubleshooting

### Common Issues

**1. "Connection to Ollama failed"**
```bash
# Check if Ollama is running
ollama serve

# Verify the model is available
ollama list
```

**2. "No PDF files found"**
- Ensure PDF files are in the specified folder
- Check file permissions
- Verify file extensions are `.pdf`

**3. "CUDA out of memory" (if using GPU)**
```python
# Force CPU usage for embeddings
self.embeddings = HuggingFaceEmbeddings(
    model_name=self.embedding_model_name,
    model_kwargs={'device': 'cpu'},  # Force CPU
    encode_kwargs={'normalize_embeddings': True}
)
```

**4. "Empty documents after processing"**
- Check if PDFs contain extractable text (not just images)
- Try different PDF processing libraries if needed

### Performance Issues

**Slow embedding generation:**
- Use a smaller embedding model
- Process fewer documents at once
- Save vector store for reuse

**Slow LLM responses:**
- Use a smaller model (e.g., `gemma2:2b`)
- Reduce context size by retrieving fewer documents
- Increase Ollama's context window

### Memory Issues

For large document collections:

```python
# Process documents in batches
def process_large_collection(self, pdf_folder, batch_size=10):
    pdf_files = list(Path(pdf_folder).glob("*.pdf"))
    
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        # Process batch...
```

## 📚 Alternative Models

### Embedding Models

- `nomic-ai/nomic-embed-text-v1` (recommended)
- `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers/all-mpnet-base-v2`

### LLM Models

```bash
# Install different models
ollama pull gemma3:4b      # Recommended (good balance)
ollama pull gemma2:2b      # Faster, less capable
ollama pull llama3.1:8b    # Alternative good balance
ollama pull mistral:7b     # Alternative option
ollama pull codellama:7b   # For code-related documents
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use and modify as needed.

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Ensure Ollama is running with the correct model
4. Check that PDF files contain extractable text

## 📁 Project Structure

```
doksearch/
├── requirements.txt              # Python dependencies
├── rag_system.py                # Original RAG system (CLI)
├── enhanced_rag_system.py       # Enhanced RAG with incremental processing
├── web_ui.py                    # Streamlit web interface
├── example_usage.py             # Usage examples and demos
├── README.md                    # This documentation
├── vector_db/                   # Vector database storage (created automatically)
│   ├── metadata.json           # File processing metadata
│   └── vector_store/           # FAISS vector store files
└── pdfs/                       # Your PDF files (example folder)
```

## 🚀 Quick Start Script

Create a `start.bat` (Windows) or `start.sh` (Linux/Mac) file for easy launching:

**Windows (start.bat):**
```batch
@echo off
echo Starting Ollama...
start ollama serve
timeout /t 3
echo Pulling Gemma3:4b model...
ollama pull gemma3:4b
echo Starting RAG Web UI...
streamlit run web_ui.py
```

**Linux/Mac (start.sh):**
```bash
#!/bin/bash
echo "Starting Ollama..."
ollama serve &
sleep 3
echo "Pulling Gemma3:4b model..."
ollama pull gemma3:4b
echo "Starting RAG Web UI..."
streamlit run web_ui.py
```

## 🔮 Future Enhancements

- [x] Web interface for easier interaction ✅
- [x] Vector database persistence ✅
- [x] Incremental file processing ✅
- [x] Real-time analytics dashboard ✅
- [ ] Support for other document formats (DOCX, TXT, etc.)
- [ ] Multiple language support
- [ ] Document summarization features
- [ ] Integration with cloud embedding services
- [ ] Advanced chunking strategies based on document structure
- [ ] User authentication and multi-user support
- [ ] API endpoints for external integration
- [ ] Automated model downloading and management 