import streamlit as sl
import os
import tempfile
from rag_system import RAGSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
sl.set_page_config(
    page_title="RAG Document System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
sl.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@sl.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return RAGSystem()

def main():
    # Header
    sl.markdown('<h1 class="main-header">📚 RAG Document System</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = get_rag_system()
    
    # Sidebar
    sl.sidebar.title("🔧 System Controls")
    
    # System stats
    with sl.sidebar.expander("📊 System Statistics", expanded=False):
        stats = rag_system.get_system_stats()
        sl.write(f"**Vector Store Documents:** {stats['vector_store'].get('total_documents', 0)}")
        sl.write(f"**OpenAI Available:** {'✅' if stats['openai_available'] else '❌'}")
        sl.write(f"**Supported Formats:** {', '.join(stats['supported_extensions'])}")
    
    # Reset system
    if sl.sidebar.button("🗑️ Reset System"):
        if sl.sidebar.checkbox("Confirm reset"):
            success = rag_system.reset_system()
            if success:
                sl.sidebar.success("System reset successfully!")
                sl.rerun()
            else:
                sl.sidebar.error("Failed to reset system")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = sl.tabs(["📁 Upload Documents", "🌐 Web Scraping", "❓ Ask Questions", "📖 View Documents"])
    
    # Tab 1: Upload Documents
    with tab1:
        sl.header("📁 Upload Documents")
        
        # File upload
        uploaded_files = sl.file_uploader(
            "Choose PDF or DOC files",
            type=['pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if sl.button("🚀 Process Uploaded Files"):
                with sl.spinner("Processing documents..."):
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files
                        saved_files = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            saved_files.append(file_path)
                        
                        # Process documents
                        all_chunks = []
                        for file_path in saved_files:
                            chunks = rag_system.document_processor.process_document(file_path)
                            all_chunks.extend(chunks)
                        
                        if all_chunks:
                            # Add to vector store
                            success = rag_system.vector_store.add_documents(all_chunks)
                            if success:
                                sl.success(f"✅ Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} files!")
                                sl.rerun()
                            else:
                                sl.error("❌ Failed to add documents to vector store")
                        else:
                            sl.warning("⚠️ No text could be extracted from the uploaded files")
        
        # Directory processing
        sl.subheader("Or Process Local Directory")
        directory_path = sl.text_input("Enter directory path:", value="./docsJuly")
        
        # Document processing options
        processing_mode = sl.selectbox(
            "How to handle existing documents:",
            options=[
                ("add", "Add Only (Fail if exists)"),
                ("upsert", "Overwrite Existing"),
                ("skip_existing", "Skip Existing")
            ],
            format_func=lambda x: x[1],
            index=2  # Default to skip existing
        )
        selected_mode = processing_mode[0]
        
        if sl.button("📂 Process Directory"):
            if os.path.exists(directory_path):
                with sl.spinner("Processing directory..."):
                    result = rag_system.process_local_documents(directory_path, mode=selected_mode)
                    if result['success']:
                        sl.success("✅ Successfully processed directory!")
                        
                        # Show statistics
                        col1, col2, col3 = sl.columns(3)
                        with col1:
                            sl.metric("Chunks Processed", result['total_chunks_processed'])
                        with col2:
                            sl.metric("Total in Vector Store", result['vector_store_total'])
                        with col3:
                            sl.metric("Processing Mode", selected_mode)
                        
                        sl.rerun()
                    else:
                        sl.error(f"❌ Failed to process directory: {result.get('error', 'Unknown error')}")
            else:
                sl.error(f"❌ Directory not found: {directory_path}")
    
    # Tab 2: Web Scraping
    with tab2:
        sl.header("🌐 Web Scraping")
        
        # URL input
        url = sl.text_input("Enter website URL:", value="https://emma.msrb.org")
        max_docs = sl.slider("Maximum documents to download:", min_value=1, max_value=50, value=10)
        
        if sl.button("🕷️ Scrape Documents"):
            if url:
                with sl.spinner("Scraping documents..."):
                    success = rag_system.scrape_and_process_web_documents(url, max_docs)
                    if success:
                        sl.success("✅ Successfully scraped and processed documents!")
                        sl.rerun()
                    else:
                        sl.error("❌ Failed to scrape documents")
            else:
                sl.error("❌ Please enter a URL")
        
        # Show downloaded files
        downloaded_files = rag_system.web_scraper.get_downloaded_files()
        if downloaded_files:
            sl.subheader("📥 Downloaded Files")
            for file_path in downloaded_files:
                sl.write(f"• {os.path.basename(file_path)}")
    
    # Tab 3: Ask Questions
    with tab3:
        sl.header("❓ Ask Questions")
        
        # Question input
        question = sl.text_input("Enter your question:")
        top_k = sl.slider("Number of top results:", min_value=1, max_value=20, value=5)
        
        if sl.button("🔍 Search") and question:
            with sl.spinner("Searching for answers..."):
                result = rag_system.ask_question(question, top_k)
                
                # Display answer
                sl.subheader("💡 Answer")
                sl.write(result['answer'])
                
                # Display confidence
                col1, col2, col3 = sl.columns(3)
                with col1:
                    sl.metric("Confidence", f"{result['confidence']:.1%}")
                with col2:
                    sl.metric("Documents Found", result['search_results_count'])
                with col3:
                    sl.metric("Results Considered", top_k)
                
                # Display references
                if result['references']:
                    sl.subheader("📖 References")
                    for i, ref in enumerate(result['references'], 1):
                        with sl.expander(f"Reference {i}: {ref['file_name']} (Page {ref['page']})"):
                            sl.write(f"**Similarity:** {ref['similarity_score']:.2%}")
                            sl.write(f"**Folder:** {ref['folder']}")
                            sl.write(f"**Text:** {ref['text']}")
        
        # Quick questions
        sl.subheader("💭 Quick Questions")
        quick_questions = [
            "What is the main topic of these documents?",
            "What are the key findings?",
            "What are the important dates mentioned?",
            "Who are the main parties involved?"
        ]
        
        cols = sl.columns(2)
        for i, q in enumerate(quick_questions):
            with cols[i % 2]:
                if sl.button(q, key=f"quick_{i}"):
                    with sl.spinner("Searching..."):
                        result = rag_system.ask_question(q, 5)
                        sl.write(f"**Answer:** {result['answer']}")
    
    # Tab 4: View Documents
    with tab4:
        sl.header("📖 View Documents")
        
        # Show vector store contents
        stats = rag_system.get_system_stats()
        total_docs = stats['vector_store'].get('total_documents', 0)
        
        if total_docs > 0:
            sl.success(f"📚 Vector store contains {total_docs} document chunks")
            
            # Search within documents
            sl.subheader("🔍 Search Within Documents")
            search_query = sl.text_input("Search for specific content:")
            
            if search_query and sl.button("🔍 Search"):
                with sl.spinner("Searching..."):
                    results = rag_system.vector_store.search(search_query, 10)
                    
                    if results:
                        sl.write(f"Found {len(results)} relevant chunks:")
                        for i, result in enumerate(results, 1):
                            with sl.expander(f"Result {i}: {result['metadata']['file_name']} (Page {result['metadata']['page']})"):
                                sl.write(f"**Similarity:** {result['similarity_score']:.2%}")
                                sl.write(f"**Folder:** {result['metadata']['folder']}")
                                sl.write(f"**Text:** {result['text']}")
                    else:
                        sl.warning("No results found")
        else:
            sl.info("📝 No documents in the system yet. Upload some documents or scrape from a website first!")

if __name__ == "__main__":
    main() 