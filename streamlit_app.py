import streamlit as st
import os
import tempfile
from rag_system import RAGSystem
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Document System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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

@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return RAGSystem()

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document System</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = get_rag_system()
    
    # Sidebar
    st.sidebar.title("🔧 System Controls")
    
    # System stats
    with st.sidebar.expander("📊 System Statistics", expanded=False):
        stats = rag_system.get_system_stats()
        st.write(f"**Vector Store Documents:** {stats['vector_store'].get('total_documents', 0)}")
        st.write(f"**OpenAI Available:** {'✅' if stats['openai_available'] else '❌'}")
        st.write(f"**Supported Formats:** {', '.join(stats['supported_extensions'])}")
    
    # Reset system
    if st.sidebar.button("🗑️ Reset System"):
        if st.sidebar.checkbox("Confirm reset"):
            success = rag_system.reset_system()
            if success:
                st.sidebar.success("System reset successfully!")
                st.rerun()
            else:
                st.sidebar.error("Failed to reset system")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Documents", "🌐 Web Scraping", "❓ Ask Questions", "📖 View Documents"])
    
    # Tab 1: Upload Documents
    with tab1:
        st.header("📁 Upload Documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF or DOC files",
            type=['pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("🚀 Process Uploaded Files"):
                with st.spinner("Processing documents..."):
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
                                st.success(f"✅ Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} files!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to add documents to vector store")
                        else:
                            st.warning("⚠️ No text could be extracted from the uploaded files")
        
        # Directory processing
        st.subheader("Or Process Local Directory")
        directory_path = st.text_input("Enter directory path:", value="./docsJuly")
        
        if st.button("📂 Process Directory"):
            if os.path.exists(directory_path):
                with st.spinner("Processing directory..."):
                    success = rag_system.process_local_documents(directory_path)
                    if success:
                        st.success("✅ Successfully processed directory!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process directory")
            else:
                st.error(f"❌ Directory not found: {directory_path}")
    
    # Tab 2: Web Scraping
    with tab2:
        st.header("🌐 Web Scraping")
        
        # URL input
        url = st.text_input("Enter website URL:", value="https://emma.msrb.org")
        max_docs = st.slider("Maximum documents to download:", min_value=1, max_value=50, value=10)
        
        if st.button("🕷️ Scrape Documents"):
            if url:
                with st.spinner("Scraping documents..."):
                    success = rag_system.scrape_and_process_web_documents(url, max_docs)
                    if success:
                        st.success("✅ Successfully scraped and processed documents!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to scrape documents")
            else:
                st.error("❌ Please enter a URL")
        
        # Show downloaded files
        downloaded_files = rag_system.web_scraper.get_downloaded_files()
        if downloaded_files:
            st.subheader("📥 Downloaded Files")
            for file_path in downloaded_files:
                st.write(f"• {os.path.basename(file_path)}")
    
    # Tab 3: Ask Questions
    with tab3:
        st.header("❓ Ask Questions")
        
        # Question input
        question = st.text_input("Enter your question:")
        top_k = st.slider("Number of top results:", min_value=1, max_value=20, value=5)
        
        if st.button("🔍 Search") and question:
            with st.spinner("Searching for answers..."):
                result = rag_system.ask_question(question, top_k)
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                # Display confidence
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col2:
                    st.metric("Documents Found", result['search_results_count'])
                with col3:
                    st.metric("Results Considered", top_k)
                
                # Display references
                if result['references']:
                    st.subheader("📖 References")
                    for i, ref in enumerate(result['references'], 1):
                        with st.expander(f"Reference {i}: {ref['file_name']} (Page {ref['page']})"):
                            st.write(f"**Similarity:** {ref['similarity_score']:.2%}")
                            st.write(f"**Folder:** {ref['folder']}")
                            st.write(f"**Text:** {ref['text']}")
        
        # Quick questions
        st.subheader("💭 Quick Questions")
        quick_questions = [
            "What is the main topic of these documents?",
            "What are the key findings?",
            "What are the important dates mentioned?",
            "Who are the main parties involved?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(q, key=f"quick_{i}"):
                    with st.spinner("Searching..."):
                        result = rag_system.ask_question(q, 5)
                        st.write(f"**Answer:** {result['answer']}")
    
    # Tab 4: View Documents
    with tab4:
        st.header("📖 View Documents")
        
        # Show vector store contents
        stats = rag_system.get_system_stats()
        total_docs = stats['vector_store'].get('total_documents', 0)
        
        if total_docs > 0:
            st.success(f"📚 Vector store contains {total_docs} document chunks")
            
            # Search within documents
            st.subheader("🔍 Search Within Documents")
            search_query = st.text_input("Search for specific content:")
            
            if search_query and st.button("🔍 Search"):
                with st.spinner("Searching..."):
                    results = rag_system.vector_store.search(search_query, 10)
                    
                    if results:
                        st.write(f"Found {len(results)} relevant chunks:")
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i}: {result['metadata']['file_name']} (Page {result['metadata']['page']})"):
                                st.write(f"**Similarity:** {result['similarity_score']:.2%}")
                                st.write(f"**Folder:** {result['metadata']['folder']}")
                                st.write(f"**Text:** {result['text']}")
                    else:
                        st.warning("No results found")
        else:
            st.info("📝 No documents in the system yet. Upload some documents or scrape from a website first!")

if __name__ == "__main__":
    main() 