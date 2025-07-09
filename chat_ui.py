import streamlit as st
import os
import tempfile
from rag_system import RAGSystem
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        margin-right: 2rem;
    }
    .reference-box {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.3rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .high-confidence {
        background-color: #4caf50;
        color: white;
    }
    .medium-confidence {
        background-color: #ff9800;
        color: white;
    }
    .low-confidence {
        background-color: #f44336;
        color: white;
    }
    .stTextInput > div > div > input {
        border-radius: 1rem;
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        border-radius: 1rem;
        padding: 0.5rem 1rem;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    return RAGSystem()

def get_confidence_color(confidence):
    """Get color class based on confidence score."""
    if confidence >= 0.7:
        return "high-confidence"
    elif confidence >= 0.4:
        return "medium-confidence"
    else:
        return "low-confidence"

def format_references(references):
    """Format references for display."""
    if not references:
        return ""
    
    ref_text = ""
    for i, ref in enumerate(references, 1):
        ref_text += f"""
        <div class="reference-box">
            <strong>📄 Reference {i}: {ref['file_name']} (Page {ref['page']})</strong><br>
            <small>📁 Folder: {ref['folder']}</small><br>
            <small>🎯 Similarity: {ref['similarity_score']:.1%}</small><br>
            <em>"{ref['text']}"</em>
        </div>
        """
    return ref_text

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def process_documents(rag_system, directory_path):
    """Process documents and update session state."""
    with st.spinner("Processing documents..."):
        success = rag_system.process_local_documents(directory_path)
        if success:
            st.session_state.documents_processed = True
            st.success(f"✅ Successfully processed documents from {directory_path}")
            return True
        else:
            st.error("❌ Failed to process documents")
            return False

def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence_class = get_confidence_color(message['confidence'])
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong>
                <span class="confidence-badge {confidence_class}">
                    {message['confidence']:.1%} confidence
                </span><br>
                {message['answer']}
                {format_references(message['references'])}
            </div>
            """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">💬 RAG Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = get_rag_system()
    if not st.session_state.system_initialized:
        st.session_state.system_initialized = True
    
    # Sidebar for setup and controls
    with st.sidebar:
        st.title("🔧 Setup & Controls")
        
        # Document processing section
        with st.expander("📁 Document Setup", expanded=not st.session_state.documents_processed):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF/DOC files",
                type=['pdf', 'doc', 'docx'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("🚀 Process Uploaded Files"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        saved_files = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            saved_files.append(file_path)
                        
                        all_chunks = []
                        for file_path in saved_files:
                            chunks = rag_system.document_processor.process_document(file_path)
                            all_chunks.extend(chunks)
                        
                        if all_chunks:
                            success = rag_system.vector_store.add_documents(all_chunks)
                            if success:
                                st.session_state.documents_processed = True
                                st.success(f"✅ Processed {len(all_chunks)} chunks")
                                st.rerun()
                            else:
                                st.error("❌ Failed to add to vector store")
                        else:
                            st.warning("⚠️ No text extracted")
            
            # Directory processing
            st.subheader("Or Process Directory")
            directory_path = st.text_input("Directory path:", value="./docsJuly")
            
            if st.button("📂 Process Directory"):
                if os.path.exists(directory_path):
                    process_documents(rag_system, directory_path)
                    st.rerun()
                else:
                    st.error(f"❌ Directory not found: {directory_path}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Web scraping section
        with st.expander("🌐 Web Scraping"):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            url = st.text_input("Website URL:", value="https://emma.msrb.org")
            max_docs = st.slider("Max documents:", 1, 20, 5)
            
            if st.button("🕷️ Scrape Documents"):
                with st.spinner("Scraping..."):
                    success = rag_system.scrape_and_process_web_documents(url, max_docs)
                    if success:
                        st.session_state.documents_processed = True
                        st.success("✅ Scraped successfully")
                        st.rerun()
                    else:
                        st.error("❌ Scraping failed")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        with st.expander("📊 System Info"):
            stats = rag_system.get_system_stats()
            st.write(f"**Documents:** {stats['vector_store'].get('total_documents', 0)}")
            st.write(f"**OpenAI:** {'✅' if stats['openai_available'] else '❌'}")
            st.write(f"**Formats:** {', '.join(stats['supported_extensions'])}")
        
        # Controls
        with st.expander("⚙️ Controls"):
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("🔄 Reset System"):
                if st.checkbox("Confirm reset"):
                    success = rag_system.reset_system()
                    if success:
                        st.session_state.documents_processed = False
                        st.session_state.chat_history = []
                        st.success("✅ System reset")
                        st.rerun()
                    else:
                        st.error("❌ Reset failed")
    
    # Main chat area
    if not st.session_state.documents_processed:
        st.info("📝 Please upload documents or process a directory to start chatting!")
        st.stop()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    st.markdown("---")
    
    # Question input with suggestions
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="What is the main topic of these documents?",
            key="question_input"
        )
    
    with col2:
        top_k = st.selectbox("Results", [3, 5, 10, 15], index=1)
    
    # Quick question buttons
    st.subheader("💭 Quick Questions")
    quick_questions = [
        "What is the main topic?",
        "What are the key findings?",
        "Who are the parties involved?",
        "What are the important dates?",
        "What are the financial terms?",
        "What are the risks mentioned?"
    ]
    
    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, key=f"quick_{i}"):
                st.session_state.question_input = q
                st.rerun()
    
    # Process question
    if question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            result = rag_system.ask_question(question, top_k)
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'answer': result['answer'],
            'confidence': result['confidence'],
            'references': result['references'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Clear input and rerun to show new message
        st.rerun()

if __name__ == "__main__":
    main() 