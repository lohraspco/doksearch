import streamlit as st
import os
import tempfile
from rag_system import RAGSystem
import logging
from datetime import datetime
import json
import pandas as pd
from logging_config import setup_logging, get_logger

# Set up comprehensive logging
loggers = setup_logging(log_level=logging.INFO, log_file="advanced_chat.log")
logger = get_logger('advanced_chat')

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced chat interface
st.markdown("""
<style>
    .chat-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 3rem;
        border-bottom-right-radius: 0.3rem;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 3rem;
        border-bottom-left-radius: 0.3rem;
    }
    .reference-box {
        background: rgba(255,255,255,0.9);
        color: #333;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 2rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        backdrop-filter: blur(10px);
    }
    .high-confidence {
        background: rgba(76, 175, 80, 0.9);
        color: white;
    }
    .medium-confidence {
        background: rgba(255, 152, 0, 0.9);
        color: white;
    }
    .low-confidence {
        background: rgba(244, 67, 54, 0.9);
        color: white;
    }
    .chat-input {
        background: white;
        border-radius: 2rem;
        padding: 1rem 1.5rem;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    .chat-input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    .quick-question-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 1rem;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .quick-question-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .sidebar-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        color: #666;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_system():
    """Initialize and cache the RAG system."""
    logger.info("🔄 Creating RAG system instance (cached)")
    rag_system = RAGSystem()
    logger.info("✅ RAG system instance created and cached")
    return rag_system

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
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'chat_settings' not in st.session_state:
        st.session_state.chat_settings = {
            'top_k': 5,
            'temperature': 0.3,
            'max_tokens': 500,
            'show_references': True,
            'show_confidence': True
        }
    
    # Check if documents are already processed by checking vector store
    if 'documents_processed' not in st.session_state:
        try:
            has_docs, doc_count = check_documents_status()
            st.session_state.documents_processed = has_docs
            if st.session_state.documents_processed:
                logger.info(f"Auto-detected {doc_count} existing documents in vector store")
        except Exception as e:
            logger.warning(f"Could not check existing documents: {e}")
            st.session_state.documents_processed = False

def export_conversation():
    """Export conversation to JSON."""
    if st.session_state.chat_history:
        conversation_data = {
            'conversation_id': st.session_state.conversation_id,
            'export_date': datetime.now().isoformat(),
            'messages': st.session_state.chat_history
        }
        return json.dumps(conversation_data, indent=2)
    return None

def display_chat_history():
    """Display the chat history with advanced styling."""
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message['content']}
                <small style="opacity: 0.8; display: block; margin-top: 0.5rem;">
                    {datetime.fromisoformat(message['timestamp']).strftime('%H:%M')}
                </small>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence_class = get_confidence_color(message['confidence'])
            confidence_display = f"""
            <span class="confidence-badge {confidence_class}">
                {message['confidence']:.1%} confidence
            </span>
            """ if st.session_state.chat_settings['show_confidence'] else ""
            
            references_display = format_references(message['references']) if st.session_state.chat_settings['show_references'] else ""
            
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong>
                {confidence_display}<br>
                {message['answer']}
                {references_display}
                <small style="opacity: 0.8; display: block; margin-top: 0.5rem;">
                    {datetime.fromisoformat(message['timestamp']).strftime('%H:%M')}
                </small>
            </div>
            """, unsafe_allow_html=True)

def show_typing_indicator():
    """Show typing indicator."""
    st.markdown("""
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <span>Assistant is thinking...</span>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def check_documents_status():
    """Check if documents are available in the vector store."""
    try:
        from vector_store import VectorStore
        vs = VectorStore()
        stats = vs.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        return total_docs > 0, total_docs
    except Exception as e:
        logger.error(f"Error checking documents status: {e}")
        return False, 0

def refresh_documents_status():
    """Refresh the documents processed status."""
    has_docs, doc_count = check_documents_status()
    st.session_state.documents_processed = has_docs
    return has_docs, doc_count

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Advanced RAG Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system (cached)
    if not st.session_state.system_initialized:
        logger.info("🔄 Initializing RAG system for the first time")
        rag_system = get_rag_system()
        st.session_state.system_initialized = True
        logger.info("✅ RAG system initialized and marked as ready")
    else:
        rag_system = get_rag_system()
    
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
            
            # Document processing options
            st.subheader("📋 Processing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                processing_mode = st.selectbox(
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
            
            with col2:
                st.write("**Mode Explanation:**")
                if selected_mode == "add":
                    st.info("Will fail if documents already exist")
                elif selected_mode == "upsert":
                    st.info("Will overwrite existing documents")
                else:  # skip_existing
                    st.info("Will only add new documents")
            
            if st.button("📂 Process Directory"):
                if os.path.exists(directory_path):
                    with st.spinner("Processing documents..."):
                        result = rag_system.process_local_documents(directory_path, mode=selected_mode)
                        
                        if result['success']:
                            st.session_state.documents_processed = True
                            
                            # Display detailed results
                            st.success(f"✅ Successfully processed documents from {directory_path}")
                            
                            # Show statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Chunks Processed", result['total_chunks_processed'])
                            with col2:
                                st.metric("Total in Vector Store", result['vector_store_total'])
                            with col3:
                                st.metric("Processing Mode", selected_mode)
                            
                            # Show existing document info if available
                            if result.get('existing_info'):
                                existing_info = result['existing_info']
                                with st.expander("📊 Document Analysis"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Existing Files:** {len(existing_info['existing_files'])}")
                                        st.write(f"**New Files:** {len(existing_info['new_files'])}")
                                    with col2:
                                        st.write(f"**Existing Chunks:** {existing_info['existing_chunks']}")
                                        st.write(f"**New Chunks:** {existing_info['new_chunks']}")
                                    
                                    if existing_info['existing_files']:
                                        st.write("**Existing Files:**")
                                        for file in existing_info['existing_files'][:5]:  # Show first 5
                                            st.write(f"• {os.path.basename(file)}")
                                        if len(existing_info['existing_files']) > 5:
                                            st.write(f"... and {len(existing_info['existing_files']) - 5} more")
                                    
                                    if existing_info['new_files']:
                                        st.write("**New Files:**")
                                        for file in existing_info['new_files'][:5]:  # Show first 5
                                            st.write(f"• {os.path.basename(file)}")
                                        if len(existing_info['new_files']) > 5:
                                            st.write(f"... and {len(existing_info['new_files']) - 5} more")
                            
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to process documents: {result.get('error', 'Unknown error')}")
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
        
        # Chat settings
        with st.expander("⚙️ Chat Settings"):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            st.session_state.chat_settings['top_k'] = st.slider("Top K results:", 1, 20, 5)
            st.session_state.chat_settings['show_references'] = st.checkbox("Show references", True)
            st.session_state.chat_settings['show_confidence'] = st.checkbox("Show confidence", True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # System info
        with st.expander("📊 System Info"):
            stats = rag_system.get_system_stats()
            
            # Check current document status
            has_docs, doc_count = check_documents_status()
            status_icon = "✅" if has_docs else "❌"
            
            st.markdown(f"""
            <div class="stats-card">
                <strong>📄 Documents:</strong> {doc_count} {status_icon}
            </div>
            <div class="stats-card">
                <strong>🤖 OpenAI:</strong> {'✅' if stats['openai_available'] else '❌'}
            </div>
            <div class="stats-card">
                <strong>📁 Formats:</strong> {', '.join(stats['supported_extensions'])}
            </div>
            """, unsafe_allow_html=True)
            
            # Add refresh button for system info
            if st.button("🔄 Refresh System Info", key="refresh_sys_info"):
                st.rerun()
        
        # Conversation management
        with st.expander("💾 Conversation"):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            st.write(f"**Conversation ID:** {st.session_state.conversation_id}")
            st.write(f"**Messages:** {len(st.session_state.chat_history)}")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Export conversation
            if st.session_state.chat_history:
                conversation_json = export_conversation()
                st.download_button(
                    label="📥 Export Conversation",
                    data=conversation_json,
                    file_name=f"conversation_{st.session_state.conversation_id}.json",
                    mime="application/json"
                )
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area with tabs
    if not st.session_state.documents_processed:
        st.info("📝 Please upload documents or process a directory to start chatting!")
        
        # Add a refresh button to check for existing documents
        if st.button("🔄 Check for Existing Documents"):
            has_docs, doc_count = refresh_documents_status()
            if has_docs:
                st.success(f"✅ Found {doc_count} existing documents! You can now start chatting.")
                st.rerun()
            else:
                st.warning("No existing documents found. Please upload or process documents.")
        
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📚 Documents"])
    
    # Chat Tab
    with tab1:
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
            send_button = st.button("🚀 Send", use_container_width=True)
        
        # Quick question buttons
        st.subheader("💭 Quick Questions")
        quick_questions = [
            "What is the main topic?",
            "What are the key findings?",
            "Who are the parties involved?",
            "What are the important dates?",
            "What are the financial terms?",
            "What are the risks mentioned?",
            "What are the obligations?",
            "What are the deadlines?"
        ]
        
        cols = st.columns(4)
        for i, q in enumerate(quick_questions):
            with cols[i % 4]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.question_input = q
                    st.rerun()
        
        # Process question
        if question and (send_button or st.session_state.question_input == question):
            # Check if this question was already processed
            if not st.session_state.chat_history or st.session_state.chat_history[-1].get('content') != question:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Show typing indicator
                with st.container():
                    show_typing_indicator()
                
                # Get answer
                result = rag_system.ask_question(question, st.session_state.chat_settings['top_k'])
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'references': result['references'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Clear the question input to prevent infinite loop
                st.session_state.question_input = ""
                
                # Clear input and rerun to show new message
                st.rerun()
    
    # Documents Tab
    with tab2:
        st.header("📚 Processed Documents")
        
        # Get system stats
        stats = rag_system.get_system_stats()
        total_docs = stats['vector_store'].get('total_documents', 0)
        
        if total_docs == 0:
            st.info("📝 No documents have been processed yet. Please upload or process documents first.")
        else:
            # Document overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", total_docs)
            with col2:
                st.metric("Embedding Model", stats['vector_store'].get('embedding_model', 'Unknown'))
            with col3:
                st.metric("Collection", stats['vector_store'].get('collection_name', 'Unknown'))
            
            st.markdown("---")
            
            # Document search and browsing
            st.subheader("🔍 Search Documents")
            
            # Search interface
            search_query = st.text_input(
                "Search in documents:",
                placeholder="Enter keywords to search in your documents...",
                key="doc_search"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                search_button = st.button("🔍 Search", use_container_width=True)
            with col2:
                show_all_button = st.button("📋 Show All", use_container_width=True)
            
            # Display search results
            if search_button and search_query:
                with st.spinner("Searching documents..."):
                    search_results = rag_system.vector_store.search(search_query, top_k=20)
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} relevant documents")
                        
                        for i, result in enumerate(search_results, 1):
                            # Get metadata from the result structure
                            metadata = result.get('metadata', {})
                            file_name = metadata.get('file_name', 'Unknown')
                            page = metadata.get('page', 'Unknown')
                            folder = metadata.get('folder', 'Unknown')
                            
                            with st.expander(f"📄 {file_name} (Page {page}) - Similarity: {result['similarity_score']:.1%}"):
                                st.write(f"**Folder:** {folder}")
                                st.write(f"**Text:**")
                                st.text_area(f"Content {i}", result['text'], height=150, key=f"content_{i}")
                    else:
                        st.warning("No documents found matching your search.")
            
            elif show_all_button:
                with st.spinner("Loading all documents..."):
                    # Get all documents from vector store
                    try:
                        all_docs = rag_system.vector_store.get_all_documents()
                        
                        if all_docs:
                            st.success(f"Showing all {len(all_docs)} documents")
                            
                            # Group by file
                            files = {}
                            for doc in all_docs:
                                file_key = f"{doc['folder']}/{doc['file_name']}"
                                if file_key not in files:
                                    files[file_key] = []
                                files[file_key].append(doc)
                            
                            # Display grouped by file
                            for file_key, docs in files.items():
                                with st.expander(f"📄 {file_key} ({len(docs)} chunks)"):
                                    for i, doc in enumerate(docs[:10]):  # Show first 10 chunks per file
                                        st.write(f"**Page {doc['page']}:**")
                                        st.text_area(f"Chunk {i+1}", doc['text'], height=100, key=f"all_{file_key}_{i}")
                                    
                                    if len(docs) > 10:
                                        st.info(f"... and {len(docs) - 10} more chunks")
                        else:
                            st.warning("No documents found in the vector store.")
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
                        st.info("This feature requires the vector store to support document retrieval.")
            
            # Document statistics
            st.markdown("---")
            st.subheader("📊 Document Statistics")
            
            # Show some basic stats
            st.write(f"**Total Documents Processed:** {total_docs}")
            st.write(f"**Vector Store:** {stats['vector_store'].get('collection_name', 'Unknown')}")
            st.write(f"**Embedding Model:** {stats['vector_store'].get('embedding_model', 'Unknown')}")
            
            # Document processing info
            st.markdown("---")
            st.subheader("🔄 Document Management")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh Status"):
                    has_docs, doc_count = refresh_documents_status()
                    if has_docs:
                        st.success(f"✅ Found {doc_count} documents")
                    else:
                        st.warning("No documents found")
                    st.rerun()
            
            with col2:
                if st.button("📊 Update Stats"):
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Clear Documents"):
                    if st.checkbox("Confirm clear all documents"):
                        success = rag_system.reset_system()
                        if success:
                            st.session_state.documents_processed = False
                            st.success("✅ All documents cleared")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear documents")

if __name__ == "__main__":
    main() 