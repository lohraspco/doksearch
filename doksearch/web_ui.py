#!/usr/bin/env python3
"""
Fast & Slick Streamlit Web UI for RAG System
Optimized for performance and user experience.
"""

import streamlit as st
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from streamlit_option_menu import option_menu

from enhanced_rag_system import EnhancedHybridRAGSystem

# Page configuration
st.set_page_config(
    page_title="⚡ Fast RAG System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"  # Start collapsed for speed
)

# Streamlined CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .sidebar .sidebar-content { background-color: #f8f9fa; }
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Efficient session state initialization
@st.cache_resource
def get_rag_system():
    """Cache the RAG system initialization."""
    return EnhancedHybridRAGSystem()

def init_session_state():
    """Initialize session state efficiently."""
    defaults = {
        'rag_system': None,
        'chat_history': [],
        'selected_folder': "",
        'system_ready': False,
        'last_stats_check': None,
        'cached_stats': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_cached_stats():
    """Get database stats with caching to avoid repeated calls."""
    now = datetime.now()
    
    # Cache stats for 10 seconds
    if (st.session_state.cached_stats is None or 
        st.session_state.last_stats_check is None or
        (now - st.session_state.last_stats_check).seconds > 10):
        
        if st.session_state.rag_system:
            try:
                st.session_state.cached_stats = st.session_state.rag_system.get_database_stats()
                st.session_state.last_stats_check = now
            except Exception as e:
                st.session_state.cached_stats = {"total_files": 0, "total_chunks": 0, "processed_files": []}
    
    return st.session_state.cached_stats or {"total_files": 0, "total_chunks": 0, "processed_files": []}

def show_quick_status():
    """Show quick system status in sidebar."""
    if st.session_state.rag_system:
        stats = get_cached_stats()
        if stats["total_files"] > 0:
            st.sidebar.success(f"📄 {stats['total_files']} files ready")
            st.sidebar.info(f"🧩 {stats['total_chunks']} chunks loaded")
        else:
            st.sidebar.warning("No documents loaded")
    else:
        st.sidebar.error("System not initialized")

def main():
    """Main application function."""
    init_session_state()
    
    # Fast header
    st.markdown('<div class="main-header">⚡ Fast RAG System</div>', unsafe_allow_html=True)
    
    # Streamlined sidebar
    with st.sidebar:
        st.markdown("### 🚀 Navigation")
        
        # Quick status
        show_quick_status()
        
        # Simplified menu
        selected = option_menu(
            "",
            ["🏠 Home", "📁 Documents", "💬 Chat", "📊 Stats"],
            icons=['house', 'folder', 'chat', 'graph-up'],
            menu_icon="cast",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"},
            }
        )
        
        # Quick actions
        if st.button("🔄 Refresh", key="refresh", help="Refresh system status"):
            st.session_state.cached_stats = None
            st.rerun()
    
    # Route to pages
    if selected == "🏠 Home":
        show_home()
    elif selected == "📁 Documents":
        show_documents()
    elif selected == "💬 Chat":
        show_chat()
    elif selected == "📊 Stats":
        show_stats()

def show_home():
    """Fast home page."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚀 Quick Start")
        
        # Initialize system button
        if st.session_state.rag_system is None:
            if st.button("⚡ Initialize System", type="primary", key="init_system"):
                with st.spinner("Initializing..."):
                    st.session_state.rag_system = get_rag_system()
                    st.session_state.system_ready = True
                st.success("System ready!")
                st.rerun()
        else:
            st.success("✅ System Ready")
            
            # Quick folder input
            folder_path = st.text_input(
                "📁 PDF Folder Path",
                value=st.session_state.selected_folder,
                placeholder="Enter folder path and hit Enter",
                key="quick_folder"
            )
            
            if folder_path and folder_path != st.session_state.selected_folder:
                st.session_state.selected_folder = folder_path
                
                if os.path.exists(folder_path):
                    pdf_files = list(Path(folder_path).glob("*.pdf"))
                    if pdf_files:
                        st.info(f"Found {len(pdf_files)} PDF files")
                        
                        if st.button("⚡ Process Now", type="primary", key="quick_process"):
                            process_folder_fast(folder_path)
                    else:
                        st.warning("No PDF files found")
                else:
                    st.error("Folder not found")
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        if st.session_state.rag_system:
            stats = get_cached_stats()
            
            # Compact metrics
            st.metric("Files", stats["total_files"])
            st.metric("Chunks", stats["total_chunks"])
            
            if stats["total_files"] > 0:
                st.success("Ready to chat!")
            else:
                st.info("Add documents to start")

def show_documents():
    """Fast document management."""
    st.markdown("### 📁 Document Management")
    
    # Initialize system if needed
    if st.session_state.rag_system is None:
        if st.button("⚡ Initialize System", type="primary"):
            with st.spinner("Initializing..."):
                st.session_state.rag_system = get_rag_system()
            st.success("System ready!")
            st.rerun()
        return
    
    # Folder selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        folder_path = st.text_input(
            "PDF Folder Path",
            value=st.session_state.selected_folder,
            placeholder="Enter the full path to your PDF folder"
        )
    
    with col2:
        st.write("")  # Spacer
        force_reprocess = st.checkbox("🔄 Force Reprocess")
    
    if folder_path:
        st.session_state.selected_folder = folder_path
        
        if os.path.exists(folder_path):
            pdf_files = list(Path(folder_path).glob("*.pdf"))
            
            if pdf_files:
                st.success(f"✅ Found {len(pdf_files)} PDF files")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("⚡ Process Documents", type="primary", key="process_docs"):
                        process_folder_fast(folder_path, force_reprocess)
                
                with col2:
                    if st.button("🗑️ Clear Database", key="clear_db"):
                        clear_database_fast()
                
                # Show processed files (compact view)
                show_processed_files_compact()
            else:
                st.warning("No PDF files found in folder")
        else:
            st.error("Folder does not exist")

def show_chat():
    """Fast chat interface."""
    st.markdown("### 💬 Chat Interface")
    
    if not st.session_state.rag_system:
        st.warning("Please initialize system first")
        return
    
    stats = get_cached_stats()
    if stats["total_files"] == 0:
        st.warning("Please process some documents first")
        return
    
    # Chat input
    question = st.chat_input("Ask a question about your documents...")
    
    if question:
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.ask_question(question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}",
                    "sources": []
                })
    
    # Display chat history (optimized)
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Compact source display
                if message.get("sources"):
                    with st.expander(f"📚 Sources ({len(message['sources'])})", expanded=False):
                        for j, source in enumerate(message["sources"][:3], 1):  # Limit to 3 sources
                            st.markdown(f"**{j}. {source['filename']}**")
                            st.text(source["content_preview"][:150] + "..." if len(source["content_preview"]) > 150 else source["content_preview"])
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def show_stats():
    """Fast statistics page."""
    st.markdown("### 📊 Statistics")
    
    if not st.session_state.rag_system:
        st.warning("Please initialize system first")
        return
    
    stats = get_cached_stats()
    processed_files = stats.get("processed_files", [])
    
    if not processed_files:
        st.info("No data available. Process some documents first!")
        return
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Files", stats["total_files"])
    with col2:
        st.metric("🧩 Total Chunks", stats["total_chunks"])
    with col3:
        avg_chunks = stats["total_chunks"] / stats["total_files"] if stats["total_files"] > 0 else 0
        st.metric("📊 Avg Chunks/File", f"{avg_chunks:.1f}")
    with col4:
        total_size = sum(f.get("size", 0) for f in processed_files)
        st.metric("💾 Total Size", format_file_size(total_size))
    
    # Simple charts
    if len(processed_files) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 File Sizes")
            size_data = pd.DataFrame({
                "File": [f["filename"] for f in processed_files],
                "Size": [f.get("size", 0) for f in processed_files]
            })
            st.bar_chart(size_data.set_index("File"))
        
        with col2:
            st.markdown("#### 🧩 Chunk Counts")
            chunk_data = pd.DataFrame({
                "File": [f["filename"] for f in processed_files],
                "Chunks": [f["chunk_count"] for f in processed_files]
            })
            st.bar_chart(chunk_data.set_index("File"))

def process_folder_fast(folder_path: str, force_reprocess: bool = False):
    """Fast document processing with minimal UI updates."""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        status.text("⚡ Processing documents...")
        progress.progress(20)
        
        # Process documents
        result = st.session_state.rag_system.load_pdf_folder(folder_path, force_reprocess)
        progress.progress(80)
        
        # Clear cache
        st.session_state.cached_stats = None
        progress.progress(100)
        
        # Show result
        if result["new_files_processed"]:
            st.success(f"✅ Processed {result['total_files']} files successfully!")
        else:
            st.info("ℹ️ No new files to process")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        progress.empty()
        status.empty()

def clear_database_fast():
    """Fast database clearing."""
    try:
        with st.spinner("Clearing database..."):
            st.session_state.rag_system.clear_vector_database()
            st.session_state.cached_stats = None
        st.success("✅ Database cleared!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_processed_files_compact():
    """Show processed files in compact view."""
    stats = get_cached_stats()
    processed_files = stats.get("processed_files", [])
    
    if processed_files:
        st.markdown("#### 📚 Processed Files")
        
        # Compact table
        df_data = []
        for file_info in processed_files:
            df_data.append({
                "File": file_info["filename"],
                "Chunks": file_info["chunk_count"],
                "Size": format_file_size(file_info.get("size", 0)),
                "Status": "✅" if file_info.get("exists", True) else "❌"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

def format_file_size(size_bytes: int) -> str:
    """Format file size efficiently."""
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB']
    for unit in units:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

if __name__ == "__main__":
    main() 