#!/usr/bin/env python3
"""
Streamlit Web UI for RAG System with Hybrid Ensemble Retriever
Features:
- Folder selection for PDFs
- Incremental processing (skip already processed files)
- Vector database management (clear, view stats)
- Document tracking and display
- Interactive Q&A interface
"""

import streamlit as sl
import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
from streamlit_option_menu import option_menu

from enhanced_rag_system import EnhancedHybridRAGSystem

# Page configuration
sl.set_page_config(
    page_title="RAG System with Hybrid Retriever",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
sl.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in sl.session_state:
        sl.session_state.rag_system = None
    if 'processing_log' not in sl.session_state:
        sl.session_state.processing_log = []
    if 'chat_history' not in sl.session_state:
        sl.session_state.chat_history = []
    if 'selected_folder' not in sl.session_state:
        sl.session_state.selected_folder = ""

def add_to_log(message: str, log_type: str = "info"):
    """Add a message to the processing log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    sl.session_state.processing_log.append({
        "timestamp": timestamp,
        "message": message,
        "type": log_type
    })

def display_log():
    """Display the processing log."""
    if sl.session_state.processing_log:
        sl.subheader("📋 Processing Log")
        for entry in sl.session_state.processing_log[-10:]:  # Show last 10 entries
            if entry["type"] == "error":
                sl.error(f"[{entry['timestamp']}] {entry['message']}")
            elif entry["type"] == "warning":
                sl.warning(f"[{entry['timestamp']}] {entry['message']}")
            elif entry["type"] == "success":
                sl.success(f"[{entry['timestamp']}] {entry['message']}")
            else:
                sl.info(f"[{entry['timestamp']}] {entry['message']}")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_system_health():
    """Get system health status."""
    if sl.session_state.rag_system:
        return sl.session_state.rag_system.health_check()
    return {
        "ollama_connection": False,
        "embeddings_loaded": False,
        "vector_store_exists": False,
        "system_initialized": False
    }

def main():
    """Main application function."""
    init_session_state()
    
    # Header
    sl.markdown('<div class="main-header">🔍 RAG System with Hybrid Retriever</div>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    with sl.sidebar:
        sl.image("https://via.placeholder.com/200x100/1f77b4/white?text=RAG+System", 
                caption="Hybrid Ensemble Retriever")
        
        selected = option_menu(
            "Navigation",
            ["🏠 Home", "📁 Document Management", "💬 Chat Interface", "📊 Analytics", "⚙️ Settings"],
            icons=['house', 'folder', 'chat', 'graph-up', 'gear'],
            menu_icon="cast",
            default_index=0,
        )
        
        # System health in sidebar
        sl.divider()
        sl.subheader("🏥 System Health")
        health = get_system_health()
        
        for component, status in health.items():
            status_text = "🟢 Connected" if status else "🔴 Disconnected"
            status_class = "status-success" if status else "status-error"
            sl.markdown(f'<div class="{status_class}">{component.replace("_", " ").title()}: {status_text}</div>', 
                       unsafe_allow_html=True)
    
    # Main content based on selection
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "📁 Document Management":
        show_document_management()
    elif selected == "💬 Chat Interface":
        show_chat_interface()
    elif selected == "📊 Analytics":
        show_analytics()
    elif selected == "⚙️ Settings":
        show_settings()

def show_home_page():
    """Display the home page."""
    sl.header("Welcome to the RAG System")
    
    col1, col2 = sl.columns(2)
    
    with col1:
        sl.subheader("🚀 Quick Start")
        sl.markdown("""
        1. **📁 Select PDF Folder**: Choose a folder containing your PDF documents
        2. **⚡ Process Documents**: The system will extract text and create embeddings
        3. **💬 Ask Questions**: Use natural language to query your documents
        4. **📊 View Results**: Get answers with source references
        """)
        
        if sl.button("🚀 Get Started", type="primary", use_container_width=True):
            sl.switch_page("Document Management")
    
    with col2:
        sl.subheader("✨ Features")
        sl.markdown("""
        - **🔄 Hybrid Retrieval**: Combines BM25 + Semantic search
        - **⚡ Incremental Processing**: Skip already processed files
        - **💾 Persistent Storage**: Reuse vector database
        - **🧠 Local LLM**: Uses Gemma3:4b via Ollama
        - **🏠 Local Embeddings**: No API calls needed
        - **📈 Analytics**: View processing statistics
        """)
    
    # Quick stats if system is initialized
    if sl.session_state.rag_system:
        sl.divider()
        stats = sl.session_state.rag_system.get_database_stats()
        
        col1, col2, col3, col4 = sl.columns(4)
        with col1:
            sl.metric("📄 Total Files", stats["total_files"])
        with col2:
            sl.metric("🧩 Total Chunks", stats["total_chunks"])
        with col3:
            sl.metric("💾 Database Size", format_file_size(stats["database_size"]))
        with col4:
            if stats["last_updated"]:
                last_updated = datetime.fromisoformat(stats["last_updated"])
                sl.metric("🕒 Last Updated", last_updated.strftime("%Y-%m-%d"))

def show_document_management():
    """Display document management interface."""
    sl.header("📁 Document Management")
    
    # Initialize RAG system if not done
    if sl.session_state.rag_system is None:
        with sl.spinner("🔧 Initializing RAG system..."):
            try:
                sl.session_state.rag_system = EnhancedHybridRAGSystem()
                add_to_log("RAG system initialized successfully", "success")
            except Exception as e:
                sl.error(f"Failed to initialize RAG system: {e}")
                add_to_log(f"Failed to initialize RAG system: {e}", "error")
                return
    
    # Folder selection
    sl.subheader("📂 Select PDF Folder")
    
    col1, col2 = sl.columns([3, 1])
    
    with col1:
        folder_path = sl.text_input(
            "Folder Path",
            value=sl.session_state.selected_folder,
            placeholder="Enter the path to your PDF folder (e.g., C:/Users/username/Documents/PDFs)",
            help="Enter the full path to the folder containing your PDF files"
        )
    
    with col2:
        sl.markdown("<br>", unsafe_allow_html=True)  # Spacer
        if sl.button("📁 Browse", help="Click to browse for folder"):
            sl.info("💡 Tip: Copy and paste the folder path from your file explorer")
    
    if folder_path and folder_path != sl.session_state.selected_folder:
        sl.session_state.selected_folder = folder_path
    
    # Check if folder exists and contains PDFs
    if folder_path:
        if os.path.exists(folder_path):
            pdf_files = list(Path(folder_path).glob("*.pdf"))
            if pdf_files:
                sl.success(f"✅ Found {len(pdf_files)} PDF files in the selected folder")
                
                # Show some file names
                with sl.expander("📄 Preview Files in Folder"):
                    for i, pdf_file in enumerate(pdf_files[:10]):  # Show first 10
                        file_size = format_file_size(pdf_file.stat().st_size)
                        sl.text(f"{pdf_file.name} ({file_size})")
                    if len(pdf_files) > 10:
                        sl.text(f"... and {len(pdf_files) - 10} more files")
                
                # Processing options
                col1, col2, col3 = sl.columns(3)
                
                with col1:
                    force_reprocess = sl.checkbox(
                        "🔄 Force Reprocess All Files",
                        help="Process all files again, even if they were already processed"
                    )
                
                with col2:
                    if sl.button("⚡ Process Documents", type="primary", use_container_width=True):
                        process_documents(folder_path, force_reprocess)
                
                with col3:
                    if sl.button("🗑️ Clear Vector Database", use_container_width=True):
                        if sl.session_state.get('confirm_clear', False):
                            clear_vector_database()
                            sl.session_state.confirm_clear = False
                        else:
                            sl.session_state.confirm_clear = True
                            sl.warning("⚠️ Click again to confirm clearing the database")
                
            else:
                sl.warning("⚠️ No PDF files found in the selected folder")
        else:
            sl.error("❌ Folder path does not exist")
    
    # Display processed documents
    show_processed_documents()
    
    # Display processing log
    display_log()

def process_documents(folder_path: str, force_reprocess: bool = False):
    """Process documents in the selected folder."""
    if not sl.session_state.rag_system:
        sl.error("RAG system not initialized")
        return
    
    add_to_log(f"Starting document processing for: {folder_path}")
    
    progress_bar = sl.progress(0)
    status_text = sl.empty()
    
    try:
        with sl.spinner("🔍 Processing documents..."):
            status_text.text("🔍 Processing documents...")
            progress_bar.progress(25)
            
            # Process documents
            result = sl.session_state.rag_system.load_pdf_folder(folder_path, force_reprocess)
            progress_bar.progress(75)
            
            status_text.text("✅ Processing completed!")
            progress_bar.progress(100)
            
            # Show results
            if result["new_files_processed"]:
                sl.success(f"✅ Successfully processed documents!")
                add_to_log(f"Successfully processed {result['total_files']} files", "success")
            else:
                sl.info("ℹ️ No new files to process (all files already processed)")
                add_to_log("No new files to process", "info")
            
            # Display summary
            col1, col2, col3 = sl.columns(3)
            with col1:
                sl.metric("📄 Total Files", result["total_files"])
            with col2:
                sl.metric("🧩 Total Chunks", result["total_chunks"])
            with col3:
                sl.metric("🆕 New Files", "Yes" if result["new_files_processed"] else "No")
            
            time.sleep(1)  # Brief pause for user to see completion
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        sl.error(f"❌ Error processing documents: {e}")
        add_to_log(f"Error processing documents: {e}", "error")
        progress_bar.empty()
        status_text.empty()

def clear_vector_database():
    """Clear the vector database."""
    if not sl.session_state.rag_system:
        sl.error("RAG system not initialized")
        return
    
    try:
        with sl.spinner("🗑️ Clearing vector database..."):
            success = sl.session_state.rag_system.clear_vector_database()
            
        if success:
            sl.success("✅ Vector database cleared successfully!")
            add_to_log("Vector database cleared", "success")
        else:
            sl.error("❌ Failed to clear vector database")
            add_to_log("Failed to clear vector database", "error")
            
    except Exception as e:
        sl.error(f"❌ Error clearing database: {e}")
        add_to_log(f"Error clearing database: {e}", "error")

def show_processed_documents():
    """Display list of processed documents."""
    if not sl.session_state.rag_system:
        return
    
    try:
        stats = sl.session_state.rag_system.get_database_stats()
        processed_files = stats["processed_files"]
        
        if processed_files:
            sl.subheader("📚 Processed Documents")
            
            # Create DataFrame for display
            df_data = []
            for file_info in processed_files:
                df_data.append({
                    "Filename": file_info["filename"],
                    "Chunks": file_info["chunk_count"],
                    "Size": format_file_size(file_info["size"]),
                    "Processed": datetime.fromisoformat(file_info["processed_at"]).strftime("%Y-%m-%d %H:%M"),
                    "Status": "✅ Available" if file_info["exists"] else "❌ Missing"
                })
            
            df = pd.DataFrame(df_data)
            
            # Display metrics
            col1, col2, col3 = sl.columns(3)
            with col1:
                sl.metric("📄 Total Files", len(processed_files))
            with col2:
                total_chunks = sum(f["chunk_count"] for f in processed_files)
                sl.metric("🧩 Total Chunks", total_chunks)
            with col3:
                missing_files = sum(1 for f in processed_files if not f["exists"])
                sl.metric("❌ Missing Files", missing_files)
            
            # Display table
            sl.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Filename": sl.column_config.TextColumn("📄 Filename", width="medium"),
                    "Chunks": sl.column_config.NumberColumn("🧩 Chunks", width="small"),
                    "Size": sl.column_config.TextColumn("💾 Size", width="small"),
                    "Processed": sl.column_config.TextColumn("🕒 Processed", width="medium"),
                    "Status": sl.column_config.TextColumn("📊 Status", width="small")
                }
            )
        else:
            sl.info("ℹ️ No documents have been processed yet. Select a folder and process some PDFs!")
            
    except Exception as e:
        sl.error(f"Error displaying processed documents: {e}")

def show_chat_interface():
    """Display the chat interface."""
    sl.header("💬 Chat with Your Documents")
    
    if not sl.session_state.rag_system or not sl.session_state.rag_system.is_initialized:
        sl.warning("⚠️ Please process some documents first before starting a chat.")
        if sl.button("📁 Go to Document Management"):
            sl.switch_page("Document Management")
        return
    
    # Chat input
    question = sl.chat_input("Ask a question about your documents...")
    
    if question:
        # Add user message to chat history
        sl.session_state.chat_history.append({"role": "user", "content": question})
        
        with sl.spinner("🤔 Thinking..."):
            try:
                response = sl.session_state.rag_system.ask_question(question)
                
                # Add assistant response to chat history
                sl.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
                add_to_log(f"Answered question: {question[:50]}...", "success")
                
            except Exception as e:
                error_msg = f"Error processing question: {e}"
                sl.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "sources": []
                })
                add_to_log(error_msg, "error")
    
    # Display chat history
    for i, message in enumerate(sl.session_state.chat_history):
        if message["role"] == "user":
            with sl.chat_message("user"):
                sl.write(message["content"])
        else:
            with sl.chat_message("assistant"):
                sl.write(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with sl.expander(f"📚 Sources ({len(message['sources'])})"):
                        for j, source in enumerate(message["sources"], 1):
                            sl.markdown(f"**{j}. {source['filename']}** (Chunk {source.get('chunk_id', 'N/A')})")
                            sl.text(source["content_preview"])
                            sl.divider()
    
    # Clear chat history button
    if sl.session_state.chat_history:
        if sl.button("🗑️ Clear Chat History", type="secondary"):
            sl.session_state.chat_history = []
            sl.rerun()

def show_analytics():
    """Display analytics and statistics."""
    sl.header("📊 Analytics & Statistics")
    
    if not sl.session_state.rag_system:
        sl.warning("⚠️ Please initialize the system firsl.")
        return
    
    try:
        stats = sl.session_state.rag_system.get_database_stats()
        processed_files = stats["processed_files"]
        
        if not processed_files:
            sl.info("ℹ️ No data available. Process some documents first!")
            return
        
        # Overview metrics
        col1, col2, col3, col4 = sl.columns(4)
        
        with col1:
            sl.metric("📄 Total Files", stats["total_files"])
        with col2:
            sl.metric("🧩 Total Chunks", stats["total_chunks"])
        with col3:
            sl.metric("💾 Database Size", format_file_size(stats["database_size"]))
        with col4:
            avg_chunks = stats["total_chunks"] / stats["total_files"] if stats["total_files"] > 0 else 0
            sl.metric("📊 Avg Chunks/File", f"{avg_chunks:.1f}")
        
        # Charts
        col1, col2 = sl.columns(2)
        
        with col1:
            # File size distribution
            sl.subheader("📊 File Size Distribution")
            sizes = [f["size"] for f in processed_files]
            filenames = [f["filename"] for f in processed_files]
            
            # Create DataFrame for Streamlit chart
            size_df = pd.DataFrame({
                "File": filenames,
                "Size (bytes)": sizes
            })
            sl.bar_chart(size_df.set_index("File"))
        
        with col2:
            # Chunk distribution
            sl.subheader("🧩 Chunk Distribution")
            chunks = [f["chunk_count"] for f in processed_files]
            
            # Create DataFrame for Streamlit chart
            chunk_df = pd.DataFrame({
                "File": filenames,
                "Chunks": chunks
            })
            sl.bar_chart(chunk_df.set_index("File"))
        
        # Processing timeline
        sl.subheader("📅 Processing Timeline")
        processing_dates = [
            datetime.fromisoformat(f["processed_at"]).date() 
            for f in processed_files
        ]
        
        # Count files processed per date
        date_counts = {}
        for date in processing_dates:
            date_counts[date] = date_counts.get(date, 0) + 1
        
        if date_counts:
            dates = list(date_counts.keys())
            counts = list(date_counts.values())
            
            # Create DataFrame for Streamlit chart
            timeline_df = pd.DataFrame({
                "Date": dates,
                "Files Processed": counts
            })
            sl.line_chart(timeline_df.set_index("Date"))
        
        # Detailed table
        sl.subheader("📋 Detailed File Information")
        df_data = []
        for file_info in processed_files:
            df_data.append({
                "Filename": file_info["filename"],
                "Chunks": file_info["chunk_count"],
                "Size": format_file_size(file_info["size"]),
                "Processed": datetime.fromisoformat(file_info["processed_at"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Path": file_info["filepath"]
            })
        
        df = pd.DataFrame(df_data)
        sl.dataframe(df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        sl.error(f"Error displaying analytics: {e}")

def show_settings():
    """Display settings and configuration."""
    sl.header("⚙️ Settings & Configuration")
    
    # Model settings
    sl.subheader("🤖 Model Configuration")
    
    col1, col2 = sl.columns(2)
    
    with col1:
        embedding_model = sl.text_input(
            "Embedding Model",
            value="nomic-ai/nomic-embed-text-v1",
            help="HuggingFace embedding model name"
        )
        
        chunk_size = sl.number_input(
            "Chunk Size",
            value=1000,
            min_value=100,
            max_value=2000,
            help="Size of text chunks for processing"
        )
    
    with col2:
        llm_model = sl.text_input(
            "LLM Model",
            value="gemma3:4b",
            help="Ollama model name"
        )
        
        chunk_overlap = sl.number_input(
            "Chunk Overlap",
            value=200,
            min_value=0,
            max_value=500,
            help="Overlap between text chunks"
        )
    
    # Vector database settings
    sl.subheader("💾 Vector Database")
    
    vector_db_path = sl.text_input(
        "Vector Database Path",
        value="./vector_db",
        help="Path where vector database is stored"
    )
    
    # System actions
    sl.subheader("🔧 System Actions")
    
    col1, col2, col3 = sl.columns(3)
    
    with col1:
        if sl.button("🔄 Restart System", use_container_width=True):
            sl.session_state.rag_system = None
            sl.session_state.chat_history = []
            sl.session_state.processing_log = []
            sl.success("✅ System restarted!")
            sl.rerun()
    
    with col2:
        if sl.button("🧹 Clear Logs", use_container_width=True):
            sl.session_state.processing_log = []
            sl.success("✅ Logs cleared!")
    
    with col3:
        if sl.button("💾 Export Logs", use_container_width=True):
            if sl.session_state.processing_log:
                log_text = "\n".join([
                    f"[{entry['timestamp']}] {entry['type'].upper()}: {entry['message']}"
                    for entry in sl.session_state.processing_log
                ])
                sl.download_button(
                    "📥 Download Logs",
                    data=log_text,
                    file_name=f"rag_system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                sl.info("No logs to export")
    
    # System information
    sl.subheader("ℹ️ System Information")
    
    info_data = {
        "Python Path": os.sys.executable,
        "Working Directory": os.getcwd(),
        "Vector DB Path": vector_db_path,
        "Session Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for key, value in info_data.items():
        sl.text(f"{key}: {value}")
    
    # Health check
    sl.subheader("🏥 System Health Check")
    
    if sl.button("🔍 Run Health Check", use_container_width=True):
        health = get_system_health()
        
        for component, status in health.items():
            if status:
                sl.success(f"✅ {component.replace('_', ' ').title()}: OK")
            else:
                sl.error(f"❌ {component.replace('_', ' ').title()}: Failed")

if __name__ == "__main__":
    main() 