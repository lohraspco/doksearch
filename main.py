#!/usr/bin/env python3
"""
RAG System for PDF and DOC Files with Web Scraping
==================================================

This system provides:
1. Document processing for PDF and DOC files
2. Web scraping for documents from websites
3. Vector-based search and retrieval
4. Question answering with references
5. Support for EMMA MSRB and other websites

Usage:
    python main.py --help
    python main.py process-local --dir ./docsJuly
    python main.py scrape-web --url https://emma.msrb.org
    python main.py ask --question "What is the main topic?"
    python main.py interactive
"""

import argparse
import sys
import os
from rag_system import RAGSystem
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_local_documents(rag_system, directory, mode="skip_existing"):
    """Process documents from a local directory."""
    print(f"Processing documents from: {directory}")
    print(f"Processing mode: {mode}")
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return False
    
    result = rag_system.process_local_documents(directory, mode=mode)
    
    if result['success']:
        print(f"✅ Successfully processed documents!")
        print(f"📊 Chunks processed: {result['total_chunks_processed']}")
        print(f"📊 Vector store total: {result['vector_store_total']}")
        
        # Show existing document info if available
        if result.get('existing_info'):
            existing_info = result['existing_info']
            print(f"📊 Document Analysis:")
            print(f"  Existing files: {len(existing_info['existing_files'])}")
            print(f"  New files: {len(existing_info['new_files'])}")
            print(f"  Existing chunks: {existing_info['existing_chunks']}")
            print(f"  New chunks: {existing_info['new_chunks']}")
    else:
        print(f"❌ Failed to process documents: {result.get('error', 'Unknown error')}")
    
    return result['success']

def scrape_web_documents(rag_system, url, max_docs):
    """Scrape and process documents from a website."""
    print(f"Scraping documents from: {url}")
    print(f"Maximum documents to download: {max_docs}")
    
    success = rag_system.scrape_and_process_web_documents(url, max_docs)
    
    if success:
        stats = rag_system.get_system_stats()
        print(f"✅ Successfully scraped and processed documents!")
        print(f"📊 Vector store now contains {stats['vector_store'].get('total_documents', 0)} documents")
    else:
        print("❌ Failed to scrape documents")
    
    return success

def ask_question(rag_system, question, top_k):
    """Ask a question and display the answer with references."""
    print(f"🤔 Question: {question}")
    print("🔍 Searching for relevant information...")
    
    result = rag_system.ask_question(question, top_k)
    
    print(f"\n💡 Answer: {result['answer']}")
    print(f"🎯 Confidence: {result['confidence']:.2%}")
    print(f"📚 Found {result['search_results_count']} relevant documents")
    
    if result['references']:
        print("\n📖 References:")
        for i, ref in enumerate(result['references'], 1):
            print(f"  {i}. {ref['file_name']} (Page {ref['page']}) - Similarity: {ref['similarity_score']:.2%}")
            print(f"     Folder: {ref['folder']}")
            print(f"     Text: {ref['text']}")
            print()
    
    return result

def interactive_mode(rag_system):
    """Run the system in interactive mode."""
    print("🚀 RAG System Interactive Mode")
    print("=" * 50)
    print("Commands:")
    print("  ask <question>  - Ask a question")
    print("  stats          - Show system statistics")
    print("  reset          - Reset the system (clear all documents)")
    print("  quit           - Exit the program")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = rag_system.get_system_stats()
                print(f"📊 System Statistics:")
                print(f"  Vector Store: {stats['vector_store']}")
                print(f"  OpenAI Available: {stats['openai_available']}")
                print(f"  Supported Extensions: {', '.join(stats['supported_extensions'])}")
            
            elif user_input.lower() == 'reset':
                confirm = input("Are you sure you want to reset the system? (y/N): ").strip().lower()
                if confirm == 'y':
                    success = rag_system.reset_system()
                    if success:
                        print("✅ System reset successfully")
                    else:
                        print("❌ Failed to reset system")
            
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                if question:
                    ask_question(rag_system, question, 5)
                else:
                    print("Please provide a question after 'ask'")
            
            else:
                print("❓ Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="RAG System for PDF and DOC Files with Web Scraping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process local documents
    process_parser = subparsers.add_parser('process-local', help='Process documents from a local directory')
    process_parser.add_argument('--dir', required=True, help='Directory containing documents')
    process_parser.add_argument('--mode', choices=['skip_existing', 'upsert', 'add'], default='skip_existing', help='Processing mode: skip_existing (default), upsert (overwrite), add (fail if exists)')
    
    # Scrape web documents
    scrape_parser = subparsers.add_parser('scrape-web', help='Scrape documents from a website')
    scrape_parser.add_argument('--url', required=True, help='URL to scrape')
    scrape_parser.add_argument('--max-docs', type=int, default=10, help='Maximum documents to download')
    
    # Ask question
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('--question', required=True, help='Question to ask')
    ask_parser.add_argument('--top-k', type=int, default=5, help='Number of top results to consider')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Run in interactive mode')
    
    # Stats
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Reset
    subparsers.add_parser('reset', help='Reset the system (clear all documents)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize RAG system
    print("🔧 Initializing RAG System...")
    rag_system = RAGSystem()
    
    try:
        if args.command == 'process-local':
            process_local_documents(rag_system, args.dir, args.mode)
        
        elif args.command == 'scrape-web':
            scrape_web_documents(rag_system, args.url, args.max_docs)
        
        elif args.command == 'ask':
            ask_question(rag_system, args.question, args.top_k)
        
        elif args.command == 'interactive':
            interactive_mode(rag_system)
        
        elif args.command == 'stats':
            stats = rag_system.get_system_stats()
            print(f"📊 System Statistics:")
            print(f"  Vector Store: {stats['vector_store']}")
            print(f"  OpenAI Available: {stats['openai_available']}")
            print(f"  Supported Extensions: {', '.join(stats['supported_extensions'])}")
        
        elif args.command == 'reset':
            confirm = input("Are you sure you want to reset the system? (y/N): ").strip().lower()
            if confirm == 'y':
                success = rag_system.reset_system()
                if success:
                    print("✅ System reset successfully")
                else:
                    print("❌ Failed to reset system")
    
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main() 