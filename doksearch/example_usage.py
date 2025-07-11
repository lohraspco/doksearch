#!/usr/bin/env python3
"""
Example usage of the RAG System with Hybrid Ensemble Retriever
Demonstrates various ways to use the RAG system programmatically.
"""

import os
from rag_system import HybridRAGSystem

def example_basic_usage():
    """Basic example of using the RAG system."""
    print("=" * 60)
    print("BASIC RAG SYSTEM USAGE EXAMPLE")
    print("=" * 60)
    
    # Initialize the RAG system
    rag = HybridRAGSystem(
        embedding_model="nomic-ai/nomic-embed-text-v1",
        llm_model="gemma3:4b"  # Make sure this model is available in Ollama
    )
    
    # Specify your PDF folder path
    pdf_folder = "./pdfs"  # Change this to your PDF folder path
    
    try:
        # Load and process PDFs
        print(f"Loading PDFs from: {pdf_folder}")
        rag.load_pdf_folder(pdf_folder)
        
        # Ask some example questions
        questions = [
            "What is the main topic discussed in the documents?",
            "Can you summarize the key findings?",
            "What are the main conclusions?",
            "What methodology was used?",
            "Are there any recommendations mentioned?"
        ]
        
        for question in questions:
            print(f"\n{'='*40}")
            print(f"Question: {question}")
            print(f"{'='*40}")
            
            response = rag.ask_question(question)
            
            print(f"Answer: {response['answer']}")
            
            if response['sources']:
                print(f"\nSources ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['filename']}")
                    print(f"     Preview: {source['content_preview'][:100]}...")
            else:
                print("No sources found.")
                
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure the model is available: ollama pull gemma3:4b")
        print("3. Check that the PDF folder path exists and contains PDF files")


def example_interactive_session():
    """Example of an interactive session."""
    print("\n" + "=" * 60)
    print("INTERACTIVE SESSION EXAMPLE")
    print("=" * 60)
    
    # Initialize the RAG system
    rag = HybridRAGSystem()
    
    # Specify your PDF folder path
    pdf_folder = input("Enter the path to your PDF folder: ").strip()
    
    if not os.path.exists(pdf_folder):
        print(f"Error: Folder '{pdf_folder}' does not exist.")
        return
    
    try:
        # Load PDFs
        rag.load_pdf_folder(pdf_folder)
        
        print("\n🤖 RAG System is ready! Ask me anything about your documents.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...")
            response = rag.ask_question(question)
            
            print(f"\n💡 Answer: {response['answer']}")
            
            if response['sources']:
                print(f"\n📚 Relevant sources:")
                for i, source in enumerate(response['sources'], 1):
                    print(f"  {i}. {source['filename']}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_custom_configuration():
    """Example with custom configuration."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Initialize with custom settings
    rag = HybridRAGSystem(
        embedding_model="nomic-ai/nomic-embed-text-v1",
        llm_model="gemma3:4b",  # or try "llama3.1:8b", "mistral:7b", etc.
        ollama_base_url="http://localhost:11434"  # Default Ollama URL
    )
    
    pdf_folder = "./pdfs"
    
    try:
        # Load PDFs
        rag.load_pdf_folder(pdf_folder)
        
        # Example of saving and loading vector store for faster subsequent runs
        vector_store_path = "./vector_store"
        rag.save_vector_store(vector_store_path)
        print(f"Vector store saved to: {vector_store_path}")
        
        # You can later load the vector store instead of reprocessing PDFs
        # rag.load_vector_store(vector_store_path)
        
        # Ask a question
        response = rag.ask_question("What are the key topics covered?")
        print(f"Answer: {response['answer']}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_questions():
    """Example of processing multiple questions in batch."""
    print("\n" + "=" * 60)
    print("BATCH QUESTIONS EXAMPLE")
    print("=" * 60)
    
    rag = HybridRAGSystem()
    pdf_folder = "./pdfs"
    
    # List of questions to ask
    batch_questions = [
        "What is the main objective of this research?",
        "What are the key findings or results?",
        "What methodology was employed?",
        "What are the limitations mentioned?",
        "What future work is suggested?",
        "Who are the authors and what are their affiliations?",
        "What datasets were used?",
        "What are the main contributions?",
        "Are there any ethical considerations mentioned?",
        "What related work is cited?"
    ]
    
    try:
        rag.load_pdf_folder(pdf_folder)
        
        results = []
        for i, question in enumerate(batch_questions, 1):
            print(f"\nProcessing question {i}/{len(batch_questions)}: {question}")
            response = rag.ask_question(question)
            results.append(response)
            print(f"Answer: {response['answer'][:100]}..." if len(response['answer']) > 100 else response['answer'])
        
        # Save results to file
        import json
        with open("batch_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBatch results saved to: batch_results.json")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("RAG System Examples")
    print("Make sure you have:")
    print("1. Ollama running: ollama serve")
    print("2. Required model: ollama pull gemma3:4b")
    print("3. PDF files in a folder")
    print("4. Installed dependencies: pip install -r requirements.txt")
    
    # Choose which example to run
    print("\nAvailable examples:")
    print("1. Basic usage")
    print("2. Interactive session")
    print("3. Custom configuration")
    print("4. Batch questions")
    print("5. All examples")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_interactive_session()
    elif choice == "3":
        example_custom_configuration()
    elif choice == "4":
        example_batch_questions()
    elif choice == "5":
        example_basic_usage()
        example_interactive_session()
        example_custom_configuration()
        example_batch_questions()
    else:
        print("Invalid choice. Running basic usage example.")
        example_basic_usage()


if __name__ == "__main__":
    main() 