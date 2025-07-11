#!/usr/bin/env python3
"""
Diagnostic script to test exact phrase matching in the RAG system.
This will help us understand why "use of proceed" isn't found above the similarity threshold.
"""

import os
import sys
from typing import List, Dict
import logging

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from vector_store import VectorStore
from config import Config
from logging_config import get_logger

logger = get_logger('exact_search_test')

def test_exact_phrase_search():
    """Test searching for exact phrases in the documents."""
    
    print("🔍 Testing Exact Phrase Search")
    print("=" * 50)
    
    # Initialize systems
    rag_system = RAGSystem()
    vector_store = VectorStore()
    
    # Test phrase
    test_phrase = "use of proceed"
    
    print(f"Search phrase: '{test_phrase}'")
    print(f"Current similarity threshold: {Config.SIMILARITY_THRESHOLD}")
    print()
    
    # Get raw search results with different top_k values
    print("📊 Raw Search Results (all distances):")
    print("-" * 40)
    
    raw_results = vector_store.collection.query(
        query_embeddings=vector_store._encode_texts([test_phrase]),
        n_results=20,  # Get top 20 results
        include=['documents', 'metadatas', 'distances']
    )
    
    if raw_results['documents'] and raw_results['documents'][0]:
        for i, (doc, metadata, distance) in enumerate(zip(
            raw_results['documents'][0],
            raw_results['metadatas'][0], 
            raw_results['distances'][0]
        )):
            similarity = 1 - distance
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'Unknown')
            
            print(f"{i+1:2d}. {file_name} (p.{page}) - Similarity: {similarity:.4f}")
            
            # Check if the exact phrase appears in the text
            if test_phrase.lower() in doc.lower():
                print(f"    ✅ EXACT MATCH FOUND in chunk!")
                print(f"    📄 Snippet: ...{doc[max(0, doc.lower().find(test_phrase.lower())-50):doc.lower().find(test_phrase.lower())+len(test_phrase)+50]}...")
            else:
                print(f"    ❌ No exact match in this chunk")
            print()
    
    print("\n" + "=" * 50)
    
    # Test with lower thresholds
    print("🎯 Testing Different Similarity Thresholds:")
    print("-" * 40)
    
    thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    for threshold in thresholds:
        # Temporarily change threshold
        original_threshold = vector_store.similarity_threshold
        vector_store.similarity_threshold = threshold
        
        results = vector_store.search(test_phrase, top_k=10)
        
        print(f"Threshold {threshold:.1f}: {len(results)} results")
        
        if results:
            for i, result in enumerate(results[:3], 1):
                metadata = result['metadata']
                file_name = metadata.get('file_name', 'Unknown')
                page = metadata.get('page', 'Unknown')
                similarity = result['similarity_score']
                
                # Check for exact match
                has_exact = test_phrase.lower() in result['text'].lower()
                match_indicator = "✅" if has_exact else "❌"
                
                print(f"  {i}. {file_name} (p.{page}) - {similarity:.3f} {match_indicator}")
        
        # Restore original threshold
        vector_store.similarity_threshold = original_threshold
        print()
    
    print("=" * 50)
    
    # Test text chunk inspection
    print("🔍 Inspecting Text Chunks for Exact Matches:")
    print("-" * 40)
    
    # Get all documents from the collection
    all_docs = vector_store.collection.get(
        include=['documents', 'metadatas']
    )
    
    exact_matches = []
    if all_docs['documents']:
        for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
            if test_phrase.lower() in doc.lower():
                exact_matches.append({
                    'index': i,
                    'text': doc,
                    'metadata': metadata,
                    'phrase_position': doc.lower().find(test_phrase.lower())
                })
    
    print(f"Found {len(exact_matches)} chunks containing exact phrase '{test_phrase}'")
    
    if exact_matches:
        print("\nExact matches found in these chunks:")
        for i, match in enumerate(exact_matches[:5], 1):  # Show first 5
            metadata = match['metadata']
            file_name = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'Unknown')
            pos = match['phrase_position']
            
            print(f"\n{i}. {file_name} (page {page})")
            print(f"   Position in chunk: {pos}")
            
            # Show context around the phrase
            start = max(0, pos - 100)
            end = min(len(match['text']), pos + len(test_phrase) + 100)
            context = match['text'][start:end]
            
            print(f"   Context: ...{context}...")
            
            # Now test the similarity for this specific chunk
            chunk_embedding = vector_store._encode_texts([match['text']])
            query_embedding = vector_store._encode_texts([test_phrase])
            
            # Calculate cosine similarity manually
            from numpy import dot
            from numpy.linalg import norm
            
            chunk_vec = chunk_embedding[0]
            query_vec = query_embedding[0]
            
            cosine_sim = dot(chunk_vec, query_vec) / (norm(chunk_vec) * norm(query_vec))
            print(f"   Manual cosine similarity: {cosine_sim:.4f}")
    else:
        print(f"\n❌ No chunks found containing exact phrase '{test_phrase}'")
        print("This suggests the phrase might be split across chunks or doesn't exist.")
    
    print("\n" + "=" * 50)
    
    # Suggest solutions
    print("💡 Suggestions:")
    print("-" * 40)
    print("1. If exact matches exist but similarity is low:")
    print("   - Lower the similarity threshold in config.py")
    print("   - Consider using a different embedding model")
    print("   - Try keyword-based search as fallback")
    print()
    print("2. If no exact matches found:")
    print("   - Check if phrase spans multiple chunks")
    print("   - Verify the document was processed correctly")
    print("   - Try searching for partial phrases")
    print()
    print("3. Current similarity threshold is quite low (0.3)")
    print("   - Consider lowering to 0.1-0.2 for broader results")
    print("   - Or implement hybrid search (semantic + keyword)")

def test_related_phrases():
    """Test searching for related phrases."""
    
    print("\n" + "=" * 50)
    print("🔍 Testing Related Phrases:")
    print("-" * 40)
    
    vector_store = VectorStore()
    
    related_phrases = [
        "proceed",
        "use of proceeds", 
        "proceeds",
        "use proceeds",
        "utilization of proceeds",
        "proceed with"
    ]
    
    for phrase in related_phrases:
        results = vector_store.search(phrase, top_k=3)
        print(f"'{phrase}': {len(results)} results")
        
        if results:
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                file_name = metadata.get('file_name', 'Unknown')
                page = metadata.get('page', 'Unknown')
                similarity = result['similarity_score']
                
                print(f"  {i}. {file_name} (p.{page}) - {similarity:.3f}")
        print()

if __name__ == "__main__":
    try:
        test_exact_phrase_search()
        test_related_phrases()
        
    except Exception as e:
        logger.error(f"Error in exact search test: {e}")
        print(f"❌ Error: {e}") 