#!/usr/bin/env python3
"""
Test script for enhanced search functionality.
Tests query enhancement for "use of proceed" and other difficult queries.
"""

import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem
from query_enhancer import query_enhancer
from logging_config import get_logger

logger = get_logger('enhanced_search_test')

def main():
    print("🔍 Testing Enhanced Search Functionality")
    print("=" * 60)
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Test queries that previously had issues
    test_queries = [
        "use of proceed",
        "use of proceeds", 
        "financial terms",
        "bond maturity",
        "redemption provision"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing query: '{query}'")
        print("-" * 50)
        
        # Test query enhancement
        enhancement_test = rag_system.test_query_enhancement(query)
        
        if 'error' in enhancement_test:
            print(f"❌ Error: {enhancement_test['error']}")
            continue
        
        print(f"Query variants: {len(enhancement_test['query_variants'])}")
        for i, variant in enumerate(enhancement_test['query_variants'][:5], 1):
            print(f"  {i}. '{variant}'")
        
        print(f"\nResults comparison:")
        print(f"  Basic search: {enhancement_test['basic_results_count']} results")
        print(f"  Enhanced search: {enhancement_test['enhanced_results_count']} results")
        print(f"  Improvement: +{enhancement_test['improvement']} results")
        
        # Show top results
        if enhancement_test['enhanced_results']:
            print(f"\nTop enhanced results:")
            for i, result in enumerate(enhancement_test['enhanced_results'], 1):
                metadata = result.get('metadata', {})
                file_name = metadata.get('file_name', 'Unknown')
                page = metadata.get('page', 'Unknown')
                similarity = result.get('similarity_score', 0)
                query_variant = result.get('query_variant', query)
                
                print(f"  {i}. {file_name} (p.{page}) - {similarity:.3f}")
                if query_variant != query:
                    print(f"     Found via: '{query_variant}'")
    
    print("\n" + "=" * 60)
    print("🧪 Testing RAG Question with Enhancement")
    print("-" * 60)
    
    # Test full RAG pipeline with enhanced search
    test_question = "What is the use of proceed mentioned in the documents?"
    
    print(f"Question: {test_question}")
    print()
    
    # Test with enhanced search
    print("Enhanced Search Results:")
    result_enhanced = rag_system.ask_question(test_question, use_enhanced_search=True)
    
    print(f"Answer: {result_enhanced['answer'][:200]}...")
    print(f"Confidence: {result_enhanced['confidence']:.1%}")
    print(f"References: {len(result_enhanced['references'])}")
    
    if result_enhanced['references']:
        print("\nTop references:")
        for i, ref in enumerate(result_enhanced['references'][:3], 1):
            print(f"  {i}. {ref['file_name']} (p.{ref['page']}) - {ref['similarity_score']:.3f}")
    
    print("\n" + "-" * 60)
    
    # Test without enhanced search for comparison
    print("Traditional Search Results:")
    result_traditional = rag_system.ask_question(test_question, use_enhanced_search=False)
    
    print(f"Answer: {result_traditional['answer'][:200]}...")
    print(f"Confidence: {result_traditional['confidence']:.1%}")
    print(f"References: {len(result_traditional['references'])}")
    
    # Compare results
    print(f"\n📊 Comparison:")
    print(f"Enhanced references: {len(result_enhanced['references'])}")
    print(f"Traditional references: {len(result_traditional['references'])}")
    print(f"Enhancement improved results: {len(result_enhanced['references']) > len(result_traditional['references'])}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error in enhanced search test: {e}")
        logger.error(f"Enhanced search test error: {e}") 