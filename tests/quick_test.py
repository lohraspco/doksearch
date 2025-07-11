#!/usr/bin/env python3
"""Quick test of enhanced search functionality."""

from rag_system import RAGSystem
from query_enhancer import query_enhancer

print("🔍 Quick Enhanced Search Test")
print("=" * 50)

# Initialize RAG system
rag_system = RAGSystem()

# Test query enhancement for 'use of proceed'
test_query = "use of proceed"
print(f"Testing query enhancement for: '{test_query}'")

enhanced_queries = query_enhancer.enhance_query(test_query)
print(f"Generated {len(enhanced_queries)} variants:")
for i, q in enumerate(enhanced_queries[:5], 1):
    print(f"  {i}. '{q}'")

print("\nTesting enhanced search...")
enhanced_results = query_enhancer.get_best_matches(
    test_query,
    lambda q: rag_system.vector_store.search(q, top_k=3),
    max_variants=3
)

print(f"Enhanced search found {len(enhanced_results)} results")
for i, result in enumerate(enhanced_results, 1):
    metadata = result.get('metadata', {})
    file_name = metadata.get('file_name', 'Unknown')
    page = metadata.get('page', 'Unknown')
    similarity = result.get('similarity_score', 0)
    query_variant = result.get('query_variant', 'original')
    
    print(f"  {i}. {file_name} (p.{page}) - {similarity:.3f} via '{query_variant}'")

print("\n" + "=" * 50)
print("Testing full RAG question...")

# Test full question
question = "What is the use of proceed mentioned in the documents?"
result = rag_system.ask_question(question, use_enhanced_search=True)

print(f"Question: {question}")
print(f"Answer: {result['answer'][:150]}...")
print(f"Confidence: {result['confidence']:.1%}")
print(f"References: {len(result['references'])}")

if result['references']:
    print("\nTop references:")
    for i, ref in enumerate(result['references'][:3], 1):
        print(f"  {i}. {ref['file_name']} (p.{ref['page']}) - {ref['similarity_score']:.3f}")

print("\n✅ Test completed!") 