#!/usr/bin/env python3
"""
Query Enhancement Module for RAG System
Handles query expansion, singular/plural variations, and related terms
"""

import re
from typing import List, Dict, Set
from logging_config import get_logger

logger = get_logger('query_enhancer')

class QueryEnhancer:
    """Enhances search queries with variations and related terms."""
    
    def __init__(self):
        # Common singular to plural mappings for financial/legal terms
        self.singular_plural_map = {
            'proceed': 'proceeds',
            'bond': 'bonds', 
            'security': 'securities',
            'obligation': 'obligations',
            'covenant': 'covenants',
            'provision': 'provisions',
            'term': 'terms',
            'condition': 'conditions',
            'requirement': 'requirements',
            'restriction': 'restrictions',
            'right': 'rights',
            'payment': 'payments',
            'interest': 'interests',
            'principal': 'principals',
            'maturity': 'maturities',
            'redemption': 'redemptions',
            'issuance': 'issuances',
            'holder': 'holders',
            'trustee': 'trustees',
            'issuer': 'issuers',
        }
        
        # Related terms and synonyms
        self.related_terms = {
            'use of proceeds': ['use of proceed', 'utilization of proceeds', 'proceeds usage', 'proceeds application'],
            'proceeds': ['proceed', 'funds', 'money', 'capital', 'financing'],
            'bond': ['bonds', 'security', 'securities', 'obligation', 'obligations'],
            'interest': ['rate', 'yield', 'coupon', 'return'],
            'maturity': ['term', 'duration', 'expiration', 'due date'],
            'redemption': ['call', 'repayment', 'retirement', 'payoff'],
            'default': ['breach', 'violation', 'non-payment', 'failure'],
            'covenant': ['agreement', 'provision', 'condition', 'requirement'],
        }
    
    def enhance_query(self, query: str) -> List[str]:
        """
        Enhance a query by generating variations and related terms.
        
        Args:
            query: Original search query
            
        Returns:
            List of enhanced query variations
        """
        logger.info(f"Enhancing query: '{query}'")
        
        enhanced_queries = [query]  # Always include original
        
        # Add singular/plural variations
        enhanced_queries.extend(self._add_singular_plural_variations(query))
        
        # Add related terms
        enhanced_queries.extend(self._add_related_terms(query))
        
        # Add partial phrase variations
        enhanced_queries.extend(self._add_partial_phrases(query))
        
        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in enhanced_queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())
        
        logger.info(f"Generated {len(unique_queries)} query variations")
        for i, q in enumerate(unique_queries, 1):
            logger.debug(f"  {i}. '{q}'")
        
        return unique_queries
    
    def _add_singular_plural_variations(self, query: str) -> List[str]:
        """Add singular/plural variations of the query."""
        variations = []
        words = query.lower().split()
        
        for word in words:
            # Check if word is in our singular->plural mapping
            if word in self.singular_plural_map:
                plural_word = self.singular_plural_map[word]
                new_query = query.lower().replace(word, plural_word)
                variations.append(new_query)
            
            # Check if word is a plural that we can make singular
            for singular, plural in self.singular_plural_map.items():
                if word == plural:
                    new_query = query.lower().replace(word, singular)
                    variations.append(new_query)
        
        return variations
    
    def _add_related_terms(self, query: str) -> List[str]:
        """Add queries with related terms."""
        variations = []
        query_lower = query.lower()
        
        for key_phrase, related in self.related_terms.items():
            if key_phrase in query_lower:
                for related_term in related:
                    new_query = query_lower.replace(key_phrase, related_term)
                    variations.append(new_query)
            
            # Check if any related term is in the query
            for related_term in related:
                if related_term in query_lower:
                    new_query = query_lower.replace(related_term, key_phrase)
                    variations.append(new_query)
        
        return variations
    
    def _add_partial_phrases(self, query: str) -> List[str]:
        """Add partial phrase variations for longer queries."""
        variations = []
        words = query.split()
        
        if len(words) > 2:
            # Add combinations of 2 words
            for i in range(len(words) - 1):
                partial = f"{words[i]} {words[i+1]}"
                variations.append(partial)
            
            # Add individual significant words (skip common words)
            skip_words = {'of', 'the', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'to'}
            for word in words:
                if word.lower() not in skip_words and len(word) > 3:
                    variations.append(word)
        
        return variations
    
    def get_best_matches(self, query: str, search_function, max_variants: int = 3) -> List[Dict]:
        """
        Get the best search results by trying multiple query variations.
        
        Args:
            query: Original query
            search_function: Function to call for searching (should accept query and return results)
            max_variants: Maximum number of query variants to try
            
        Returns:
            Combined and deduplicated search results
        """
        logger.info(f"Getting best matches for: '{query}'")
        
        # Get enhanced queries
        enhanced_queries = self.enhance_query(query)[:max_variants]
        
        all_results = []
        results_seen = set()
        
        for enhanced_query in enhanced_queries:
            logger.debug(f"Searching with variant: '{enhanced_query}'")
            
            try:
                results = search_function(enhanced_query)
                
                for result in results:
                    # Create a unique identifier for this result
                    result_id = f"{result.get('metadata', {}).get('file_name', '')}_{result.get('metadata', {}).get('page', '')}_{hash(result.get('text', '')[:100])}"
                    
                    if result_id not in results_seen:
                        # Add the query variant that found this result
                        result['query_variant'] = enhanced_query
                        all_results.append(result)
                        results_seen.add(result_id)
                
                logger.debug(f"Found {len(results)} results for '{enhanced_query}'")
                
            except Exception as e:
                logger.error(f"Error searching with variant '{enhanced_query}': {e}")
        
        # Sort by similarity score (descending)
        all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        logger.info(f"Combined results: {len(all_results)} unique matches")
        
        return all_results


# Global instance
query_enhancer = QueryEnhancer() 