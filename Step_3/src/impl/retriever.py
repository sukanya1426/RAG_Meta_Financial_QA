from typing import Dict, Any, List, Optional
from ..interface.base_datastore import BaseDatastore
from ..interface.base_retriever import BaseRetriever
from .query_optimizer import QueryOptimizer
from .reranker import AdvancedReranker
import logging
import time

logger = logging.getLogger(__name__)

class AdvancedRetriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore, enable_optimization: bool = True, enable_reranking: bool = True):
        self.datastore = datastore
        self.enable_optimization = enable_optimization
        self.enable_reranking = enable_reranking
        
        # Initialize advanced components
        if enable_optimization:
            try:
                self.query_optimizer = QueryOptimizer()
                logger.info("Query optimizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize query optimizer: {str(e)}")
                self.query_optimizer = None
                self.enable_optimization = False
        
        if enable_reranking:
            try:
                self.reranker = AdvancedReranker()
                logger.info("Advanced reranker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {str(e)}")
                self.reranker = None
                self.enable_reranking = False

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Standard vector search with optional query optimization."""
        if self.enable_optimization and self.query_optimizer:
            try:
                optimization_result = self.query_optimizer.optimize_query(query, "rewrite")
                optimized_queries = optimization_result.get('optimized_queries', [query])
                if optimized_queries:
                    query = optimized_queries[0]  # Use the first optimized query
                    logger.info(f"Using optimized query: {query}")
            except Exception as e:
                logger.error(f"Query optimization failed: {str(e)}")
        
        return self.datastore.search(query, top_k=top_k)
    
    def advanced_search(self, 
                       query: str, 
                       top_k: int = 5, 
                       optimization_type: str = "auto",
                       rerank_method: str = "hybrid",
                       chunk_size_preference: Optional[str] = None) -> Dict[str, Any]:
        """
        Advanced search with query optimization, hybrid retrieval, and reranking.
        
        Args:
            query: Original search query
            top_k: Number of top results to return
            optimization_type: Type of query optimization ('auto', 'rewrite', 'expand', 'decompose')
            rerank_method: Reranking method ('cross_encoder', 'bm25', 'tfidf', 'financial', 'hybrid')
            chunk_size_preference: Preferred chunk size ('small', 'medium', 'large')
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        
        search_result = {
            'original_query': query,
            'optimized_queries': [query],
            'search_results': [],
            'metadata': {
                'optimization_enabled': self.enable_optimization,
                'reranking_enabled': self.enable_reranking,
                'search_time': 0.0,
                'optimization_strategy': optimization_type,
                'rerank_method': rerank_method
            }
        }
        
        try:
            # Step 1: Query Optimization
            final_queries = [query]
            if self.enable_optimization and self.query_optimizer:
                logger.info(f"Optimizing query with strategy: {optimization_type}")
                optimization_result = self.query_optimizer.optimize_query(query, optimization_type)
                final_queries = optimization_result.get('optimized_queries', [query])
                search_result['optimized_queries'] = final_queries
                search_result['metadata']['query_type'] = optimization_result.get('query_type', 'unknown')
                logger.info(f"Generated {len(final_queries)} optimized queries")
            
            # Step 2: Multi-Query Retrieval
            all_candidates = []
            for i, opt_query in enumerate(final_queries[:3]):  # Limit to top 3 optimized queries
                try:
                    # Use hybrid search if available
                    if hasattr(self.datastore, 'hybrid_search'):
                        hybrid_results = self.datastore.hybrid_search(opt_query, top_k * 2)
                        
                        # Convert hybrid results to candidate format
                        candidates = self._process_hybrid_results(hybrid_results, opt_query)
                    else:
                        # Fallback to regular search
                        logger.warning("Hybrid search not available, using regular search")
                        text_results = self.datastore.search(opt_query, top_k * 2)
                        candidates = [{'content': result, 'source': f'result_{j}', 'query_variant': i} 
                                    for j, result in enumerate(text_results)]
                    
                    all_candidates.extend(candidates)
                    logger.info(f"Query variant {i+1} retrieved {len(candidates)} candidates")
                    
                except Exception as e:
                    logger.error(f"Error retrieving for query variant {i}: {str(e)}")
                    continue
            
            # Remove duplicates based on content similarity
            unique_candidates = self._deduplicate_candidates(all_candidates)
            logger.info(f"After deduplication: {len(unique_candidates)} unique candidates")
            
            # Step 3: Chunk Size Filtering (if preference specified)
            if chunk_size_preference:
                unique_candidates = self._filter_by_chunk_size(unique_candidates, chunk_size_preference)
                logger.info(f"After chunk size filtering: {len(unique_candidates)} candidates")
            
            # Step 4: Advanced Reranking
            if self.enable_reranking and self.reranker and unique_candidates:
                logger.info(f"Reranking {len(unique_candidates)} candidates using {rerank_method}")
                
                # Fit reranker on current candidates if needed
                if not self.reranker.corpus_fitted:
                    docs = [c.get('content', '') for c in unique_candidates]
                    self.reranker.fit_corpus(docs)
                
                reranked_candidates = self.reranker.rerank(
                    query, 
                    unique_candidates, 
                    top_k=top_k,
                    method=rerank_method
                )
                search_result['search_results'] = reranked_candidates
                search_result['metadata']['rerank_analysis'] = self.reranker.analyze_reranking_impact(
                    unique_candidates, reranked_candidates
                )
            else:
                # No reranking, just return top candidates
                search_result['search_results'] = unique_candidates[:top_k]
            
            # Step 5: Add search metadata
            search_time = time.time() - start_time
            search_result['metadata']['search_time'] = search_time
            search_result['metadata']['total_candidates_found'] = len(all_candidates)
            search_result['metadata']['unique_candidates'] = len(unique_candidates)
            search_result['metadata']['final_results'] = len(search_result['search_results'])
            
            logger.info(f"Advanced search completed in {search_time:.2f}s, returned {len(search_result['search_results'])} results")
            
        except Exception as e:
            logger.error(f"Error in advanced search: {str(e)}")
            # Fallback to basic search
            basic_results = self.datastore.search(query, top_k)
            search_result['search_results'] = [{'content': result, 'source': f'fallback_{i}'} 
                                             for i, result in enumerate(basic_results)]
            search_result['metadata']['error'] = str(e)
            search_result['metadata']['fallback_used'] = True
        
        return search_result
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Enhanced hybrid search with optimization and reranking."""
        return self.advanced_search(
            query=query,
            top_k=top_k,
            optimization_type="auto",
            rerank_method="hybrid"
        )
    
    def iterative_search(self, 
                        query: str, 
                        max_iterations: int = 3,
                        convergence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Iterative retrieval that refines search based on initial results.
        
        Args:
            query: Original search query
            max_iterations: Maximum number of search iterations
            convergence_threshold: Similarity threshold to stop iterations
            
        Returns:
            Dictionary containing final results and iteration metadata
        """
        logger.info(f"Starting iterative search with max {max_iterations} iterations")
        
        results = {
            'original_query': query,
            'iterations': [],
            'final_results': [],
            'converged': False,
            'total_iterations': 0
        }
        
        current_query = query
        previous_results = []
        
        for iteration in range(max_iterations):
            try:
                # Perform search for current iteration
                search_result = self.advanced_search(
                    current_query, 
                    top_k=5,
                    optimization_type="expand" if iteration == 0 else "rewrite"
                )
                
                current_results = search_result['search_results']
                
                # Store iteration data
                iteration_data = {
                    'iteration': iteration + 1,
                    'query': current_query,
                    'results_count': len(current_results),
                    'search_time': search_result['metadata'].get('search_time', 0)
                }
                results['iterations'].append(iteration_data)
                
                # Check convergence
                if previous_results and self._check_convergence(previous_results, current_results, convergence_threshold):
                    logger.info(f"Search converged at iteration {iteration + 1}")
                    results['converged'] = True
                    results['final_results'] = current_results
                    break
                
                # Prepare query for next iteration based on current results
                if iteration < max_iterations - 1:
                    current_query = self._refine_query_from_results(query, current_results)
                    logger.info(f"Refined query for iteration {iteration + 2}: {current_query}")
                
                previous_results = current_results
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                break
        
        results['total_iterations'] = len(results['iterations'])
        if not results['final_results']:
            results['final_results'] = previous_results
        
        logger.info(f"Iterative search completed after {results['total_iterations']} iterations")
        return results
    
    def _process_hybrid_results(self, hybrid_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Process hybrid search results into candidate format."""
        candidates = []
        
        # Process text context
        text_context = hybrid_results.get('text_context', [])
        for i, content in enumerate(text_context):
            candidates.append({
                'content': content,
                'source': f'text_context_{i}',
                'content_type': 'text',
                'query': query
            })
        
        # Process structured data
        structured_data = hybrid_results.get('structured_data', [])
        for i, data in enumerate(structured_data):
            content = data.get('content', '')
            if not content and 'data' in data:
                # Convert structured data to text representation
                content = self._structured_data_to_text(data['data'])
            
            candidates.append({
                'content': content,
                'source': data.get('source', f'structured_{i}'),
                'content_type': 'table',
                'structured_data': data.get('data'),
                'query': query
            })
        
        # Process financial data
        financial_data = hybrid_results.get('financial_data', [])
        for i, data in enumerate(financial_data):
            candidates.append({
                'content': data.get('content', ''),
                'source': data.get('source', f'financial_{i}'),
                'content_type': data.get('type', 'financial'),
                'query': query
            })
        
        return candidates
    
    def _structured_data_to_text(self, structured_data: Any) -> str:
        """Convert structured data to text representation."""
        if isinstance(structured_data, list) and structured_data:
            # Assume it's a list of dictionaries (table rows)
            if isinstance(structured_data[0], dict):
                headers = list(structured_data[0].keys())
                text_parts = [" | ".join(headers)]
                text_parts.append("-" * 50)
                
                for row in structured_data:
                    row_text = " | ".join(str(row.get(header, '')) for header in headers)
                    text_parts.append(row_text)
                
                return "\n".join(text_parts)
        
        return str(structured_data)
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate candidates based on content similarity."""
        if not candidates:
            return []
        
        unique_candidates = []
        seen_content = set()
        
        for candidate in candidates:
            content = candidate.get('content', '')
            # Simple deduplication based on content hash
            content_hash = hash(content.lower().strip())
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _filter_by_chunk_size(self, candidates: List[Dict[str, Any]], preference: str) -> List[Dict[str, Any]]:
        """Filter candidates by chunk size preference."""
        if not candidates:
            return candidates
        
        # Define chunk size ranges (in words)
        size_ranges = {
            'small': (0, 100),
            'medium': (100, 300),
            'large': (300, float('inf'))
        }
        
        min_size, max_size = size_ranges.get(preference, (0, float('inf')))
        
        filtered = []
        for candidate in candidates:
            content = candidate.get('content', '')
            word_count = len(content.split())
            
            if min_size <= word_count < max_size:
                candidate['word_count'] = word_count
                filtered.append(candidate)
        
        return filtered
    
    def _check_convergence(self, 
                          previous_results: List[Dict[str, Any]], 
                          current_results: List[Dict[str, Any]], 
                          threshold: float) -> bool:
        """Check if search results have converged."""
        if not previous_results or not current_results:
            return False
        
        # Simple convergence check based on content overlap
        prev_content = set(r.get('content', '')[:100] for r in previous_results)
        curr_content = set(r.get('content', '')[:100] for r in current_results)
        
        overlap = len(prev_content & curr_content)
        similarity = overlap / max(len(prev_content), len(curr_content))
        
        return similarity >= threshold
    
    def _refine_query_from_results(self, original_query: str, results: List[Dict[str, Any]]) -> str:
        """Refine query based on search results for next iteration."""
        if not results:
            return original_query
        
        # Extract key terms from top results
        top_contents = [r.get('content', '') for r in results[:2]]
        combined_content = ' '.join(top_contents)
        
        # Simple refinement: add most frequent terms not in original query
        original_terms = set(original_query.lower().split())
        content_terms = combined_content.lower().split()
        
        # Find frequent terms in results that aren't in original query
        term_freq = {}
        for term in content_terms:
            if len(term) > 3 and term not in original_terms:
                term_freq[term] = term_freq.get(term, 0) + 1
        
        # Add top 2 frequent terms
        if term_freq:
            top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:2]
            additional_terms = [term for term, freq in top_terms]
            refined_query = f"{original_query} {' '.join(additional_terms)}"
            return refined_query
        
        return original_query

# Maintain backward compatibility
class Retriever(AdvancedRetriever):
    """Backward compatible retriever class."""
    def __init__(self, datastore: BaseDatastore):
        super().__init__(datastore, enable_optimization=False, enable_reranking=False)