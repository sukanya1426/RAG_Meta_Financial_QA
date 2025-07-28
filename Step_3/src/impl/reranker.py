"""
Reranker for Step 3: Implements advanced reranking strategies for better result relevance.
Uses cross-encoder models, BM25, and custom relevance scoring.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

logger = logging.getLogger(__name__)

class AdvancedReranker:
    """Advanced reranking using multiple relevance signals."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        try:
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {str(e)}")
            self.cross_encoder = None
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.bm25 = None
        self.corpus_fitted = False
        
        # Financial relevance weights
        self.relevance_weights = {
            'cross_encoder': 0.4,
            'bm25': 0.25,
            'tfidf': 0.15,
            'financial_terms': 0.1,
            'temporal_match': 0.1
        }
        
        self.financial_keywords = {
            'metrics': ['revenue', 'income', 'profit', 'margin', 'eps', 'dap', 'mau', 'arpu'],
            'periods': ['q1', 'q2', 'q3', 'q4', '2024', '2023', 'quarter', 'year'],
            'business': ['family of apps', 'reality labs', 'meta', 'facebook', 'instagram', 'whatsapp'],
            'operations': ['expenses', 'costs', 'investments', 'capex', 'opex', 'r&d']
        }

    def fit_corpus(self, documents: List[str]) -> None:
        """Fit BM25 and TF-IDF models on the document corpus."""
        try:
            if not documents:
                logger.warning("Empty corpus provided for fitting")
                return
            
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            # Fit TF-IDF vectorizer
            self.tfidf_vectorizer.fit(documents)
            
            self.corpus_fitted = True
            logger.info(f"Fitted reranker on corpus of {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error fitting corpus: {str(e)}")
            self.corpus_fitted = False

    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]], 
               top_k: int = 5,
               method: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Rerank candidate documents using specified method.
        
        Args:
            query: Search query
            candidates: List of candidate documents with content and metadata
            top_k: Number of top results to return
            method: Reranking method ('cross_encoder', 'bm25', 'tfidf', 'hybrid')
            
        Returns:
            Reranked list of candidates with scores
        """
        if not candidates:
            return []
        
        logger.info(f"Reranking {len(candidates)} candidates using {method} method")
        
        try:
            if method == "cross_encoder":
                return self._rerank_cross_encoder(query, candidates, top_k)
            elif method == "bm25":
                return self._rerank_bm25(query, candidates, top_k)
            elif method == "tfidf":
                return self._rerank_tfidf(query, candidates, top_k)
            elif method == "financial":
                return self._rerank_financial(query, candidates, top_k)
            else:  # hybrid
                return self._rerank_hybrid(query, candidates, top_k)
                
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Return original candidates with default scores
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = 1.0 - (i * 0.1)
            return candidates[:top_k]

    def _rerank_cross_encoder(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder model."""
        if not self.cross_encoder:
            logger.warning("Cross-encoder not available, using default scoring")
            return candidates[:top_k]
        
        try:
            # Prepare query-document pairs
            pairs = [(query, candidate.get('content', '')) for candidate in candidates]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Add scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(scores[i])
                candidate['rerank_method'] = 'cross_encoder'
            
            # Sort by score and return top_k
            sorted_candidates = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"Cross-encoder reranking completed, top score: {sorted_candidates[0].get('rerank_score', 0):.3f}")
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            return candidates[:top_k]

    def _rerank_bm25(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using BM25 scoring."""
        try:
            query_tokens = query.lower().split()
            docs = [candidate.get('content', '') for candidate in candidates]
            
            # Always refit BM25 for current candidates to avoid index mismatch
            tokenized_docs = [doc.lower().split() for doc in docs]
            current_bm25 = BM25Okapi(tokenized_docs)
            
            # Get BM25 scores
            scores = current_bm25.get_scores(query_tokens)
            
            # Normalize scores
            if len(scores) > 0 and max(scores) > 0:
                scores = scores / max(scores)
            
            # Add scores to candidates
            for i, candidate in enumerate(candidates):
                if i < len(scores):  # Safety check
                    candidate['rerank_score'] = float(scores[i])
                else:
                    candidate['rerank_score'] = 0.0
                candidate['rerank_method'] = 'bm25'
            
            sorted_candidates = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"BM25 reranking completed, top score: {sorted_candidates[0].get('rerank_score', 0):.3f}")
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in BM25 reranking: {str(e)}")
            return candidates[:top_k]

    def _rerank_tfidf(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using TF-IDF cosine similarity."""
        try:
            docs = [candidate.get('content', '') for candidate in candidates]
            
            # Fit TF-IDF if not already fitted
            if not self.corpus_fitted:
                self.tfidf_vectorizer.fit(docs)
            
            # Transform documents and query
            doc_vectors = self.tfidf_vectorizer.transform(docs)
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Add scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(similarities[i])
                candidate['rerank_method'] = 'tfidf'
            
            sorted_candidates = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"TF-IDF reranking completed, top score: {sorted_candidates[0].get('rerank_score', 0):.3f}")
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in TF-IDF reranking: {str(e)}")
            return candidates[:top_k]

    def _rerank_financial(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using financial domain-specific scoring."""
        try:
            query_lower = query.lower()
            
            for candidate in candidates:
                score = 0.0
                content = candidate.get('content', '').lower()
                
                # Financial keyword matching
                for category, keywords in self.financial_keywords.items():
                    query_matches = sum(1 for kw in keywords if kw in query_lower)
                    content_matches = sum(1 for kw in keywords if kw in content)
                    
                    if query_matches > 0 and content_matches > 0:
                        score += (query_matches * content_matches) * 0.25
                
                # Temporal alignment
                query_periods = self._extract_periods(query_lower)
                content_periods = self._extract_periods(content)
                
                if query_periods and content_periods:
                    overlap = len(set(query_periods) & set(content_periods))
                    score += overlap * 0.3
                
                # Structural data bonus
                if candidate.get('content_type') == 'table' and any(kw in query_lower for kw in ['compare', 'vs', 'number', 'amount']):
                    score += 0.2
                
                # Length penalty for very short or very long content
                content_length = len(content.split())
                if 20 <= content_length <= 200:
                    score += 0.1
                
                candidate['rerank_score'] = score
                candidate['rerank_method'] = 'financial'
            
            sorted_candidates = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"Financial reranking completed, top score: {sorted_candidates[0].get('rerank_score', 0):.3f}")
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in financial reranking: {str(e)}")
            return candidates[:top_k]

    def _rerank_hybrid(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank using hybrid approach combining multiple methods."""
        try:
            # Get scores from different methods
            ce_candidates = self._rerank_cross_encoder(query, [dict(c) for c in candidates], len(candidates))
            bm25_candidates = self._rerank_bm25(query, [dict(c) for c in candidates], len(candidates))
            tfidf_candidates = self._rerank_tfidf(query, [dict(c) for c in candidates], len(candidates))
            financial_candidates = self._rerank_financial(query, [dict(c) for c in candidates], len(candidates))
            
            # Create score lookup dictionaries
            ce_scores = {c.get('source', i): c.get('rerank_score', 0) for i, c in enumerate(ce_candidates)}
            bm25_scores = {c.get('source', i): c.get('rerank_score', 0) for i, c in enumerate(bm25_candidates)}
            tfidf_scores = {c.get('source', i): c.get('rerank_score', 0) for i, c in enumerate(tfidf_candidates)}
            financial_scores = {c.get('source', i): c.get('rerank_score', 0) for i, c in enumerate(financial_candidates)}
            
            # Combine scores using weights
            for i, candidate in enumerate(candidates):
                source_key = candidate.get('source', i)
                
                hybrid_score = (
                    self.relevance_weights['cross_encoder'] * ce_scores.get(source_key, 0) +
                    self.relevance_weights['bm25'] * bm25_scores.get(source_key, 0) +
                    self.relevance_weights['tfidf'] * tfidf_scores.get(source_key, 0) +
                    (self.relevance_weights['financial_terms'] + self.relevance_weights['temporal_match']) * financial_scores.get(source_key, 0)
                )
                
                candidate['rerank_score'] = hybrid_score
                candidate['rerank_method'] = 'hybrid'
                candidate['rerank_components'] = {
                    'cross_encoder': ce_scores.get(source_key, 0),
                    'bm25': bm25_scores.get(source_key, 0),
                    'tfidf': tfidf_scores.get(source_key, 0),
                    'financial': financial_scores.get(source_key, 0)
                }
            
            sorted_candidates = sorted(candidates, key=lambda x: x.get('rerank_score', 0), reverse=True)
            logger.info(f"Hybrid reranking completed, top score: {sorted_candidates[0].get('rerank_score', 0):.3f}")
            
            return sorted_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error in hybrid reranking: {str(e)}")
            return candidates[:top_k]

    def _extract_periods(self, text: str) -> List[str]:
        """Extract time periods from text."""
        periods = []
        
        # Extract quarters
        quarters = re.findall(r'q[1-4]\s*20[0-9]{2}', text.lower())
        periods.extend(quarters)
        
        # Extract years
        years = re.findall(r'20[0-9]{2}', text)
        periods.extend(years)
        
        # Extract quarter references
        quarter_refs = re.findall(r'(?:first|second|third|fourth)\s+quarter', text.lower())
        periods.extend(quarter_refs)
        
        return list(set(periods))

    def analyze_reranking_impact(self, 
                                original_results: List[Dict[str, Any]], 
                                reranked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of reranking on result ordering."""
        try:
            original_order = [r.get('source', i) for i, r in enumerate(original_results)]
            reranked_order = [r.get('source', i) for i, r in enumerate(reranked_results)]
            
            # Calculate rank changes
            rank_changes = []
            for i, source in enumerate(reranked_order):
                if source in original_order:
                    original_rank = original_order.index(source)
                    rank_change = original_rank - i
                    rank_changes.append(rank_change)
            
            # Calculate metrics
            analysis = {
                'total_candidates': len(original_results),
                'reranked_candidates': len(reranked_results),
                'rank_changes': rank_changes,
                'mean_rank_change': np.mean(rank_changes) if rank_changes else 0,
                'max_rank_improvement': max(rank_changes) if rank_changes else 0,
                'max_rank_degradation': min(rank_changes) if rank_changes else 0,
                'stability_score': 1 - (np.std(rank_changes) / len(rank_changes)) if rank_changes else 1,
                'top_3_overlap': len(set(original_order[:3]) & set(reranked_order[:3])),
                'reranking_method': reranked_results[0].get('rerank_method', 'unknown') if reranked_results else 'none'
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing reranking impact: {str(e)}")
            return {'error': str(e)}

    def get_reranking_explanation(self, query: str, candidate: Dict[str, Any]) -> str:
        """Generate explanation for why a candidate was ranked at its position."""
        try:
            score = candidate.get('rerank_score', 0)
            method = candidate.get('rerank_method', 'unknown')
            content = candidate.get('content', '')[:100]
            
            explanation = f"Relevance Score: {score:.3f} (Method: {method})\n"
            explanation += f"Content Preview: {content}...\n"
            
            if 'rerank_components' in candidate:
                components = candidate['rerank_components']
                explanation += "Score Breakdown:\n"
                for component, score in components.items():
                    explanation += f"  - {component}: {score:.3f}\n"
            
            # Add specific insights
            query_lower = query.lower()
            content_lower = content.lower()
            
            insights = []
            for category, keywords in self.financial_keywords.items():
                matches = [kw for kw in keywords if kw in query_lower and kw in content_lower]
                if matches:
                    insights.append(f"Matched {category} terms: {', '.join(matches)}")
            
            if insights:
                explanation += "Key Matches:\n"
                for insight in insights:
                    explanation += f"  - {insight}\n"
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
