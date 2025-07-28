"""
Advanced Evaluation Framework for Step 3: Comprehensive evaluation metrics for RAG systems.
Implements retrieval metrics (Precision@k, Recall@k, MRR), answer quality metrics (BLEU, ROUGE),
and end-to-end evaluation with manual rubrics.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import re
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation."""
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    map_score: float = 0.0  # Mean Average Precision
    ndcg_at_5: float = 0.0  # Normalized Discounted Cumulative Gain

@dataclass
class AnswerQualityMetrics:
    """Metrics for answer quality evaluation."""
    exact_match: float = 0.0
    partial_match: float = 0.0
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    fluency: float = 0.0
    bleu_score: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0

@dataclass
class EndToEndMetrics:
    """End-to-end evaluation metrics."""
    overall_quality: float = 0.0
    user_satisfaction: float = 0.0
    task_completion: float = 0.0
    response_time: float = 0.0
    confidence_score: float = 0.0

@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    query_id: str
    query: str
    expected_answer: str
    generated_answer: str
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[str]
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerQualityMetrics
    end_to_end_metrics: EndToEndMetrics
    timestamp: str
    metadata: Dict[str, Any]

class AdvancedEvaluator:
    """Advanced evaluation framework for RAG systems."""
    
    def __init__(self):
        self.financial_entities = {
            'metrics': ['revenue', 'net income', 'profit', 'eps', 'margin', 'dap', 'mau'],
            'values': ['billion', 'million', 'percent', '%', '$'],
            'periods': ['q1', 'q2', 'q3', 'q4', '2024', '2023', 'quarter'],
            'companies': ['meta', 'facebook', 'instagram', 'whatsapp']
        }
        
        # Evaluation rubric for manual scoring
        self.evaluation_rubric = {
            'factual_accuracy': {
                'excellent': 1.0,
                'good': 0.8,
                'fair': 0.6,
                'poor': 0.3,
                'incorrect': 0.0
            },
            'completeness': {
                'complete': 1.0,
                'mostly_complete': 0.8,
                'partially_complete': 0.6,
                'incomplete': 0.3,
                'minimal': 0.0
            },
            'relevance': {
                'highly_relevant': 1.0,
                'relevant': 0.8,
                'somewhat_relevant': 0.6,
                'marginally_relevant': 0.3,
                'irrelevant': 0.0
            }
        }

    def evaluate_retrieval(self, 
                          retrieved_docs: List[Dict[str, Any]], 
                          relevant_docs: List[str],
                          k_values: List[int] = [1, 3, 5]) -> RetrievalMetrics:
        """
        Evaluate retrieval performance using standard IR metrics.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            relevant_docs: List of relevant document IDs/sources
            k_values: List of k values for Precision@k and Recall@k
            
        Returns:
            RetrievalMetrics object with calculated metrics
        """
        metrics = RetrievalMetrics()
        
        if not retrieved_docs or not relevant_docs:
            logger.warning("Empty retrieved docs or relevant docs for evaluation")
            return metrics
        
        try:
            # Extract document IDs from retrieved docs
            retrieved_ids = [doc.get('source', str(i)) for i, doc in enumerate(retrieved_docs)]
            relevant_set = set(relevant_docs)
            
            # Calculate Precision@k and Recall@k
            for k in k_values:
                retrieved_at_k = retrieved_ids[:k]
                relevant_retrieved_at_k = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_set]
                
                precision_at_k = len(relevant_retrieved_at_k) / k if k > 0 else 0
                recall_at_k = len(relevant_retrieved_at_k) / len(relevant_set) if len(relevant_set) > 0 else 0
                
                if k == 1:
                    metrics.precision_at_1 = precision_at_k
                    metrics.recall_at_1 = recall_at_k
                elif k == 3:
                    metrics.precision_at_3 = precision_at_k
                    metrics.recall_at_3 = recall_at_k
                elif k == 5:
                    metrics.precision_at_5 = precision_at_k
                    metrics.recall_at_5 = recall_at_k
            
            # Calculate MRR (Mean Reciprocal Rank)
            metrics.mrr = self._calculate_mrr(retrieved_ids, relevant_set)
            
            # Calculate MAP (Mean Average Precision)
            metrics.map_score = self._calculate_map(retrieved_ids, relevant_set)
            
            # Calculate NDCG@5
            metrics.ndcg_at_5 = self._calculate_ndcg(retrieved_docs, relevant_set, k=5)
            
            logger.info(f"Retrieval evaluation completed - P@3: {metrics.precision_at_3:.3f}, R@3: {metrics.recall_at_3:.3f}, MRR: {metrics.mrr:.3f}")
            
        except Exception as e:
            logger.error(f"Error in retrieval evaluation: {str(e)}")
        
        return metrics

    def evaluate_answer_quality(self, 
                               generated_answer: str, 
                               expected_answer: str,
                               query: str = "") -> AnswerQualityMetrics:
        """
        Evaluate answer quality using multiple metrics.
        
        Args:
            generated_answer: Generated response from the system
            expected_answer: Expected/reference answer
            query: Original query for context
            
        Returns:
            AnswerQualityMetrics object with calculated metrics
        """
        metrics = AnswerQualityMetrics()
        
        if not generated_answer or not expected_answer:
            logger.warning("Empty generated or expected answer for evaluation")
            return metrics
        
        try:
            # Exact match
            metrics.exact_match = 1.0 if generated_answer.strip().lower() == expected_answer.strip().lower() else 0.0
            
            # Partial match (token overlap)
            metrics.partial_match = self._calculate_token_overlap(generated_answer, expected_answer)
            
            # Factual accuracy (financial entities matching)
            metrics.factual_accuracy = self._evaluate_factual_accuracy(generated_answer, expected_answer)
            
            # Completeness
            metrics.completeness = self._evaluate_completeness(generated_answer, expected_answer)
            
            # Relevance to query
            metrics.relevance = self._evaluate_relevance(generated_answer, query) if query else 0.8
            
            # Fluency (basic heuristics)
            metrics.fluency = self._evaluate_fluency(generated_answer)
            
            # BLEU score (simplified implementation)
            metrics.bleu_score = self._calculate_bleu(generated_answer, expected_answer)
            
            # ROUGE scores (simplified implementation)
            rouge_scores = self._calculate_rouge(generated_answer, expected_answer)
            metrics.rouge_1 = rouge_scores.get('rouge_1', 0.0)
            metrics.rouge_2 = rouge_scores.get('rouge_2', 0.0)
            metrics.rouge_l = rouge_scores.get('rouge_l', 0.0)
            
            logger.info(f"Answer quality evaluation completed - Accuracy: {metrics.factual_accuracy:.3f}, Completeness: {metrics.completeness:.3f}")
            
        except Exception as e:
            logger.error(f"Error in answer quality evaluation: {str(e)}")
        
        return metrics

    def evaluate_end_to_end(self, 
                           query: str,
                           generated_answer: str,
                           response_time: float,
                           user_feedback: Optional[Dict[str, Any]] = None) -> EndToEndMetrics:
        """
        Evaluate end-to-end system performance.
        
        Args:
            query: User query
            generated_answer: System response
            response_time: Time taken to generate response
            user_feedback: Optional user feedback data
            
        Returns:
            EndToEndMetrics object with calculated metrics
        """
        metrics = EndToEndMetrics()
        
        try:
            # Response time score (faster is better, with reasonable thresholds)
            metrics.response_time = response_time
            time_score = max(0, 1 - (response_time - 3) / 10) if response_time > 3 else 1.0
            
            # Confidence score based on answer characteristics
            metrics.confidence_score = self._calculate_confidence_score(generated_answer)
            
            # Task completion (heuristic based on answer length and completeness)
            metrics.task_completion = self._evaluate_task_completion(query, generated_answer)
            
            # Overall quality (weighted combination)
            metrics.overall_quality = (
                0.4 * metrics.confidence_score +
                0.3 * metrics.task_completion +
                0.2 * time_score +
                0.1 * (1.0 if len(generated_answer) > 20 else 0.5)
            )
            
            # User satisfaction (from feedback or heuristic)
            if user_feedback:
                metrics.user_satisfaction = user_feedback.get('satisfaction', 0.7)
            else:
                metrics.user_satisfaction = metrics.overall_quality * 0.9  # Heuristic
            
            logger.info(f"End-to-end evaluation completed - Overall: {metrics.overall_quality:.3f}, Time: {response_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in end-to-end evaluation: {str(e)}")
        
        return metrics

    def comprehensive_evaluation(self,
                               query: str,
                               generated_answer: str,
                               expected_answer: str,
                               retrieved_docs: List[Dict[str, Any]],
                               relevant_docs: List[str],
                               response_time: float,
                               query_id: str = "",
                               metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Perform comprehensive evaluation combining all metrics.
        
        Returns:
            EvaluationResult with all evaluation metrics
        """
        try:
            retrieval_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs)
            answer_metrics = self.evaluate_answer_quality(generated_answer, expected_answer, query)
            end_to_end_metrics = self.evaluate_end_to_end(query, generated_answer, response_time)
            
            from datetime import datetime
            
            result = EvaluationResult(
                query_id=query_id or f"query_{hash(query) % 10000}",
                query=query,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
                retrieval_metrics=retrieval_metrics,
                answer_metrics=answer_metrics,
                end_to_end_metrics=end_to_end_metrics,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            logger.info(f"Comprehensive evaluation completed for query: {query[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {str(e)}")
            # Return empty result
            return EvaluationResult(
                query_id=query_id,
                query=query,
                expected_answer=expected_answer,
                generated_answer=generated_answer,
                retrieved_docs=[],
                relevant_docs=[],
                retrieval_metrics=RetrievalMetrics(),
                answer_metrics=AnswerQualityMetrics(),
                end_to_end_metrics=EndToEndMetrics(),
                timestamp="",
                metadata={}
            )

    def aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation results across multiple queries."""
        if not results:
            return {}
        
        try:
            # Aggregate retrieval metrics
            retrieval_agg = {
                'precision_at_1': sum(r.retrieval_metrics.precision_at_1 for r in results) / len(results),
                'precision_at_3': sum(r.retrieval_metrics.precision_at_3 for r in results) / len(results),
                'precision_at_5': sum(r.retrieval_metrics.precision_at_5 for r in results) / len(results),
                'recall_at_1': sum(r.retrieval_metrics.recall_at_1 for r in results) / len(results),
                'recall_at_3': sum(r.retrieval_metrics.recall_at_3 for r in results) / len(results),
                'recall_at_5': sum(r.retrieval_metrics.recall_at_5 for r in results) / len(results),
                'mrr': sum(r.retrieval_metrics.mrr for r in results) / len(results),
                'map_score': sum(r.retrieval_metrics.map_score for r in results) / len(results),
                'ndcg_at_5': sum(r.retrieval_metrics.ndcg_at_5 for r in results) / len(results)
            }
            
            # Aggregate answer quality metrics
            answer_agg = {
                'exact_match': sum(r.answer_metrics.exact_match for r in results) / len(results),
                'partial_match': sum(r.answer_metrics.partial_match for r in results) / len(results),
                'factual_accuracy': sum(r.answer_metrics.factual_accuracy for r in results) / len(results),
                'completeness': sum(r.answer_metrics.completeness for r in results) / len(results),
                'relevance': sum(r.answer_metrics.relevance for r in results) / len(results),
                'fluency': sum(r.answer_metrics.fluency for r in results) / len(results),
                'bleu_score': sum(r.answer_metrics.bleu_score for r in results) / len(results),
                'rouge_1': sum(r.answer_metrics.rouge_1 for r in results) / len(results),
                'rouge_2': sum(r.answer_metrics.rouge_2 for r in results) / len(results),
                'rouge_l': sum(r.answer_metrics.rouge_l for r in results) / len(results)
            }
            
            # Aggregate end-to-end metrics
            end_to_end_agg = {
                'overall_quality': sum(r.end_to_end_metrics.overall_quality for r in results) / len(results),
                'user_satisfaction': sum(r.end_to_end_metrics.user_satisfaction for r in results) / len(results),
                'task_completion': sum(r.end_to_end_metrics.task_completion for r in results) / len(results),
                'avg_response_time': sum(r.end_to_end_metrics.response_time for r in results) / len(results),
                'confidence_score': sum(r.end_to_end_metrics.confidence_score for r in results) / len(results)
            }
            
            return {
                'total_queries': len(results),
                'retrieval_metrics': retrieval_agg,
                'answer_quality_metrics': answer_agg,
                'end_to_end_metrics': end_to_end_agg,
                'query_types': self._analyze_query_types(results),
                'performance_distribution': self._analyze_performance_distribution(results)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating results: {str(e)}")
            return {'error': str(e)}

    # Helper methods for metric calculations
    
    def _calculate_mrr(self, retrieved_ids: List[str], relevant_set: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_map(self, retrieved_ids: List[str], relevant_set: set) -> float:
        """Calculate Mean Average Precision."""
        if not relevant_set:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_set) if len(relevant_set) > 0 else 0.0

    def _calculate_ndcg(self, retrieved_docs: List[Dict[str, Any]], relevant_set: set, k: int = 5) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        try:
            dcg = 0.0
            for i, doc in enumerate(retrieved_docs[:k]):
                if doc.get('source') in relevant_set:
                    relevance = 1  # Binary relevance
                    dcg += relevance / math.log2(i + 2)
            
            # Ideal DCG (if all relevant docs were at the top)
            ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
            
            return dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_token_overlap(self, text1: str, text2: str) -> float:
        """Calculate token overlap between two texts."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union) if union else 0.0

    def _evaluate_factual_accuracy(self, generated: str, expected: str) -> float:
        """Evaluate factual accuracy by matching financial entities."""
        score = 0.0
        total_checks = 0
        
        for entity_type, entities in self.financial_entities.items():
            for entity in entities:
                if entity.lower() in expected.lower():
                    total_checks += 1
                    if entity.lower() in generated.lower():
                        score += 1
        
        return score / total_checks if total_checks > 0 else 0.8

    def _evaluate_completeness(self, generated: str, expected: str) -> float:
        """Evaluate completeness of the answer."""
        # Simple heuristic: ratio of key information covered
        expected_parts = self._extract_key_information(expected)
        generated_parts = self._extract_key_information(generated)
        
        if not expected_parts:
            return 0.8
        
        matched_parts = sum(1 for part in expected_parts if any(part.lower() in gen.lower() for gen in generated_parts))
        return matched_parts / len(expected_parts)

    def _evaluate_relevance(self, generated: str, query: str) -> float:
        """Evaluate relevance of answer to query."""
        if not query:
            return 0.8
        
        query_terms = set(query.lower().split())
        answer_terms = set(generated.lower().split())
        
        overlap = query_terms & answer_terms
        return len(overlap) / len(query_terms) if query_terms else 0.8

    def _evaluate_fluency(self, text: str) -> float:
        """Evaluate fluency using simple heuristics."""
        if not text:
            return 0.0
        
        # Basic fluency indicators
        sentence_count = len(re.findall(r'[.!?]+', text))
        word_count = len(text.split())
        
        # Penalty for very short or very long sentences
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        length_score = 1.0 if 10 <= avg_sentence_length <= 25 else 0.8
        
        # Check for repeated words
        words = text.lower().split()
        unique_words = set(words)
        repetition_score = len(unique_words) / len(words) if words else 0
        
        return (length_score + repetition_score) / 2

    def _calculate_bleu(self, generated: str, expected: str) -> float:
        """Simplified BLEU score calculation."""
        try:
            gen_tokens = generated.lower().split()
            exp_tokens = expected.lower().split()
            
            if not gen_tokens or not exp_tokens:
                return 0.0
            
            # 1-gram precision
            gen_1gram = set(gen_tokens)
            exp_1gram = set(exp_tokens)
            precision_1 = len(gen_1gram & exp_1gram) / len(gen_1gram) if gen_1gram else 0
            
            # Simple approximation (full BLEU would include n-grams and brevity penalty)
            return precision_1
            
        except Exception:
            return 0.0

    def _calculate_rouge(self, generated: str, expected: str) -> Dict[str, float]:
        """Simplified ROUGE score calculation."""
        try:
            gen_tokens = generated.lower().split()
            exp_tokens = expected.lower().split()
            
            if not gen_tokens or not exp_tokens:
                return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
            
            # ROUGE-1 (unigram overlap)
            gen_1gram = set(gen_tokens)
            exp_1gram = set(exp_tokens)
            rouge_1 = len(gen_1gram & exp_1gram) / len(exp_1gram) if exp_1gram else 0
            
            # ROUGE-2 (bigram overlap)
            gen_2gram = set(zip(gen_tokens[:-1], gen_tokens[1:]))
            exp_2gram = set(zip(exp_tokens[:-1], exp_tokens[1:]))
            rouge_2 = len(gen_2gram & exp_2gram) / len(exp_2gram) if exp_2gram else 0
            
            # ROUGE-L (longest common subsequence - simplified)
            rouge_l = self._lcs_length(gen_tokens, exp_tokens) / len(exp_tokens) if exp_tokens else 0
            
            return {
                'rouge_1': rouge_1,
                'rouge_2': rouge_2,
                'rouge_l': rouge_l
            }
            
        except Exception:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

    def _calculate_confidence_score(self, answer: str) -> float:
        """Calculate confidence score based on answer characteristics."""
        if not answer:
            return 0.0
        
        # Factors affecting confidence
        length_score = min(len(answer.split()) / 50, 1.0)  # Longer answers generally more confident
        specificity_score = len(re.findall(r'\d+', answer)) * 0.1  # Numbers indicate specificity
        certainty_score = 0.8 if any(word in answer.lower() for word in ['is', 'was', 'were', 'total']) else 0.6
        
        return min((length_score + specificity_score + certainty_score) / 3, 1.0)

    def _evaluate_task_completion(self, query: str, answer: str) -> float:
        """Evaluate how well the answer completes the task implied by the query."""
        if not query or not answer:
            return 0.0
        
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Check if question type is addressed
        if query_lower.startswith('what') and any(word in answer_lower for word in ['is', 'was', 'total']):
            return 1.0
        elif query_lower.startswith('how') and any(word in answer_lower for word in ['by', 'through', 'via']):
            return 1.0
        elif 'compare' in query_lower and any(word in answer_lower for word in ['vs', 'compared', 'versus', 'than']):
            return 1.0
        elif 'summarize' in query_lower and len(answer.split()) > 20:
            return 1.0
        else:
            return 0.7  # Default for partially complete

    def _extract_key_information(self, text: str) -> List[str]:
        """Extract key information pieces from text."""
        # Extract numbers, percentages, and financial terms
        key_info = []
        
        # Extract monetary values
        money_pattern = r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million))?'
        key_info.extend(re.findall(money_pattern, text, re.IGNORECASE))
        
        # Extract percentages
        percent_pattern = r'\d+(?:\.\d+)?%'
        key_info.extend(re.findall(percent_pattern, text))
        
        # Extract quarters and years
        period_pattern = r'Q[1-4]\s*20\d{2}'
        key_info.extend(re.findall(period_pattern, text, re.IGNORECASE))
        
        return key_info

    def _analyze_query_types(self, results: List[EvaluationResult]) -> Dict[str, int]:
        """Analyze distribution of query types."""
        query_types = defaultdict(int)
        
        for result in results:
            query = result.query.lower()
            if 'what' in query:
                query_types['factual'] += 1
            elif 'compare' in query or 'vs' in query:
                query_types['comparison'] += 1
            elif 'summarize' in query or 'summary' in query:
                query_types['summary'] += 1
            elif 'how' in query:
                query_types['procedural'] += 1
            else:
                query_types['other'] += 1
        
        return dict(query_types)

    def _analyze_performance_distribution(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance distribution across results."""
        overall_scores = [r.end_to_end_metrics.overall_quality for r in results]
        
        if not overall_scores:
            return {}
        
        return {
            'min_score': min(overall_scores),
            'max_score': max(overall_scores),
            'avg_score': sum(overall_scores) / len(overall_scores),
            'score_distribution': {
                'excellent': sum(1 for s in overall_scores if s >= 0.9),
                'good': sum(1 for s in overall_scores if 0.7 <= s < 0.9),
                'fair': sum(1 for s in overall_scores if 0.5 <= s < 0.7),
                'poor': sum(1 for s in overall_scores if s < 0.5)
            }
        }
