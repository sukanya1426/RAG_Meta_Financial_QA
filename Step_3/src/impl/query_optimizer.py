"""
Query Optimizer for Step 3: Enhances user queries for better retrieval performance.
Implements query rewriting, expansion, and decomposition strategies.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..util.invoke_ai import invoke_ai
import json

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """Optimizes queries through rewriting, expansion, and decomposition."""
    
    def __init__(self):
        self.financial_terms = {
            'revenue': ['revenue', 'total revenue', 'net revenue', 'sales', 'income'],
            'profit': ['profit', 'net income', 'earnings', 'net profit'],
            'expenses': ['expenses', 'costs', 'operating expenses', 'expenditures'],
            'margin': ['margin', 'operating margin', 'profit margin', 'gross margin'],
            'growth': ['growth', 'increase', 'year-over-year', 'yoy', 'compared to'],
            'quarterly': ['Q1', 'Q2', 'Q3', 'Q4', 'quarter', 'quarterly'],
            'metrics': ['DAP', 'MAU', 'ARPU', 'EPS', 'earnings per share']
        }
        
        self.query_templates = {
            'comparison': "Compare {metric} between {period1} and {period2}",
            'trend': "What is the trend of {metric} over {timeframe}?",
            'breakdown': "Provide a detailed breakdown of {category}",
            'financial': "What was Meta's {metric} in {period}?"
        }

    def optimize_query(self, query: str, optimization_type: str = "auto") -> Dict[str, Any]:
        """
        Optimize a query using various strategies.
        
        Args:
            query: Original user query
            optimization_type: Type of optimization ('rewrite', 'expand', 'decompose', 'auto')
            
        Returns:
            Dictionary containing optimized queries and metadata
        """
        logger.info(f"Optimizing query: {query}")
        
        result = {
            'original_query': query,
            'optimized_queries': [],
            'query_type': self._classify_query(query),
            'optimization_strategy': optimization_type,
            'metadata': {}
        }
        
        try:
            if optimization_type == "auto":
                # Automatically determine best optimization strategy
                optimization_type = self._determine_optimization_strategy(query)
                result['optimization_strategy'] = optimization_type
            
            if optimization_type == "rewrite":
                optimized = self._rewrite_query(query)
                result['optimized_queries'] = [optimized]
                
            elif optimization_type == "expand":
                expanded = self._expand_query(query)
                result['optimized_queries'] = expanded
                
            elif optimization_type == "decompose":
                decomposed = self._decompose_query(query)
                result['optimized_queries'] = decomposed
                
            else:  # multi-strategy
                rewritten = self._rewrite_query(query)
                expanded = self._expand_query(rewritten)
                result['optimized_queries'] = [rewritten] + expanded
            
            # Add query enrichment
            result['enriched_terms'] = self._enrich_with_financial_terms(query)
            result['metadata']['num_optimized'] = len(result['optimized_queries'])
            
            logger.info(f"Generated {len(result['optimized_queries'])} optimized queries")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing query: {str(e)}")
            result['optimized_queries'] = [query]  # Fallback to original
            return result

    def _classify_query(self, query: str) -> str:
        """Classify query type for targeted optimization."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'compared to']):
            return 'comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'change', 'growth']):
            return 'trend'
        elif any(word in query_lower for word in ['breakdown', 'detailed', 'components', 'categories']):
            return 'breakdown'
        elif any(word in query_lower for word in ['what', 'how much', 'total', 'amount']):
            return 'factual'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'explain']):
            return 'explanatory'
        else:
            return 'general'

    def _determine_optimization_strategy(self, query: str) -> str:
        """Determine the best optimization strategy based on query characteristics."""
        query_type = self._classify_query(query)
        query_length = len(query.split())
        
        if query_type == 'comparison' and query_length > 10:
            return 'decompose'
        elif query_type in ['factual', 'general'] and query_length < 5:
            return 'expand'
        elif any(term in query.lower() for term in ['simple', 'basic', 'quick']):
            return 'rewrite'
        else:
            return 'expand'

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query for better clarity and specificity."""
        try:
            system_message = "You are a financial query optimization assistant. Rewrite queries to be more specific and clear for document retrieval."
            
            user_message = f"""
            Rewrite the following financial query to be more specific and clear for document retrieval:
            
            Original Query: "{query}"
            
            Instructions:
            1. Make the query more specific by adding financial context
            2. Include relevant time periods if missing
            3. Use precise financial terminology
            4. Keep the core intent unchanged
            5. Return only the rewritten query, no explanation
            
            Rewritten Query:
            """
            
            response = invoke_ai(system_message, user_message)
            rewritten = response.strip().strip('"').strip("'")
            
            # Validate rewritten query
            if len(rewritten) > 10 and rewritten.lower() != query.lower():
                logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
                return rewritten
            else:
                return query
                
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related terms and variations."""
        expanded_queries = [query]
        
        try:
            # Add financial term expansions
            for category, terms in self.financial_terms.items():
                for term in terms:
                    if term.lower() in query.lower():
                        # Create variations with synonyms
                        for synonym in terms:
                            if synonym != term:
                                expanded_query = query.lower().replace(term.lower(), synonym)
                                if expanded_query != query.lower():
                                    expanded_queries.append(expanded_query.capitalize())
            
            # Add temporal variations
            expanded_queries.extend(self._add_temporal_variations(query))
            
            # Add context-specific expansions
            expanded_queries.extend(self._add_context_expansions(query))
            
            # Remove duplicates while preserving order
            seen = set()
            unique_queries = []
            for q in expanded_queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
            
            logger.info(f"Expanded query into {len(unique_queries)} variations")
            return unique_queries[:5]  # Limit to top 5 variations
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return [query]

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into simpler sub-queries."""
        try:
            system_message = "You are a financial query decomposition assistant. Break down complex queries into simpler, focused sub-queries."
            
            user_message = f"""
            Decompose the following complex financial query into 2-3 simpler, focused sub-queries:
            
            Complex Query: "{query}"
            
            Instructions:
            1. Break down into specific, answerable sub-questions
            2. Each sub-query should focus on one aspect
            3. Ensure sub-queries together answer the original question
            4. Use financial terminology appropriately
            5. Return as a JSON list of strings
            
            Example format: ["Sub-query 1", "Sub-query 2", "Sub-query 3"]
            
            Sub-queries:
            """
            
            response = invoke_ai(system_message, user_message)
            
            # Parse JSON response
            try:
                sub_queries = json.loads(response.strip())
                if isinstance(sub_queries, list) and len(sub_queries) > 1:
                    logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
                    return sub_queries
            except json.JSONDecodeError:
                # Fallback: split by common separators
                if any(sep in query for sep in [' and ', ' or ', ', ']):
                    parts = re.split(r'\s+(?:and|or)\s+|,\s+', query)
                    if len(parts) > 1:
                        return [part.strip() for part in parts if len(part.strip()) > 5]
            
            return [query]
            
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}")
            return [query]

    def _add_temporal_variations(self, query: str) -> List[str]:
        """Add temporal variations to the query."""
        variations = []
        query_lower = query.lower()
        
        # Add period-specific versions
        periods = ['Q1 2024', 'Q1 2023', 'first quarter 2024', 'Q4 2023']
        if not any(period.lower() in query_lower for period in periods):
            for period in periods[:2]:  # Limit to most relevant
                variations.append(f"{query} in {period}")
        
        # Add comparison versions
        if 'compare' not in query_lower and 'vs' not in query_lower:
            if any(term in query_lower for term in ['revenue', 'income', 'profit']):
                variations.append(f"Compare {query} between Q1 2024 and Q1 2023")
        
        return variations

    def _add_context_expansions(self, query: str) -> List[str]:
        """Add context-specific expansions based on Meta's business."""
        expansions = []
        query_lower = query.lower()
        
        # Add Meta-specific context
        if 'meta' not in query_lower:
            expansions.append(f"Meta's {query}")
        
        # Add business segment context
        if any(term in query_lower for term in ['revenue', 'income']) and 'family of apps' not in query_lower:
            expansions.append(f"{query} from Family of Apps")
            expansions.append(f"{query} from Reality Labs")
        
        # Add metric context
        if any(term in query_lower for term in ['users', 'people']) and 'daily active' not in query_lower:
            expansions.append(f"Daily active people for {query}")
        
        return expansions

    def _enrich_with_financial_terms(self, query: str) -> List[str]:
        """Enrich query with relevant financial terms for better matching."""
        enriched_terms = []
        query_lower = query.lower()
        
        for category, terms in self.financial_terms.items():
            if any(term in query_lower for term in terms):
                enriched_terms.extend(terms)
        
        # Add common financial abbreviations
        abbreviations = {
            'revenue': ['rev', 'sales'],
            'profit': ['net income', 'earnings'],
            'expenses': ['opex', 'costs'],
            'margin': ['operating margin', 'profit margin']
        }
        
        for key, abbrevs in abbreviations.items():
            if key in query_lower:
                enriched_terms.extend(abbrevs)
        
        return list(set(enriched_terms))

    def get_query_insights(self, query: str) -> Dict[str, Any]:
        """Analyze query to provide insights for optimization."""
        return {
            'query_type': self._classify_query(query),
            'complexity_score': self._calculate_complexity(query),
            'financial_terms_detected': self._enrich_with_financial_terms(query),
            'suggested_optimization': self._determine_optimization_strategy(query),
            'potential_ambiguities': self._detect_ambiguities(query)
        }

    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0-1)."""
        factors = {
            'length': min(len(query.split()) / 20, 1.0),  # Longer queries are more complex
            'conjunctions': len(re.findall(r'\b(and|or|but|however|while)\b', query.lower())) * 0.2,
            'questions': len(re.findall(r'\?', query)) * 0.1,
            'comparisons': len(re.findall(r'\b(vs|versus|compare|compared to)\b', query.lower())) * 0.3,
            'temporal': len(re.findall(r'\b(Q[1-4]|quarter|year|2024|2023)\b', query)) * 0.1
        }
        return min(sum(factors.values()), 1.0)

    def _detect_ambiguities(self, query: str) -> List[str]:
        """Detect potential ambiguities in the query."""
        ambiguities = []
        query_lower = query.lower()
        
        # Check for pronoun ambiguities
        if any(pronoun in query_lower for pronoun in ['it', 'they', 'this', 'that']):
            ambiguities.append("Contains pronouns that may be ambiguous")
        
        # Check for missing time context
        if not any(time in query_lower for time in ['2024', '2023', 'q1', 'q2', 'q3', 'q4', 'quarter']):
            ambiguities.append("Missing specific time period")
        
        # Check for vague terms
        vague_terms = ['good', 'bad', 'high', 'low', 'many', 'few', 'recent']
        if any(term in query_lower for term in vague_terms):
            ambiguities.append("Contains subjective or vague terms")
        
        return ambiguities
