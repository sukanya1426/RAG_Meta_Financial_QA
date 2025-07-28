"""
API Quota Management Configuration for Step 3 RAG System
"""

import os
from typing import Dict, Any

class QuotaConfig:
    """Configuration for managing API quotas and reducing usage"""
    
    def __init__(self):
        # Gemini API limits (conservative estimates)
        self.requests_per_minute = 10
        self.requests_per_day = 1000
        
        # Evaluation settings to reduce API usage
        self.evaluation_settings = {
            # Reduce number of queries in comprehensive evaluation
            "max_queries_per_eval": 5,  # Down from 15
            
            # Skip certain evaluation components when quota is low
            "skip_query_optimization": False,
            "skip_chunk_size_benchmarking": True,  # This uses a lot of API calls
            "skip_ablation_studies": True,
            
            # Use simpler evaluation methods
            "use_simple_evaluation": True,
            "batch_size": 1,  # Process one query at a time
            
            # Reduce query optimization attempts
            "max_optimization_variants": 2,  # Down from 5
            "optimization_strategy": "simple",  # Use simpler optimization
        }
        
        # Fallback responses for quota exhaustion
        self.fallback_responses = {
            "query_optimization": "What was Meta's total revenue in Q1 2024?",
            "response_generation": "I apologize, but I cannot generate a response due to API quota limitations. Please try again later.",
        }
    
    def get_reduced_evaluation_config(self) -> Dict[str, Any]:
        """Get configuration for reduced evaluation when quota is limited"""
        return {
            "test_queries": [
                "What was Meta's total revenue in Q1 2024?",
                "How many daily active people did Meta have in Q1 2024?",
                "What was Meta's net income in Q1 2024?",
            ],
            "chunk_sizes": [400],  # Test only one chunk size
            "skip_advanced_features": True,
            "use_cached_responses": True,
        }
    
    def should_skip_component(self, component: str) -> bool:
        """Check if a component should be skipped to save quota"""
        skip_settings = {
            "query_optimization": self.evaluation_settings.get("skip_query_optimization", False),
            "chunk_benchmarking": self.evaluation_settings.get("skip_chunk_size_benchmarking", True),
            "ablation_study": self.evaluation_settings.get("skip_ablation_studies", True),
        }
        return skip_settings.get(component, False)

# Global configuration instance
quota_config = QuotaConfig()
