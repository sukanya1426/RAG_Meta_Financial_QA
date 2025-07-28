from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional
from .interface import (
    BaseDatastore,
    BaseIndexer,
    BaseRetriever,
    BaseResponseGenerator,
    BaseEvaluator,
    EvaluationResult,
)
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None

    def reset(self) -> None:
        """Reset the datastore."""
        logger.info("Resetting the datastore...")
        self.datastore.reset()

    def add_documents(self, documents: List[str]) -> None:
        """Index a list of documents."""
        logger.info(f"Indexing {len(documents)} documents")
        items = self.indexer.index(documents)
        if not items:
            logger.warning("No items generated from documents")
            return
        self.datastore.add_items(items)
        logger.info(f"Added {len(items)} items to the datastore")

    def process_query(self, query: str) -> str:
        """Process a query using hybrid search for better structured data handling."""
        logger.info(f"Processing query: {query}")
        
        # Check if retriever supports hybrid search
        if hasattr(self.retriever, 'hybrid_search'):
            logger.info("Using hybrid search")
            search_results = self.retriever.hybrid_search(query)
            
            # Check if response generator supports hybrid responses
            if hasattr(self.response_generator, 'generate_hybrid_response'):
                response = self.response_generator.generate_hybrid_response(query, search_results)
            else:
                # Fallback to regular response generation with text context
                text_context = search_results.get('text_context', [])
                response = self.response_generator.generate_response(query, text_context)
        else:
            # Fallback to regular search
            logger.info("Using regular search")
            search_results = self.retriever.search(query)
            logger.info(f"Found {len(search_results)} results for query")
            for i, result in enumerate(search_results):
                logger.info(f"Result {i+1}: {result[:100]}...")
            response = self.response_generator.generate_response(query, search_results)
        
        logger.info(f"Generated response: {response[:100]}...")
        return response

    def evaluate(self, sample_questions: List[Dict[str, str]]) -> List[EvaluationResult]:
        logger.info(f"Evaluating {len(sample_questions)} questions")
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]

        with ThreadPoolExecutor(max_workers=10) as executor:
            results: List[EvaluationResult] = list(
                executor.map(self._evaluate_single_question, questions, expected_answers)
            )

        for i, result in enumerate(results):
            result_emoji = "✅" if result.is_correct else "❌"
            logger.info(f"{result_emoji} Q {i+1}: {result.question}")
            logger.info(f"Response: {result.response}")
            logger.info(f"Expected Answer: {result.expected_answer}")
            logger.info(f"Reasoning: {result.reasoning}")
            logger.info("--------------------------------")

        number_correct = sum(result.is_correct for result in results)
        logger.info(f"Total Score: {number_correct}/{len(results)}")
        return results

    def _evaluate_single_question(self, question: str, expected_answer: str) -> EvaluationResult:
        response = self.process_query(question)
        return self.evaluator.evaluate(question, response, expected_answer)