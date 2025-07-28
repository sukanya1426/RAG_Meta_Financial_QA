from typing import Dict, Any
from ..interface.base_datastore import BaseDatastore
from ..interface.base_retriever import BaseRetriever
import logging

logger = logging.getLogger(__name__)

class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Standard vector search."""
        return self.datastore.search(query, top_k=top_k)
    
    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform hybrid search combining vector and structured data search."""
        logger.info(f"Performing hybrid search for: {query}")
        
        # Check if datastore supports hybrid search
        if hasattr(self.datastore, 'hybrid_search'):
            return self.datastore.hybrid_search(query, top_k)
        else:
            # Fallback to regular search
            logger.warning("Datastore doesn't support hybrid search, using regular search")
            results = self.datastore.search(query, top_k)
            return {
                'text_context': results,
                'structured_data': [],
                'financial_data': [],
                'all_results': []
            }