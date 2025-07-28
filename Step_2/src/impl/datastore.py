from typing import List, Dict, Any, Union
from ..interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import shutil
import json
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Datastore(BaseDatastore):
    DB_PATH = "data/sample-lancedb"
    DB_TABLE_NAME = "rag-table"

    def __init__(self):
        self.vector_dimensions = 384  # Dimension for all-MiniLM-L6-v2
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = self.embedding_model.tokenizer
        self.vector_db = lancedb.connect(self.DB_PATH)
        self.table: Table = self._get_table()

    def reset(self) -> Table:
        try:
            table_dir = os.path.join(self.DB_PATH, f"{self.DB_TABLE_NAME}.lance")
            if os.path.exists(table_dir):
                shutil.rmtree(table_dir)
                logger.info(f"Removed existing table directory: {table_dir}")
            else:
                logger.info(f"No existing table directory found: {table_dir}")
        except Exception as e:
            logger.error(f"Error removing table directory: {str(e)}")
            raise
        
        os.makedirs(self.DB_PATH, exist_ok=True)
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.vector_dimensions)),
            pa.field("content", pa.utf8()),
            pa.field("source", pa.utf8()),
            pa.field("content_type", pa.utf8()),
            pa.field("structured_data", pa.utf8()),  # JSON string for structured data
            pa.field("metadata", pa.utf8()),  # JSON string for metadata
        ])
        self.vector_db.create_table(self.DB_TABLE_NAME, schema=schema)
        logger.info(f"✅ Table Reset/Created: {self.DB_TABLE_NAME} in {self.DB_PATH}")
        return self.vector_db.open_table(self.DB_TABLE_NAME)

    def get_vector(self, content: str) -> List[float]:
        try:
            # Truncate to 512 tokens to avoid exceeding model limit
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) > 512:
                logger.warning(f"Content exceeds 512 tokens, truncating to 512 tokens")
                tokens = tokens[:512]
                content = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            embedding = self.embedding_model.encode(content).tolist()
            if len(embedding) != self.vector_dimensions:
                logger.error(f"Invalid embedding dimension: {len(embedding)} != {self.vector_dimensions}")
                raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self.vector_dimensions}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding for content: {str(e)}")
            raise

    def add_items(self, items: List[DataItem]) -> None:
        if not items:
            logger.warning("No items provided to add to datastore")
            return
        
        logger.info(f"Adding {len(items)} items to datastore")
        entries = []
        
        for i, item in enumerate(items):
            try:
                if not item.content or not isinstance(item.content, str):
                    logger.warning(f"Skipping item {i}: Invalid or empty content")
                    continue
                
                if item.embedding is None:
                    item.embedding = self.get_vector(item.content)
                
                if len(item.embedding) != self.vector_dimensions:
                    logger.error(f"Invalid embedding dimension for item {i}: {len(item.embedding)}")
                    continue
                
                # Determine content type
                content_type = "text"
                structured_data_json = ""
                metadata_json = ""
                
                if item.metadata:
                    metadata_json = json.dumps(item.metadata)
                    content_type = item.metadata.get('type', 'text')
                    
                    if content_type == 'table' and 'structured_data' in item.metadata:
                        structured_data_json = json.dumps(item.metadata['structured_data'])
                
                entries.append({
                    "vector": item.embedding,
                    "content": item.content,
                    "source": item.source,
                    "content_type": content_type,
                    "structured_data": structured_data_json,
                    "metadata": metadata_json,
                })
                logger.debug(f"Prepared item {i}: {item.content[:50]}...")
                
            except Exception as e:
                logger.error(f"Error preparing item {i}: {str(e)}")
                continue
        
        if not entries:
            logger.error("No valid entries to insert into datastore")
            return

        try:
            if not self.table:
                logger.warning("Table is invalid, resetting datastore")
                self.table = self.reset()
            
            self.table.merge_insert("source")\
                     .when_matched_update_all()\
                     .when_not_matched_insert_all()\
                     .execute(entries)
            logger.info(f"✅ Added {len(entries)} items to the datastore")
            
        except Exception as e:
            logger.error(f"Error inserting items into datastore: {str(e)}")
            if "Not found" in str(e) or "IO" in str(e):
                logger.info("Attempting to reset datastore due to I/O error")
                self.table = self.reset()
                try:
                    self.table.merge_insert("source")\
                             .when_matched_update_all()\
                             .when_not_matched_insert_all()\
                             .execute(entries)
                    logger.info(f"✅ Successfully added {len(entries)} items after reset")
                except Exception as e2:
                    logger.error(f"Retry failed: {str(e2)}")
                    raise
            else:
                raise

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Standard vector search for text similarity."""
        try:
            vector = self.get_vector(query)
            results = (
                self.table.search(vector)
                .metric("cosine")
                .select(["content", "source", "content_type", "structured_data", "metadata"])
                .limit(top_k)
                .to_list()
            )
            
            logger.info(f"Vector search returned {len(results)} results for query: {query[:50]}...")
            return [result.get("content") for result in results]
            
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Hybrid search combining vector search and structured data queries."""
        try:
            # Vector search for semantic similarity
            vector = self.get_vector(query)
            vector_results = (
                self.table.search(vector)
                .metric("cosine")
                .select(["content", "source", "content_type", "structured_data", "metadata"])
                .limit(top_k)
                .to_list()
            )
            
            # Keyword-based search for structured data
            structured_results = self._search_structured_data(query)
            
            # Financial data specific search
            financial_results = self._search_financial_data(query)
            
            logger.info(f"Hybrid search - Vector: {len(vector_results)}, Structured: {len(structured_results)}, Financial: {len(financial_results)}")
            
            return {
                'text_context': [result.get("content") for result in vector_results],
                'structured_data': structured_results,
                'financial_data': financial_results,
                'all_results': vector_results
            }
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            return {
                'text_context': [],
                'structured_data': [],
                'financial_data': [],
                'all_results': []
            }

    def _search_structured_data(self, query: str) -> List[Dict[str, Any]]:
        """Search specifically in structured table data."""
        try:
            # Get all table entries
            all_results = self.table.search().where("content_type = 'table'").to_list()
            
            matching_tables = []
            query_lower = query.lower()
            
            for result in all_results:
                structured_data_str = result.get('structured_data', '')
                if structured_data_str:
                    try:
                        structured_data = json.loads(structured_data_str)
                        # Search in table data
                        if self._matches_query(structured_data, query_lower):
                            matching_tables.append({
                                'source': result.get('source'),
                                'data': structured_data,
                                'content': result.get('content')
                            })
                    except json.JSONDecodeError:
                        continue
            
            return matching_tables[:5]  # Limit results
            
        except Exception as e:
            logger.error(f"Error searching structured data: {str(e)}")
            return []

    def _search_financial_data(self, query: str) -> List[Dict[str, Any]]:
        """Search for specific financial metrics and comparisons."""
        try:
            financial_keywords = {
                'revenue': ['revenue', 'income', 'sales'],
                'net_income': ['net income', 'profit', 'earnings'],
                'expenses': ['expenses', 'costs', 'operating expenses'],
                'margin': ['margin', 'operating margin'],
                'eps': ['eps', 'earnings per share'],
                'dap': ['dap', 'daily active people']
            }
            
            query_lower = query.lower()
            relevant_metrics = []
            
            # Find relevant financial keywords
            for metric, keywords in financial_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    relevant_metrics.append(metric)
            
            if not relevant_metrics:
                return []
            
            # Search for financial data
            all_results = self.table.search().to_list()
            financial_data = []
            
            for result in all_results:
                content = result.get('content', '').lower()
                structured_data_str = result.get('structured_data', '')
                
                # Check if content contains financial metrics
                if any(metric.replace('_', ' ') in content for metric in relevant_metrics):
                    entry = {
                        'source': result.get('source'),
                        'content': result.get('content'),
                        'type': 'financial_text'
                    }
                    
                    if structured_data_str:
                        try:
                            entry['structured_data'] = json.loads(structured_data_str)
                            entry['type'] = 'financial_table'
                        except json.JSONDecodeError:
                            pass
                    
                    financial_data.append(entry)
            
            return financial_data[:5]
            
        except Exception as e:
            logger.error(f"Error searching financial data: {str(e)}")
            return []

    def _matches_query(self, structured_data: List[Dict[str, Any]], query: str) -> bool:
        """Check if structured data matches the query."""
        try:
            for row in structured_data:
                for key, value in row.items():
                    if isinstance(value, str) and query in key.lower():
                        return True
                    if isinstance(value, str) and query in value.lower():
                        return True
                    # Check for financial comparisons (Q1 2024 vs Q1 2023)
                    if 'q1 2024' in query and 'q1 2023' in query:
                        if '2024' in str(value) or '2023' in str(value):
                            return True
            return False
        except Exception:
            return False

    def _get_table(self) -> Table:
        try:
            table = self.vector_db.open_table(self.DB_TABLE_NAME)
            logger.info(f"Opened existing table: {self.DB_TABLE_NAME}")
            return table
        except Exception as e:
            logger.info(f"Table not found, resetting datastore: {e}")
            return self.reset()